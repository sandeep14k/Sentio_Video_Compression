import cv2
import json
import base64
import subprocess
import time
import numpy as np
from pathlib import Path
import os
import imagehash
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_IN               = Path("video_sample_1.mov")
VIDEO_OUT              = Path("compressed_output.mp4")
REPORT_HTML_OUT        = Path("compression_report.html")
SEGMENTS_JSON_OUT      = Path("segments_kept.json")

PHASH_THRESHOLD        = 0.95   
MOTION_KEEP_THRESH     = 0.15   
MOTION_DISCARD_THRESH  = 0.05   # This will auto-calibrate during the first 30s!
CONTEXT_EVERY_SEC      = 3      
OUTPUT_FPS             = 12     
OUTPUT_CRF             = 28     

# ---------------------------------------------------------------------------
# PERCEPTUAL HASH
# ---------------------------------------------------------------------------

def compute_phash(frame: np.ndarray) -> str:
    small_frame = cv2.resize(frame, (64, 64))
    img = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
    hash_obj = imagehash.phash(img, hash_size=8)
    return ''.join(['1' if val else '0' for val in hash_obj.hash.flatten()])

def phash_similarity(h1: str, h2: str) -> float:
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return 1.0 - (hamming_distance / len(h1))

# ---------------------------------------------------------------------------
# MOTION SCORE
# ---------------------------------------------------------------------------

def compute_motion_score(prev_gray, curr_gray: np.ndarray) -> float:
    if prev_gray is None or curr_gray is None:
        return 0.0
    h, w = curr_gray.shape
    scale = 100.0 / w  # SPEED HACK: Down to 100px
    new_w, new_h = 100, int(h * scale)
    pg_small = cv2.resize(prev_gray, (new_w, new_h))
    cg_small = cv2.resize(curr_gray, (new_w, new_h))
    flow = cv2.calcOpticalFlowFarneback(
        pg_small, cg_small, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

# ---------------------------------------------------------------------------
# FACE PRESENCE CHECK
# ---------------------------------------------------------------------------

def has_face(frame: np.ndarray, cascade) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    h, w = gray_eq.shape
    scale = 240.0 / w  # SPEED HACK: Down to 240px
    new_w, new_h = 240, int(h * scale)
    small_gray = cv2.resize(gray_eq, (new_w, new_h))
    faces = cascade.detectMultiScale(
        small_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    return len(faces) > 0

# ---------------------------------------------------------------------------
# FRAME KEEP DECISION
# ---------------------------------------------------------------------------

def should_keep_frame(frame: np.ndarray, prev_frame, prev_kept_hash: str,
                      last_kept_time_sec: float, current_time_sec: float, cascade) -> tuple:
    is_context = (current_time_sec - last_kept_time_sec) >= CONTEXT_EVERY_SEC
    
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_score = compute_motion_score(prev_gray, curr_gray)
    else:
        motion_score = 1.0  
        
    curr_hash = compute_phash(frame)
    sim = phash_similarity(curr_hash, prev_kept_hash) if prev_kept_hash else 0.0
    face_found = has_face(frame, cascade)
    
    keep = False
    reason = "discarded_static"
    discard_phash = (sim > PHASH_THRESHOLD)
    
    # Check against the globally dynamically calibrated threshold
    discard_motion = (motion_score < MOTION_DISCARD_THRESH)
    
    if not discard_phash and not discard_motion:
        keep = True
        reason = "motion_above_threshold"
    else:
        if discard_phash: reason = "discarded_duplicate"
        elif discard_motion: reason = "discarded_static"

    if motion_score > MOTION_KEEP_THRESH:
        keep = True
        reason = "motion_above_threshold"
    if face_found:
        keep = True
        reason = "face_and_motion" if motion_score > MOTION_KEEP_THRESH else "face_detected"
    if is_context:
        keep = True
        reason = "context_frame"

    return keep, reason, motion_score, face_found

# ---------------------------------------------------------------------------
# THUMBNAIL HELPER
# ---------------------------------------------------------------------------

def frame_to_b64_thumb(frame: np.ndarray, width: int = 200) -> str:
    h, w = frame.shape[:2]
    nh = int(h * width / w)
    thumb = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(buf).decode("utf-8")

# ---------------------------------------------------------------------------
# VIDEO WRITING
# ---------------------------------------------------------------------------

def write_frames_to_video(kept_frames: list, output_path: Path, fps: float, frame_size: tuple):
    temp_avi = "temp_raw.avi"
    cmd = [
        "ffmpeg", "-y", "-i", temp_avi, 
        "-vcodec", "libx264", "-crf", str(OUTPUT_CRF), 
        "-preset", "fast", str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    if os.path.exists(temp_avi):
        os.remove(temp_avi)

# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------
def generate_compression_report(segments: list, stats: dict, output_path: Path):
    # Calculate the visual bar width for the size comparison
    reduction_width = 100 - stats['reduction_pct']
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentio Mind | Compression Report</title>
        <style>
            :root {{
                --bg: #f8f9fa; --surface: #ffffff; --text: #202124; 
                --text-muted: #5f6368; --primary: #0b57d0; --success: #1e8e3e; 
                --border: #dadce0; --radius: 12px;
            }}
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; 
                background-color: var(--bg); color: var(--text); margin: 0; padding: 40px 20px; 
            }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 2px solid var(--border); padding-bottom: 20px; margin-bottom: 30px; }}
            .header h1 {{ margin: 0; font-size: 28px; color: var(--text); letter-spacing: -0.5px; }}
            .header p {{ margin: 5px 0 0 0; color: var(--text-muted); }}
            .bonus-badge {{ background: linear-gradient(135deg, #FFD700 0%, #F5A623 100%); color: #000; padding: 6px 12px; border-radius: 20px; font-weight: bold; font-size: 13px; display: inline-flex; align-items: center; gap: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .card {{ background: var(--surface); padding: 24px; border-radius: var(--radius); border: 1px solid var(--border); box-shadow: 0 4px 6px rgba(0,0,0,0.02); transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-2px); }}
            .card h3 {{ margin: 0 0 10px 0; color: var(--text-muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
            .card .value {{ margin: 0; font-size: 28px; font-weight: 700; color: var(--primary); }}
            .card .sub-value {{ margin: 5px 0 0 0; font-size: 14px; color: var(--success); font-weight: 600; }}
            
            .visual-bar-container {{ background: var(--surface); padding: 24px; border-radius: var(--radius); border: 1px solid var(--border); margin-bottom: 40px; }}
            .visual-bar-container h3 {{ margin: 0 0 15px 0; font-size: 16px; }}
            .bar-bg {{ width: 100%; height: 24px; background: #e8eaed; border-radius: 12px; overflow: hidden; position: relative; }}
            .bar-fill {{ height: 100%; background: var(--success); width: {reduction_width}%; border-radius: 12px; transition: width 1s ease-in-out; }}
            .bar-labels {{ display: flex; justify-content: space-between; margin-top: 10px; font-size: 14px; color: var(--text-muted); font-weight: 500; }}
            
            .storyboard-title {{ font-size: 20px; margin-bottom: 20px; }}
            .storyboard {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 24px; }}
            .segment {{ background: var(--surface); border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); transition: box-shadow 0.2s; }}
            .segment:hover {{ box-shadow: 0 8px 15px rgba(0,0,0,0.05); }}
            .segment img {{ width: 100%; height: 160px; object-fit: cover; border-bottom: 1px solid var(--border); display: block; }}
            .info {{ padding: 16px; }}
            .info-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }}
            .info-label {{ color: var(--text-muted); }}
            .info-data {{ font-weight: 600; color: var(--text); }}
            .reason-badge {{ display: inline-block; padding: 4px 10px; background: #e8f0fe; color: var(--primary); border-radius: 6px; font-size: 12px; font-weight: 600; margin-top: 8px; border: 1px solid #d2e3fc; }}
            .reason-face {{ background: #ceead6; color: var(--success); border-color: #a8dab5; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>Smart Compression Report</h1>
                    <p>Processed in <strong>{stats['processing_time_sec']}s</strong> at 12 FPS (H.264)</p>
                </div>
                <div class="bonus-badge">
                    ★ Auto-Calibrated Threshold Active
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="card">
                    <h3>Storage Reduction</h3>
                    <p class="value">{stats['reduction_pct']}%</p>
                    <p class="sub-value">{stats['original_size_mb']} MB &rarr; {stats['compressed_size_mb']} MB</p>
                </div>
                <div class="card">
                    <h3>Timeline Kept</h3>
                    <p class="value">{stats['compressed_duration_sec']}s</p>
                    <p class="sub-value">Down from {stats['original_duration_sec']}s</p>
                </div>
                <div class="card">
                    <h3>Frames Retained</h3>
                    <p class="value">{stats['frames_kept']}</p>
                    <p class="sub-value">Out of {stats['frames_original']} total frames</p>
                </div>
                <div class="card">
                    <h3>Discarded: Static</h3>
                    <p class="value">{stats['frames_discarded_reasons']['low_motion_no_face']}</p>
                    <p class="sub-value">Empty scene frames dropped</p>
                </div>
            </div>

            <div class="visual-bar-container">
                <h3>Visual Storage Footprint</h3>
                <div class="bar-bg">
                    <div class="bar-fill"></div>
                </div>
                <div class="bar-labels">
                    <span>{stats['compressed_size_mb']} MB (New Size)</span>
                    <span>{stats['original_size_mb']} MB (Original Size)</span>
                </div>
            </div>

            <h2 class="storyboard-title">Intelligence Storyboard ({len(segments)} Segments Kept)</h2>
            <div class="storyboard">
    """
    
    for seg in segments:
        # Give face detection a special green badge color
        badge_class = "reason-badge reason-face" if "face" in seg['reason_kept'] else "reason-badge"
        
        html += f"""
                <div class="segment">
                    <img src="data:image/jpeg;base64,{seg['thumbnail_b64']}" alt="Segment Frame">
                    <div class="info">
                        <div class="info-row">
                            <span class="info-label">Segment ID</span>
                            <span class="info-data">#{seg['segment_id']}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Timestamp</span>
                            <span class="info-data">{seg['start_sec']}s - {seg['end_sec']}s</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Frames Kept</span>
                            <span class="info-data">{seg['frames_in_segment']}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Motion Score</span>
                            <span class="info-data">{seg['motion_score_avg']}</span>
                        </div>
                        <span class="{badge_class}">{seg['reason_kept'].replace('_', ' ').title()}</span>
                    </div>
                </div>
        """
        
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap          = cv2.VideoCapture(str(VIDEO_IN))
    total        = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total / fps_in
    
    if not VIDEO_IN.exists():
        print(f"ERROR: Could not find {VIDEO_IN}.")
        exit(1)
        
    orig_mb = VIDEO_IN.stat().st_size / 1_000_000
    print(f"Input: {VIDEO_IN}  |  {total} frames  |  {duration:.1f}s  |  {orig_mb:.1f} MB")

    temp_avi = "temp_raw.avi"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(temp_avi, fourcc, OUTPUT_FPS, (fw, fh))

    kept_frames = [] 
    segments    = []
    prev_frame  = None
    prev_gray   = None
    prev_hash   = ""
    last_kept_t = -999.0
    cur_seg     = None
    disc_dup    = 0
    disc_stat   = 0

    # BONUS TASK: Auto-Calibration variables
    calibration_scores = []
    is_calibrated = False

    frame_idx = 0
    last_processed_ts = -1.0
    
    while True:
        ret = cap.grab()
        if not ret:
            break
            
        ts = frame_idx / fps_in
        
        # STRICT TIME SKIP: Only process exactly OUTPUT_FPS (12 frames a second)
        if (ts - last_processed_ts) < (1.0 / OUTPUT_FPS):
            frame_idx += 1
            continue
            
        ret, frame = cap.retrieve()
        last_processed_ts = ts

        keep, reason, motion, face = should_keep_frame(
            frame, prev_frame, prev_hash, last_kept_t, ts, cascade
        )
        
        # BONUS TASK LOGIC: Capture background noise for first 30 seconds
        if ts <= 30.0:
            calibration_scores.append(motion)
        elif not is_calibrated and len(calibration_scores) > 0:
            # Set the new discard threshold to 1.5x the median background noise
            MOTION_DISCARD_THRESH = max(0.01, min(np.median(calibration_scores) * 1.5, 0.1))
            is_calibrated = True
            print(f"-> Bonus: Auto-calibrated threshold to {MOTION_DISCARD_THRESH:.4f}")

        if keep:
            out_writer.write(frame) 
            kept_frames.append(frame_idx) 
            prev_hash   = compute_phash(frame)
            last_kept_t = ts

            if cur_seg is None or (ts - cur_seg["end_sec"]) > 2.5:
                if cur_seg:
                    segments.append(cur_seg)
                cur_seg = {
                    "segment_id":            len(segments) + 1,
                    "start_sec":             round(ts, 2),
                    "end_sec":               round(ts, 2),
                    "frames_in_segment":     1,
                    "reason_kept":           reason,
                    "face_count_in_segment": 1 if face else 0,
                    "motion_score_avg":      round(motion, 3),
                    "thumbnail_b64":         frame_to_b64_thumb(frame),
                }
            else:
                cur_seg["end_sec"]               = round(ts, 2)
                cur_seg["frames_in_segment"]    += 1
                cur_seg["face_count_in_segment"] += 1 if face else 0
        else:
            if "duplicate" in reason:
                disc_dup  += 1
            else:
                disc_stat += 1

        prev_frame = frame
        frame_idx += 1

    if cur_seg:
        segments.append(cur_seg)
        
    cap.release()
    out_writer.release() 

    print(f"Kept {len(kept_frames)} / {total} frames across {len(segments)} segments")
    print("Writing compressed video ...")
    write_frames_to_video(kept_frames, VIDEO_OUT, OUTPUT_FPS, (fw, fh))

    comp_mb = VIDEO_OUT.stat().st_size / 1_000_000 if VIDEO_OUT.exists() else 0.0
    t_end   = time.time()

    stats = {
        "source_video":             str(VIDEO_IN),
        "compressed_video":         str(VIDEO_OUT),
        "original_size_mb":         round(orig_mb, 2),
        "compressed_size_mb":       round(comp_mb, 2),
        "reduction_pct":            round((1 - comp_mb / (orig_mb + 1e-9)) * 100, 1),
        "original_duration_sec":    round(duration, 2),
        "compressed_duration_sec":  round(len(kept_frames) / OUTPUT_FPS, 2),
        "original_fps":             round(fps_in, 2),
        "output_fps":               OUTPUT_FPS,
        "frames_original":          total,
        "frames_kept":              len(kept_frames),
        "processing_time_sec":      round(t_end - t_start, 2),
        "segments":                 segments,
        "frames_discarded_reasons": {
            "near_duplicate_phash": disc_dup,
            "low_motion_no_face":   disc_stat,
            "total_discarded":      total - len(kept_frames),
        },
    }

    with open(SEGMENTS_JSON_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    generate_compression_report(segments, stats, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {stats['processing_time_sec']}s")
    print(f"  Size:     {orig_mb:.1f} MB  →  {comp_mb:.1f} MB  ({stats['reduction_pct']}% smaller)")
    print(f"  Duration: {duration:.1f}s  →  {stats['compressed_duration_sec']:.1f}s")
    print(f"  Report  → {REPORT_HTML_OUT}")
    print(f"  JSON    → {SEGMENTS_JSON_OUT}")
    print("=" * 55)