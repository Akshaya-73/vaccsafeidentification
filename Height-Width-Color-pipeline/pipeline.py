# pipeline.py
# Orchestrator — runs all 5 stages end to end.
#
# Usage:
#   python pipeline.py --video input_video.mp4

import cv2
import argparse
from stage1_detection  import detect_dogs
from stage2_tracking   import build_tracker, track_frame, FRAME_RATE
from stage3_features   import extract_features
from stage4_db_query   import query_db
from stage5_task_manager import add_pending_task, log_match


def run_pipeline(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    fps            = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / FRAME_RATE))
    tracker        = build_tracker()

    # track which dog IDs have already been checked
    checked_ids    = {}   # track_id -> "match" | "new"

    frame_count    = 0

    print(f"Processing video: {video_path}")
    print(f"FPS: {fps:.1f} | Processing every {frame_interval} frame(s)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # ── Stage 1: detect dogs ──────────────────────────────
        dogs = detect_dogs(frame)

        if not dogs:
            continue

        # ── Stage 2: track dogs ───────────────────────────────
        tracked = track_frame(tracker, frame, dogs)

        for dog in tracked:
            track_id = dog["track_id"]

            # skip if already processed this dog
            if track_id in checked_ids:
                continue

            print(f"[Frame {frame_count}] New dog — track_id: {track_id}")

            # ── Stage 3: extract features ─────────────────────
            features = extract_features(dog)
            print(f"  Features → color: {features['color']} | "
                  f"w1: {features['w1']} | h1: {features['h1']} | h2: {features['h2']}")

            # ── Stage 4: query database ───────────────────────
            match = query_db(features)

            # ── Stage 5: handle result ────────────────────────
            if match:
                checked_ids[track_id] = "match"
                log_match(track_id, match)
            else:
                checked_ids[track_id] = "new"
                add_pending_task(track_id, features)

    cap.release()

    # ── Summary ───────────────────────────────────────────────
    total   = len(checked_ids)
    matches = sum(1 for v in checked_ids.values() if v == "match")
    new     = sum(1 for v in checked_ids.values() if v == "new")

    print(f"\n{'─'*50}")
    print(f" PIPELINE COMPLETE")
    print(f"{'─'*50}")
    print(f"  Total unique dogs : {total}")
    print(f"  Matched in DB     : {matches}")
    print(f"  New (pending)     : {new}")
    print(f"  Pending tasks     → pending_tasks.json")
    print(f"{'─'*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()

    run_pipeline(args.video)
