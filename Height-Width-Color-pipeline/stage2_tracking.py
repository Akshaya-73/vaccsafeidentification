# ─────────────────────────────────────────────
#  stage2_tracking.py
#  DeepSORT tracking — assigns persistent
#  track_id to each dog across video frames.
#
#  Standalone test:
#    python stage2_tracking.py --video input_video.mp4
# ─────────────────────────────────────────────

import cv2
import numpy as np
import argparse
from deep_sort_realtime.deepsort_tracker import DeepSort
from stage1_detection import detect_dogs

# ─────────────────────────────────────────────
FRAME_RATE   = 3    # frames per second to process
MAX_AGE      = 50   # frames to keep a lost track alive
N_INIT       = 3    # frames before a track is confirmed
# ─────────────────────────────────────────────

def build_tracker() -> DeepSort:
    return DeepSort(max_age=MAX_AGE, n_init=N_INIT)


def track_frame(tracker: DeepSort,
                frame: np.ndarray,
                dogs: list[dict]) -> list[dict]:
    """
    Update DeepSORT with current frame detections.
    Returns list of confirmed tracked dogs:
      {
        "track_id" : int
        "bbox"     : (x1, y1, x2, y2)
        "crop"     : np.ndarray   — matched crop from detection
        "mask_np"  : np.ndarray
      }
    """
    if not dogs:
        tracker.update_tracks([], frame=frame)
        return []

    # build DeepSORT detection list [x1,y1,x2,y2, conf, class_id]
    det_list = []
    for dog in dogs:
        x1, y1, x2, y2 = dog["bbox"]
        w = x2 - x1
        h = y2 - y1
        det_list.append(([x1, y1, w, h], dog["confidence"], 0))

    tracks = tracker.update_tracks(det_list, frame=frame)

    tracked = []
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id         = track.track_id
        x1, y1, x2, y2  = map(int, track.to_ltrb())

        # match track back to detection via IoU
        matched_dog = _match_track_to_dog(
            (x1, y1, x2, y2), dogs
        )

        tracked.append({
            "track_id" : track_id,
            "bbox"     : (x1, y1, x2, y2),
            "crop"     : matched_dog["crop"]    if matched_dog else None,
            "mask_np"  : matched_dog["mask_np"] if matched_dog else None,
        })

    return tracked


def _match_track_to_dog(track_bbox: tuple,
                         dogs: list[dict],
                         iou_threshold: float = 0.5) -> dict | None:
    """Match a track bbox to the closest detection using IoU."""
    tx1, ty1, tx2, ty2 = track_bbox

    for dog in dogs:
        dx1, dy1, dx2, dy2 = dog["bbox"]

        ix1 = max(tx1, dx1)
        iy1 = max(ty1, dy1)
        ix2 = min(tx2, dx2)
        iy2 = min(ty2, dy2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        det_area = (dx2 - dx1) * (dy2 - dy1)

        if det_area == 0:
            continue

        if inter / det_area > iou_threshold:
            return dog

    return None


# ─────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / FRAME_RATE))

    tracker     = build_tracker()
    frame_count = 0
    seen_ids    = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        dogs    = detect_dogs(frame)
        tracked = track_frame(tracker, frame, dogs)

        for t in tracked:
            tid = t["track_id"]
            if tid not in seen_ids:
                seen_ids.add(tid)
                print(f"  New track_id: {tid} | bbox: {t['bbox']}")

    cap.release()
    print(f"\nTotal unique dogs tracked: {len(seen_ids)}")
