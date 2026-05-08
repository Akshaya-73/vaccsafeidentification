import cv2
import numpy as np
import math
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------- User Config ---------------------
video_path = "dogs_video.mp4"
output_folder = "output_frames_updated/"
os.makedirs(output_folder, exist_ok=True)

frame_rate = 3  # FPS to process

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # you can use yolov8s-seg.pt for better accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50, n_init=3)

# --------------------- Helper Function ---------------------
def compute_measurements(mask_np):
    x, y, w, h = cv2.boundingRect(mask_np)

    # Basic width & height
    w1 = w
    h1 = h

    # Extreme points
    ys = mask_np[:, :, 1]
    topmost = mask_np[ys.argmin()][0]
    bottommost = mask_np[ys.argmax()][0]

    # Approx leg height
    center_y = y + h // 2
    h2 = bottommost[1] - center_y

    return (x, y, x + w, y + h), (w1, h1, h2)

# --------------------- Video Capture ---------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(fps / frame_rate))

frame_count = 0
saved_frame_number = 1

# --------------------- Processing Loop ---------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames to match desired FPS
    if frame_count % frame_interval != 0:
        continue

    print(f"Processing frame {frame_count}...")

    results = model(frame)

    detections = []
    measurement_map = []

    # --------------------- Detection ---------------------
    for result in results:
        if result.masks is None:
            continue

        masks = result.masks.xy
        classes = result.boxes.cls
        scores = result.boxes.conf

        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            class_name = model.names[class_id]

            if class_name == "dog":
                mask_np = np.array(mask, dtype=np.int32)

                bbox, measurements = compute_measurements(mask_np)
                x1, y1, x2, y2 = bbox
                w1, h1, h2 = measurements

                conf = float(scores[i])

                # DeepSORT format: [x1, y1, x2, y2, confidence, class_id]
                detections.append([x1, y1, x2, y2, conf, 0])

                # Store measurement separately
                measurement_map.append((bbox, measurements))

                # Draw mask overlay
                cv2.polylines(frame, [mask_np], True, (0, 255, 0), 2)

    # --------------------- Tracking ---------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        matched_measure = None

        # Match detection ↔ track using IoU
        for det_box, meas in measurement_map:
            dx1, dy1, dx2, dy2 = det_box

            iou_x1 = max(x1, dx1)
            iou_y1 = max(y1, dy1)
            iou_x2 = min(x2, dx2)
            iou_y2 = min(y2, dy2)

            inter_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
            det_area = (dx2 - dx1) * (dy2 - dy1)

            if det_area == 0:
                continue

            if inter_area / det_area > 0.5:
                matched_measure = meas
                break

        # Unique color per dog
        color = (
            int(track_id * 70 % 255),
            int(track_id * 130 % 255),
            int(track_id * 200 % 255)
        )

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        if matched_measure:
            w1, h1, h2 = matched_measure
            label = f"Dog ID:{track_id} w:{w1} h:{h1} h2:{h2}"
        else:
            label = f"Dog ID:{track_id}"

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --------------------- Save Frame ---------------------
    out_path = os.path.join(output_folder, f"{saved_frame_number}.jpg")
    cv2.imwrite(out_path, frame)

    print(f"Saved frame {saved_frame_number} → {out_path}")
    saved_frame_number += 1

# --------------------- Cleanup ---------------------
cap.release()
cv2.destroyAllWindows()

print("✅ All frames processed and saved successfully!")