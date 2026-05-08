from ultralytics import YOLO
import cv2
import numpy as np
import math
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------- User Config ---------------------
video_path = "dogs_video.mp4"
output_folder = "output_frames_updated/"  # new folder for overlay frames
os.makedirs(output_folder, exist_ok=True)

frame_rate = 3  # approx frames per second to process

# YOLOv8 Segmentation model
model = YOLO("yolov8n-seg.pt")  # or yolov8s-seg.pt

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# --------------------- Video Capture ---------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(fps / frame_rate))
frame_count = 0
saved_frame_number = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Process only every nth frame based on desired FPS
    if frame_count % frame_interval != 0:
        continue

    print(f"Processing video frame {frame_count}...")

    # --------------------- YOLO Detection ---------------------
    results = model(frame)

    detections = []

    for result in results:
        if result.masks is None:
            continue

        masks = result.masks.xy
        classes = result.boxes.cls

        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            class_name = model.names[class_id]

            if class_name == "dog":
                mask_np = np.array(mask, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(mask_np)

                # w1, h1
                w1 = w
                h1 = h

                # Extreme points
                topmost = mask_np[mask_np[:, :, 1].argmin()][0]
                bottommost = mask_np[mask_np[:, :, 1].argmax()][0]

                # h2: leg height approx (bottom to mid-body)
                center_y = y + h // 2
                h2 = bottommost[1] - center_y

                # Append to detections for DeepSORT
                x1, y1, x2, y2 = x, y, x + w, y + h
                detections.append([x1, y1, x2, y2, 0.9, "dog", (w1, h1, h2)])

    # --------------------- DeepSORT Tracking ---------------------
    tracks = tracker.update_tracks(detections, frame=frame)

    # Dictionary to hold measurements for overlay
    frame_measurements = {}

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Assign measurement by matching track bbox with detection bbox
        for det in detections:
            dx1, dy1, dx2, dy2, conf, cls_name, meas = det
            iou_x1 = max(x1, int(dx1))
            iou_y1 = max(y1, int(dy1))
            iou_x2 = min(x2, int(dx2))
            iou_y2 = min(y2, int(dy2))
            iou_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
            det_area = (dx2 - dx1) * (dy2 - dy1)
            if det_area == 0:
                continue
            if iou_area / det_area > 0.5:  # >50% overlap
                frame_measurements[track_id] = meas
                break

        # Overlay bounding box + measurements if available
        if track_id in frame_measurements:
            w1, h1, h2 = frame_measurements[track_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Dog{track_id} w:{w1} h:{h1} h2:{h2}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --------------------- Save Frame ---------------------
    out_path = os.path.join(output_folder, f"{saved_frame_number}.jpg")
    cv2.imwrite(out_path, frame)
    print(f"Saved Frame {saved_frame_number} → {out_path}")
    saved_frame_number += 1

cap.release()
cv2.destroyAllWindows()
print("✅ All frames processed and saved successfully!")