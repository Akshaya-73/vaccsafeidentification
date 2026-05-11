# ─────────────────────────────────────────────
#  stage1_detection.py
#  YOLO dog segmentation — extracts dog crops
#  and bounding boxes from a single frame.
#
#  Standalone test:
#    python stage1_detection.py --input frame_25.jpg
# ─────────────────────────────────────────────

import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# ─────────────────────────────────────────────
MODEL_PATH = "yolov8s-seg.pt"
# ─────────────────────────────────────────────

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def detect_dogs(frame: np.ndarray) -> list[dict]:
    """
    Run YOLOv8 segmentation on a frame.
    Returns a list of dicts, one per detected dog:
      {
        "crop"       : np.ndarray  — cropped dog region (BGR)
        "mask_np"    : np.ndarray  — polygon mask points
        "bbox"       : (x1, y1, x2, y2)
        "confidence" : float
      }
    """
    model   = get_model()
    results = model(frame, verbose=False)
    dogs    = []

    for result in results:
        if result.masks is None:
            continue

        masks   = result.masks.xy
        classes = result.boxes.cls
        scores  = result.boxes.conf

        for i, mask in enumerate(masks):
            class_id = int(classes[i])
            if model.names[class_id] != "dog":
                continue

            mask_np = np.array(mask, dtype=np.int32)

            # binary mask
            binary_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(binary_mask, [mask_np], 255)

            # segmented dog
            segmented = cv2.bitwise_and(frame, frame, mask=binary_mask)

            # bounding box
            x, y, w, h = cv2.boundingRect(mask_np)
            crop = segmented[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (256, 256))

            dogs.append({
                "crop"       : crop,
                "mask_np"    : mask_np,
                "bbox"       : (x, y, x+w, y+h),
                "confidence" : float(scores[i]),
            })

    return dogs


# ─────────────────────────────────────────────
#  Standalone test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to image")
    args = parser.parse_args()

    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Cannot load image: {args.input}")
        exit(1)

    dogs = detect_dogs(frame)
    print(f"Detected {len(dogs)} dog(s)")

    for idx, dog in enumerate(dogs):
        x1, y1, x2, y2 = dog["bbox"]
        print(f"  Dog {idx+1} | bbox: ({x1},{y1},{x2},{y2}) | conf: {dog['confidence']:.2f}")
        cv2.imwrite(f"debug_dog_{idx+1}.jpg", dog["crop"])
        print(f"           | crop saved → debug_dog_{idx+1}.jpg")
