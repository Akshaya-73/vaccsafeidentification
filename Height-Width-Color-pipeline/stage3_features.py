# stage3_features.py
# Feature extraction — w1, h1, h2 + color
#
# Standalone test:
#   python stage3_features.py --input frame_25.jpg

import cv2
import numpy as np
import argparse
from stage1_detection import detect_dogs


def detect_color(image_rgb):
    pixels = image_rgb.reshape(-1, 3)
    pixels = pixels[np.sum(pixels, axis=1) > 60]
    if len(pixels) == 0:
        return "unknown"
    pixels_hsv = np.uint8(pixels).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(pixels_hsv, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hsv = hsv[hsv[:, 2] > 25]
    if len(hsv) == 0:
        return "unknown"
    brightness = np.mean(hsv[:, 2])
    saturation = np.mean(hsv[:, 1])
    if brightness < 85:
        return "black"
    if brightness > 170 and saturation < 80:
        return "white"
    return "brown"


def extract_dimensions(mask_np, bbox):
    x, y, w, h = cv2.boundingRect(mask_np)
    w1 = w
    h1 = h
    ys = mask_np[:, 1]
    bottommost = mask_np[ys.argmax()]
    center_y = y + h // 2
    h2 = int(bottommost[1]) - center_y
    return {"w1": w1, "h1": h1, "h2": h2}


def extract_features(dog):
    features = {"w1": None, "h1": None, "h2": None, "color": "unknown"}
    if dog.get("mask_np") is not None and dog.get("bbox") is not None:
        dims = extract_dimensions(dog["mask_np"], dog["bbox"])
        features.update(dims)
    if dog.get("crop") is not None:
        crop = dog["crop"]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.GaussianBlur(crop_rgb, (5, 5), 0)
        features["color"] = detect_color(crop_rgb)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to image")
    args = parser.parse_args()

    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Cannot load image: {args.input}")
        exit(1)

    dogs = detect_dogs(frame)
    if not dogs:
        print("No dogs detected.")
        exit(0)

    print(f"Detected {len(dogs)} dog(s)\n")
    for idx, dog in enumerate(dogs):
        features = extract_features(dog)
        print(f"  Dog {idx+1}")
        print(f"    w1    : {features['w1']} px")
        print(f"    h1    : {features['h1']} px")
        print(f"    h2    : {features['h2']} px")
        print(f"    color : {features['color']}")
