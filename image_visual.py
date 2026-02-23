#image visualization
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================
# CONFIGURATION
# =============================
DATASET_PATH = r"D:/1csm/Interns/Hexart/Nose Prints"  
IMAGE_SIZE = (256, 256)

ORB_FEATURES = 3000
RATIO_TEST = 0.75

# =============================
# INITIALIZE ORB & MATCHER
# =============================
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# =============================
# LOAD IMAGES
# =============================
def load_images_from_folder(folder):
    images = []
    filenames = []

    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        img = cv2.imread(path, 0)  # grayscale

        if img is None:
            continue

        img = cv2.resize(img, IMAGE_SIZE)
        images.append(img)
        filenames.append(file)

    return images, filenames

# =============================
# DRAW KEYPOINTS
# =============================
def draw_keypoints(img, keypoints):
    return cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

# =============================
# SINGLE WINDOW VISUALIZATION
# =============================
def visualize_all_pairs_single_window(images, filenames, orb, bf):
    n = len(images)
    total_pairs = n * n

    cols = 3
    rows = total_pairs

    plt.figure(figsize=(20, rows * 3))
    plot_idx = 1

    for i in range(n):
        for j in range(n):

            img1 = images[i]
            img2 = images[j]

            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                continue

            matches = bf.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n2 in matches:
                if m.distance < RATIO_TEST * n2.distance:
                    good_matches.append(m)

            similarity = len(good_matches) / min(len(kp1), len(kp2))

            kp_img1 = draw_keypoints(img1, kp1)
            kp_img2 = draw_keypoints(img2, kp2)

            match_img = cv2.drawMatches(
                img1, kp1,
                img2, kp2,
                good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # ---------- KEYPOINTS IMAGE i ----------
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(kp_img1, cmap="gray")
            plt.title(f"KP {filenames[i]} ({len(kp1)})")
            plt.axis("off")
            plot_idx += 1

            # ---------- KEYPOINTS IMAGE j ----------
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(kp_img2, cmap="gray")
            plt.title(f"KP {filenames[j]} ({len(kp2)})")
            plt.axis("off")
            plot_idx += 1

            # ---------- MATCHES + METRICS ----------
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(match_img, cmap="gray")
            plt.title(
                f"{filenames[i]} vs {filenames[j]}\n"
                f"Good Matches: {len(good_matches)} | "
                f"Similarity: {similarity:.3f}"
            )
            plt.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.show()

# =============================
# MAIN EXECUTION
# =============================
if __name__ == "__main__":

    images, filenames = load_images_from_folder(DATASET_PATH)

    print(f"Total images loaded: {len(images)}")

    if len(images) == 0:
        raise ValueError("No images found in dataset path!")

    visualize_all_pairs_single_window(images, filenames, orb, bf)
