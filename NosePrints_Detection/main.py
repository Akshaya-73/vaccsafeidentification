# code

import cv2
import numpy as np
import pandas as pd
import os
from NosePrintes_Detection.augment_image import augment_image

DATASET_PATH = "D:/1csm/Interns/Hexart/Nose Prints"

# ---------------- RETINEX ----------------
def multi_scale_retinex(img, sigmas=[15, 80, 250]):
    img = img.astype(np.float32) + 1.0
    retinex = np.zeros_like(img)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex += np.log(img) - np.log(blur)
    retinex /= len(sigmas)
    return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = multi_scale_retinex(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- ORB ----------------
orb = cv2.ORB_create(nfeatures=3000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# ---------------- LOAD FEATURES ----------------
descs, kps = {}, {}

for f in os.listdir(DATASET_PATH):
    if f.endswith(".jpg"):
        img = cv2.imread(os.path.join(DATASET_PATH, f))
        g = preprocess(img)
        kp, des = orb.detectAndCompute(g, None)
        if des is not None:
            descs[f] = des
            kps[f] = len(kp)

files = sorted(descs.keys())
n = len(files)

# ---------------- MATCH FUNCTION ----------------
def match_score(des1, des2, kp1, kp2):
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good) / ((kp1 + kp2) / 2)

# ---------------- MATRIX ----------------
scores = np.zeros((n, n))
decisions = np.empty((n, n), dtype=object)

THRESHOLD = 0.30

for i in range(n):
    for j in range(n):
        if i == j:
            img = cv2.imread(os.path.join(DATASET_PATH, files[i]),0)
            img = augment_image(img)
            img = preprocess(img)
            
            kp, des = orb.detectAndCompute(img, None)
            s1 = match_score(descs[files[i]], des,kps[files[i]], len(kp))
            s2 = match_score(descs[files[i]], des,kps[files[i]], len(kp))
        else:
            s1 = match_score(descs[files[i]], descs[files[j]],
                             kps[files[i]], kps[files[j]])
            s2 = match_score(descs[files[j]], descs[files[i]],
                             kps[files[j]], kps[files[i]])
        score = min(s1, s2)

        scores[i, j] = round(score, 8)
        decisions[i, j] = "SAME" if score >= THRESHOLD else "DIFFERENT"

df_scores = pd.DataFrame(scores, index=files, columns=files)
df_decision = pd.DataFrame(decisions, index=files, columns=files)

print("\n=== OPTIMIZED MATCH MATRIX ===")
print(df_scores)

print("\n=== DECISION MATRIX ===")
print(df_decision)
