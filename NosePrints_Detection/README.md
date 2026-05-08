# NosePrints - Biometric Identification System

A nose print recognition system using ORB (Oriented FAST and Rotated BRIEF) feature matching and multi-scale Retinex preprocessing for biometric identification and verification.

## Project Overview

This project implements a biometric identification system that:
- Detects and extracts nose print features from images
- Matches nose prints using ORB descriptors
- Evaluates matching performance with signal-to-noise ratios
- Visualizes keypoint detection and feature matching
- Augments images for robustness testing

## Features

- **Multi-scale Retinex Preprocessing**: Enhances image quality and contrast
- **ORB Feature Extraction**: Detects 3000 keypoints per image
- **Feature Matching**: Uses BFMatcher for descriptor matching
- **Performance Evaluation**: Calculates separation ratios and safety margins
- **Image Augmentation**: Tests descriptor stability with contrast, blur, and rotation
- **Visualization**: Displays keypoints and matches across image pairs

## Project Structure

```
NosePrints/
├── main.py              # Core matching pipeline and similarity matrix
├── augment_image.py     # Image augmentation (contrast, blur, rotation)
├── evaluate_method.py   # Performance metrics and evaluation
├── image_visual.py      # Visualization of keypoints and matches
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## File Descriptions

### `main.py`
- **Purpose**: Main pipeline for nose print matching
- **Key Functions**:
  - `multi_scale_retinex()`: Applies multi-scale Retinex filtering
  - `preprocess()`: Resizes, converts color space, and applies Retinex
  - `match_score()`: Calculates similarity between two descriptors
- **Output**: 
  - Similarity matrix with match scores
  - Decision matrix (SAME/DIFFERENT) based on threshold
  - Default threshold: 0.30

### `augment_image.py`
- **Purpose**: Image augmentation for robustness testing
- **Augmentations**:
  - Contrast adjustment (alpha=1.2, beta=10)
  - Gaussian blur (3x3 kernel)
  - Rotation support (currently 0°, can be modified)
- **Used by**: `main.py` for same-image matching tests

### `evaluate_method.py`
- **Purpose**: Performance evaluation and metrics
- **Metrics Calculated**:
  - Average Signal (diagonal elements - true matches)
  - Average Noise (off-diagonal elements)
  - Maximum Noise (worst false positive)
  - Separation Ratio: `avg_signal / (avg_noise + 1e-6)`
  - Safety Margin: `avg_signal - max_noise`
- **Output**: Formatted evaluation report

### `image_visual.py`
- **Purpose**: Visualization of matching results
- **Displays**:
  - Keypoints detected in each image
  - Count of keypoints per image
  - Feature matches between image pairs
  - Similarity scores
- **Input**: Expects images in the dataset folder

## Installation

1. Install Python 3.8+

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Setup

Update the dataset path in `main.py`:
```python
DATASET_PATH = "path/to/your/nose/prints/folder"
```

Ensure your dataset contains `.jpg` images.

### Run the Main Pipeline
```bash
python main.py
```

This will:
- Load all images from the dataset
- Extract ORB features from each image
- Create a similarity matrix for all image pairs
- Generate a decision matrix applying the threshold

### Visualize Results
```bash
python image_visual.py
```

This displays:
- Keypoints for each image
- Feature matches between all pairs
- Visual similarity assessment

### Evaluate Performance
Import and use the evaluation function:
```python
from evaluate_method import evaluate_method

evaluate_method(score_matrix, "method_name")
```

## Parameters

### ORB Configuration
- **Feature Count**: 3000 keypoints per image
- **Feature Matcher**: BFMatcher with Hamming distance
- **Ratio Test**: 0.75 (Lowe's ratio test for good matches)

### Matching Threshold
- **Default**: 0.30
- **Modify in**: `main.py` line with `THRESHOLD = 0.30`
- Higher = Stricter (fewer matches), Lower = Lenient (more matches)

### Image Preprocessing
- **Size**: 256 x 256 pixels
- **Color Space**: Converted to grayscale
- **Retinex Sigmas**: [15, 80, 250] (multi-scale)

## Requirements

- opencv-python==4.8.1.78
- numpy==1.24.3
- pandas==2.0.3
- matplotlib==3.7.2

See `requirements.txt` for details.

## Output Example

```
=== OPTIMIZED MATCH MATRIX ===
             image1.jpg  image2.jpg  image3.jpg
image1.jpg     1.000      0.150      0.120
image2.jpg     0.145      1.000      0.110
image3.jpg     0.125      0.105      1.000

=== DECISION MATRIX ===
             image1.jpg  image2.jpg  image3.jpg
image1.jpg      SAME     DIFFERENT  DIFFERENT
image2.jpg   DIFFERENT     SAME     DIFFERENT
image3.jpg   DIFFERENT  DIFFERENT     SAME

--- Results for kp with retinex ---
Avg Signal (Diagonal): 0.673
Avg Noise (Off-Diag):  0.007
Max Noise (Worst Case): 0.491
Separation Ratio:      101.68
Safety Margin:         0.182
-------------------------   
np.float64(101.68228560033238)

```

## Algorithm Overview

1. **Image Preprocessing**: Apply multi-scale Retinex to enhance contrast
2. **Feature Extraction**: Detect ORB keypoints and descriptors
3. **Feature Matching**: Use BFMatcher with Lowe's ratio test
4. **Score Calculation**: Normalize matches by average keypoint count
5. **Decision Making**: Compare score against threshold
6. **Evaluation**: Analyze separation between signal and noise

## Notes

- The same-image matching (diagonal) uses augmented versions for robustness testing
- Off-diagonal entries represent cross-image matching
- Higher separation ratios indicate better discrimination capability
- Safety margin shows how much noise can increase before misclassification occurs

## Future Improvements

- Fine-tune ORB parameters for better performance
- Implement additional preprocessing techniques
- Add cross-validation with known identity pairs
- Optimize threshold using ROC analysis
- Implement deep learning-based feature extraction

## License

Academic/Research use

## Contact

Part of Hexart project - Safe Vaccination System
