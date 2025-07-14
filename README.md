# Visual Gun Detection System

A traditional computer‑vision pipeline for detecting guns in images using color segmentation, morphological operations, and feature‑based matching. Unlike modern deep‑learning approaches, this project explores three classic feature‑extraction techniques—Harris + FREAK, SIFT, and ORB—and compares their performance on a custom dataset of gun and non‑gun images.

---

## Overview

This repository implements a step‑by‑step computer‑vision system that:
1. Preprocesses and normalizes input images  
2. Segments potential gun regions by color  
3. Cleans up segmentation masks via morphological operations  
4. Extracts and matches keypoint descriptors against a gun feature database  
5. Classifies regions as “gun” or “non‑gun” based on match ratios  

All stages are built using OpenCV and NumPy, demonstrating how classical methods can still achieve high detection accuracy without deep learning.

---

## Our System Pipeline

1. **Preprocessing**  
   - Resize images to 400 × 300 px  
   - Apply median filter to reduce noise while preserving edges  

2. **Color Segmentation**  
   - K‑means clustering (K=5) on RGB pixels  
   - Select darkest clusters to propose candidate regions  

3. **Morphological Operations**  
   - Closing (fill holes) with a 7 × 7 kernel  
   - Opening (remove noise) with a 5 × 5 kernel  
   - Filter out regions smaller than 1,000 pixels  

4. **Feature Extraction**  
   - Harris + FREAK, SIFT, ORB (configurable)  
   - Limit to top 100–200 keypoints based on response  

5. **Feature Matching & Decision**  
   - Sum‑of‑squared‑differences + Lowe’s ratio test (threshold 0.75)  
   - Classify as “gun” if good‑match percentage > 50 %

---

## Features

- **Three feature‑extraction methods:**  
  - Harris corners + FREAK descriptors  
  - SIFT (Scale‑Invariant Feature Transform)  
  - ORB (Oriented FAST and Rotated BRIEF)  
- **Color‑based region proposal** via K‑means clustering  
- **Morphological cleanup** (closing + opening) to refine candidate masks  
- **Simple yet effective matching** using SSD and ratio test  
- **Performance evaluation** (accuracy, precision, recall, F1 score) on test images  

---

## Dataset

- **Training set:**  
    - 30 images of various gun types (handguns, rifles, etc.)  
- **Negative samples for training:**  
    - 20 images of gun‑like objects (tools, toys, etc.)  
- **Test set:**  
    - 13 images (7 containing guns, 6 without)  
- **Image size:** standardized to 400 × 300 px for consistency  

---
