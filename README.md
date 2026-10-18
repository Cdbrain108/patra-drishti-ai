# Patra Drishti (पत्र दृष्टि) 🌿
### *AI-Powered Botanical Vision & Pathology Advisor — Seeing Beyond the Leaf*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11s-10B981.svg)](https://github.com/ultralytics/ultralytics)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-Llama%203.2%20Vision-76B900.svg)](https://build.nvidia.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <img src="assets/logo.png" width="220" alt="Patra Drishti Logo" style="border-radius: 20px; box-shadow: 0 0 30px rgba(16,185,129,0.3);"/>
  <p><em>"Seeing Beyond the Leaf: YOLOv11 Computer Vision + NVIDIA Multimodal AI Pathologist."</em></p>
</div>

---

## 🔬 Project Overview

**Patra Drishti (पत्र दृष्टि)** is an end-to-end, two-stage artificial intelligence system engineered for real-time agricultural crop pathology:

1. **Stage 1 (The Eyes):** A fine-tuned **YOLOv11s** object detection model (`best.pt`) optimized for CPU inference that localizes cellular lesions, foliar chlorosis, and healthy leaves in complex real-world field lighting.
2. **Stage 2 (The Brain):** **NVIDIA NIM Multimodal Vision AI (`meta/llama-3.2-11b-vision-instruct`)** and Google Gemini 2.5 Flash taking raw images and YOLO bounding box telemetry to stream natural-language clinical diagnoses, pathogen etiology, and organic/chemical remediation protocols.

---

## 🌿 PlantDoc Dataset & Class Taxonomy (27 Classes)

Trained on **2,555+ cleaned real-world agricultural images** across 13 crops:

### 🟢 10 Healthy Foliage Varieties
- Apple healthy leaf, Blueberry healthy leaf, Cherry healthy leaf, Grape healthy leaf, Peach healthy leaf, Potato healthy leaf, Raspberry healthy leaf, Soybean healthy leaf, Strawberry healthy leaf, Tomato healthy leaf, Bell Pepper healthy leaf.

### 🔴 17 Pathological Disease Classes
- **Tomato:** Septoria Leaf Spot, Bacterial Spot, Mold Leaf, Late Blight, Early Blight, Yellow Leaf Curl Virus, Mosaic Virus.
- **Corn:** Northern Leaf Blight, Common Rust, Gray Leaf Spot.
- **Apple:** Apple Scab, Cedar Apple Rust.
- **Potato:** Potato Late Blight.
- **Grape:** Grape Black Rot.
- **Squash:** Powdery Mildew.
- **Bell Pepper:** Bacterial Leaf Spot.

---

## ⚡ Key Features

- **ChatGPT-Style Blue Dot-Matrix Loader:** Animated pulsing black-card grid with live percentage progression (`16%` → `35%` → `51%` → `88%` → `100%`) during neural inference.
- **Real-Time Streaming Consultation:** Multimodal vision clinical advice streamed token-by-token with typewriter animations.
- **CPU Optimized:** Sub-second inference runs cleanly on standard CPUs without GPU dependencies.
- **Interactive Benchmark Gallery:** Includes 18 pre-loaded crop specimens for 1-click verification across healthy and diseased classes.

---

## 🚀 Quickstart

### 1. Clone Repository
```bash
git clone https://github.com/Cdbrain108/patra-drishti-ai.git
cd patra-drishti-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Web Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Model Training Evaluation

The repository includes complete training curves and evaluation artifacts from the 60-epoch AdamW training schedule:
- `results.png`: Precision, Recall, Box Loss, and mAP50 curves.
- `confusion_matrix_normalized.png`: Multi-class normalized confusion matrix.
- `PR_curve.png` & `F1_curve.png`: Precision-Recall and F1 confidence curves.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
