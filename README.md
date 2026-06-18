# Real-Time ASL Alphabet Recognition using MediaPipe & TensorFlow

<p align="center">
  <img src="assets/demo.gif" width="750">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg">
  <img src="https://img.shields.io/badge/MediaPipe-Latest-green.svg">
  <img src="https://img.shields.io/badge/OpenCV-4.x-red.svg">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg">
</p>

## 📌 Overview

This project implements a **real-time American Sign Language (ASL) alphabet recognition system** using a webcam.

The system leverages **MediaPipe Hand Landmarker** to extract 3D hand landmarks and a trained **TensorFlow Multi-Layer Perceptron (MLP)** model to classify static ASL hand signs.

The recognized character is displayed on screen and can optionally be converted to speech using a built-in Text-to-Speech (TTS) module.

### Key Features

✅ Real-time webcam inference

✅ MediaPipe hand tracking

✅ TensorFlow/Keras classifier

✅ Landmark normalization

✅ Prediction smoothing

✅ Text-to-Speech output

✅ Lightweight and CPU-friendly

---

## 🎯 Supported Signs

The current version supports **24 static ASL alphabet classes**:

| A | B | C | D | E | F |
|---|---|---|---|---|---|
| G | H | I | K | L | M |
| N | O | P | Q | R | S |
| T | U | V | W | X | Y |

### Not Supported

| J | Z |
|---|---|

These letters require motion trajectories and cannot be reliably recognized using a single static frame.

---

# 🏗️ System Architecture

```text
Webcam
   │
   ▼
MediaPipe Hand Landmarker
   │
   ▼
21 Hand Landmarks (x,y,z)
   │
   ▼
Landmark Normalization
   │
   ▼
63-Dimensional Feature Vector
   │
   ▼
MLP Classifier
   │
   ▼
Prediction Smoothing
   │
   ▼
Recognized Character
   │
   ├── Display on Screen
   │
   └── Text-to-Speech
```

---

# 🧠 Model Pipeline

## Step 1: Hand Detection

MediaPipe detects one hand and extracts:

```text
21 landmarks × 3 coordinates
```

Output:

```text
(21,3)
```

---

## Step 2: Landmark Normalization

To improve robustness against position and scale variations:

### Translation

Move wrist landmark (0) to the origin.

```python
coords = coords - wrist
```

### Scaling

Normalize by the distance between:

```text
Wrist (0)
     ↓
Middle Finger MCP (9)
```

```python
coords = coords / scale
```

---

## Step 3: Feature Extraction

Flatten:

```text
21 × 3
```

into:

```text
63 Features
```

```python
features = coords.flatten()
```

---

## Step 4: Classification

A trained Multi-Layer Perceptron predicts the sign class.

```text
Input Layer
63 Features
      │
Dense (128)
      │
ReLU
      │
Dropout
      │
Dense (64)
      │
ReLU
      │
Dropout
      │
Dense (24)
      │
Softmax
      ▼
Predicted Letter
```

---

## Step 5: Prediction Smoothing

Instead of displaying every prediction immediately, the system waits until the same prediction appears across multiple consecutive frames.

Benefits:

- Reduces flickering
- Improves stability
- Prevents repeated TTS outputs

---

# 📂 Project Structure

```text
ASL-Alphabet-Recognition/
│
├── demo_2.py
│
├── models/
│   ├── hand_sign_static_mlp.keras
│   └── hand_sign_static_mlp.h5
│
├── mediapipe/
│   └── hand_landmarker.task
│
├── configs/
│   └── label_map.json
│
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/ASL-Alphabet-Recognition.git

cd ASL-Alphabet-Recognition
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## Basic Run

```bash
python demo.py
```

---

## Select Camera

```bash
python demo.py --camera 1
```

---

## Disable Audio

```bash
python demo.py --no_audio
```

---

## Increase Confidence Threshold

```bash
python demo.py --confidence 0.8
```

---

## Increase Stability Window

```bash
python demo.py --smooth_window 15
```

---

# 🖥️ Example Output

```text
Prediction: A
Confidence: 0.97
```

Displayed as:

```text
A (0.97)
```

and optionally spoken aloud.

---

# 📊 Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 98.0% |
| Weighted Precision | 99.0% |
| Weighted Recall | 98.0% |
| Weighted F1-Score | 98.0% |
| Number of Classes | 38 |
| Test Samples | 9,588 |

---

# 🔬 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core Language |
| TensorFlow/Keras | Classification Model |
| MediaPipe | Hand Landmark Detection |
| OpenCV | Webcam Processing |
| NumPy | Numerical Computation |
| pyttsx3 | Offline Text-to-Speech |

---

# 🚧 Limitations

Current system:

❌ Cannot recognize dynamic signs (J, Z)

❌ Supports only single-hand gestures

❌ Does not recognize words or sentences

❌ No language model for spell correction

---

# 🔮 Future Improvements

### Dynamic Sign Recognition

Support:

```text
J
Z
```

using:

- LSTM
- GRU
- Transformer

---

### Continuous Sign Recognition

```text
Sign Language
        ↓
Words
        ↓
Sentences
```

---

### Sign-to-Text Translation

Real-time sentence generation from continuous signing.

---

### Sign-to-Speech Translation

```text
Hand Sign
     ↓
Text
     ↓
Voice
```

for accessibility applications.

---

# 📈 Applications

- Deaf and hard-of-hearing assistance
- Human-computer interaction
- Educational tools
- Sign language learning
- Accessibility systems
- AI and Computer Vision research

---

# 👨‍💻 Author

**Bui Thi Quynh Nhu**

Computer Science Student

Interests:

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Accessibility Technology

---

# ⭐ If you find this project useful

Give the repository a star ⭐ and consider contributing to future improvements.

```
⭐ Star this repository
🍴 Fork this repository
🚀 Build something awesome
```

---

## 📚 Citation

```bibtex
@software{quynhnhu2026asl,
  author = {Bui Thi Quynh Nhu},
  title = {Real-Time ASL Alphabet Recognition using MediaPipe and TensorFlow},
  year = {2026},
  url = {https://github.com/emmybui/ASL-Alphabet-Recognition}
}
```
