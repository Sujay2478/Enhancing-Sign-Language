## Enhancing Sign Language Learning with MediaPipe & Deep Learning

An end-to-end web-based system for real-time British Sign Language (BSL) recognition using MediaPipe hand tracking and a deep learning classifier deployed in the browser with ONNX Runtime.

This project supports:
- Real-time hand landmark detection (1-hand and 2-hand signs)
- Neural network–based sign classification
- Confidence scoring
- Dynamic gesture recognition using DTW
- Visual feedback with skeleton overlay
- Browser-side inference (no server latency)

---

### Features

#### 1. Real-Time Hand Tracking
- Uses **MediaPipe Hands** for 3D landmark extraction.
- Detects left and right hands independently.
- Supports both **one-hand** and **two-hand** signs.

#### 2. Deep Learning Sign Classifier
- Trained in **PyTorch**
- Exported to **ONNX**
- Runs fully in browser via **onnxruntime-web**
- Input:
  - 63 features (single hand)
  - 126 features (two hands concatenated)
- Output: Softmax probabilities over BSL alphabet & numbers.

#### 3. Feature Normalisation
- Z-score normalisation using training statistics (`bsl_norm.json`)
- Ensures inference distribution matches training distribution.

#### 4. Dynamic Gesture Recognition
- Sequence matching using **Dynamic Time Warping (DTW)**
- Used for gestures like "Yes", "No", etc.

#### 5. Visual Feedback
- Live skeleton overlay
- Confidence percentage
- Correct / Incorrect indicator
- View gating for camera positioning
