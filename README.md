# Enhancing Sign Language Learning Through Augmented Reality Feedback

A browser-based **British Sign Language (BSL) learning system** that combines real-time computer vision, deep learning, and interactive feedback to help users practise and improve sign language gestures.

The application uses **MediaPipe Hands** to track hand landmarks, a **PyTorch-trained neural network** exported to **ONNX** for static sign recognition, and **Dynamic Time Warping (DTW)** for evaluating motion-based gestures. Inference runs directly in the browser, allowing users to receive immediate feedback through their webcam without requiring server-side processing.

The project was developed as an accessible approach to independent BSL learning, addressing a key limitation of traditional resources such as diagrams and videos: the lack of immediate feedback on handshape, positioning, and movement.

---

## Key Features

* **Real-time hand tracking** using MediaPipe Hands
* Support for **single-hand and two-hand BSL signs**
* **Neural network-based gesture classification**
* **In-browser ONNX inference**
* Real-time **confidence scoring**
* **Dynamic gesture recognition** using Dynamic Time Warping
* Live **hand skeleton and landmark overlays**
* Immediate visual feedback during practice
* Camera positioning and visibility guidance
* **On-device processing** with no server-side inference required

---

## How It Works

The application follows a lightweight computer-vision pipeline designed for real-time browser execution:

**Webcam Input → MediaPipe Hands → Landmark Processing → Normalisation → ONNX Model → Sign Prediction → User Feedback**

MediaPipe extracts **21 3D landmarks per hand**. These coordinates are converted into feature vectors and passed to a compact neural network for classification. The system uses **63 features for single-hand signs** and **126 features for two-hand signs**.

### 1. Real-Time Hand Tracking

**MediaPipe Hands** detects and tracks the user's hands directly from the webcam feed.

The tracking pipeline:

* Extracts 21 `(x, y, z)` landmarks per hand
* Distinguishes between left and right hands
* Supports one-hand and two-hand gestures
* Provides landmark data for classification and visualisation
* Runs continuously during practice

Using landmarks instead of raw video frames keeps the recognition pipeline lightweight and reduces sensitivity to factors such as background appearance and skin tone.

---

### 2. Deep Learning Sign Classification

Static BSL signs are recognised using a compact neural network trained with **PyTorch**.

After training, the model is exported to **ONNX** and loaded in the browser using **ONNX Runtime Web**.

#### Model Input

| Sign Type        | Input Features |
| ---------------- | -------------: |
| Single-hand sign |             63 |
| Two-hand sign    |            126 |

#### Model Output

The classifier produces probabilities across the supported BSL sign classes. The highest-scoring prediction is used to provide the recognised sign and confidence level to the learner.

Running inference in the browser removes the need for a dedicated inference server and helps keep interaction responsive.

---

### 3. Feature Normalisation

Before classification, landmark features are normalised using the same statistics used during model training.

Z-score normalisation parameters are stored in:

```text
bsl_norm.json
```

This helps ensure that the feature distribution during browser inference remains consistent with the data used to train the model.

---

### 4. Dynamic Gesture Recognition

Not every sign can be recognised from a single frame.

For gestures where **movement, trajectory, timing, or repetition** are important, the system uses **Dynamic Time Warping (DTW)** to compare the learner's motion sequence against reference gesture sequences.

DTW is useful because users naturally perform signs at different speeds. It allows sequences to be stretched or compressed in time while still comparing the overall motion pattern.

This enables support for dynamic gestures such as:

* **Yes**
* **No**
* Other motion-dependent signs

---

## Real-Time Learning Feedback

Recognition is only one part of the project. The system is designed as an interactive learning tool rather than simply a sign classifier.

During practice, users receive:

* **Predicted sign labels**
* **Confidence scores**
* **Live hand-landmark overlays**
* **Skeleton visualisation**
* **Correct / incorrect feedback**
* **Camera and hand-position guidance**
* **Motion progress feedback for dynamic gestures**

The goal is to create a continuous feedback loop in which learners can perform a sign, see the system's response, adjust their gesture, and try again.

The project evaluation specifically used gesture recognition, confidence scores, and visual overlays as its main real-time feedback mechanisms.

---

## Application Preview

![Picture1.png](bsl-mp%2Fpublic%2FImages%2FPicture1.png) 

![Picture2.png](bsl-mp%2Fpublic%2FImages%2FPicture2.png) 

![Picture3.png](bsl-mp%2Fpublic%2FImages%2FPicture3.png)

---

## System Architecture

![System Architecture](bsl-mp%2Fpublic%2FImages%2FPicture4.png)

At a high level, the application combines a React-based learning interface with MediaPipe hand tracking, feature processing, and ONNX inference:

```text
┌─────────────────────┐
│       Learner       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Webcam / AR UI   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   MediaPipe Hands   │
│   21 landmarks/hand │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Processing  │
│ + Normalisation     │
└──────────┬──────────┘
           │
           ├─────────────────────┐
           ▼                     ▼
┌─────────────────────┐   ┌─────────────────────┐
│    ONNX Classifier  │   │         DTW         │
│    Static Gestures  │   │  Dynamic Gestures   │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           └────────────┬────────────┘
                        ▼
             ┌─────────────────────┐
             │  Real-Time Feedback │
             │ Prediction          │
             │ Confidence          │
             │ Visual Overlay      │
             └─────────────────────┘
```

The project's full architecture follows this same flow from the learner-facing React interface through MediaPipe, feature processing, ONNX Runtime, and the BSL sign definitions.

---

## Training & Deployment Pipeline

The machine-learning workflow separates model training from browser inference:

```text
Raw Landmark Captures
        │
        ▼
Pre-processing
(normalisation / cleaning)
        │
        ▼
PyTorch Training
        │
        ▼
Model Checkpoint
(.pth)
        │
        ▼
ONNX Export
        │
        ▼
ONNX Runtime Web
        │
        ▼
Browser-Based BSL Tutor
```

This approach allows the model to be trained using PyTorch while keeping the deployed application lightweight and platform-independent.

---

## Technology Stack

| Component           | Technology                                         |
| ------------------- | -------------------------------------------------- |
| Frontend            | React                                              |
| Language            | TypeScript / JavaScript                            |
| Hand Tracking       | MediaPipe Hands                                    |
| Model Training      | PyTorch                                            |
| Model Deployment    | ONNX                                               |
| Browser Inference   | ONNX Runtime Web                                   |
| Dynamic Recognition | Dynamic Time Warping                               |
| Input               | Webcam                                             |
| Feedback            | Landmark overlays, predictions & confidence scores |

---

## Why Browser-Side Inference?

The application is designed to perform inference locally rather than continuously sending webcam data to a backend.

This provides several advantages:

* **Low latency** — predictions can be generated immediately
* **Privacy** — inference remains on the user's device
* **Accessibility** — no specialised hardware or native application is required
* **Scalability** — inference does not depend on server compute
* **Cross-device deployment** — the system can run on standard browser-capable devices

The project's deployment strategy specifically targets zero-install browser access, on-device inference, and reduced dependence on server infrastructure.

---

## Evaluation

The prototype was evaluated with **14 participants** who had no formal BSL training. Participants practised a selection of static and dynamic gestures before completing a questionnaire assessing usability, feedback, engagement, accessibility, and perceived learning effectiveness.

Results were positive across the main evaluation areas:

| Evaluation Area                           | Mean Score / 5 |
| ----------------------------------------- | -------------: |
| Understanding hand orientation            |       **4.33** |
| Understanding signs from different angles |       **4.33** |
| Recognising subtle finger positioning     |       **4.33** |
| Precision tracking usefulness             |       **4.11** |
| Engagement during practice                |       **4.33** |
| Motivation to continue practising         |       **4.44** |
| Potential to improve accessibility        |       **4.33** |
| Preference over textbook learning         |       **4.22** |

The highest score was for **motivation to continue practising BSL (4.44/5)**, while participants also responded positively to the system's visualisation, engagement, and accessibility.

---

## Current Limitations

The current prototype has several areas for further improvement:

* Hand tracking can be affected by **poor lighting and occlusion**
* Camera positioning can influence recognition quality
* A limited training dataset restricts generalisation across different users
* Lower-powered devices may experience reduced frame rates
* Dynamic gesture feedback could be made more interpretable
* Confidence scores indicate model certainty but do not always explain *why* a gesture is incorrect

These limitations form part of the project's future development direction, particularly around improving tracking reliability and providing more explicit corrective guidance.

---

## Future Improvements

Potential extensions include:

* Larger and more diverse BSL training datasets
* Improved left/right-hand handling
* More dynamic BSL gestures
* Temporal smoothing for more stable tracking
* Joint-level error detection
* Explicit corrective instructions for incorrect gestures
* Directional feedback showing users how to adjust their hands
* Improved mobile performance
* More advanced 3D gesture visualisation
* Expansion from isolated signs toward larger BSL vocabulary and learning exercises

---

## Project Motivation

Traditional self-learning resources can demonstrate what a sign should look like, but they cannot tell learners whether they are performing it correctly.

This project explores how **computer vision + lightweight machine learning + interactive visual feedback** can bridge that gap.

Rather than acting only as a recognition system, the goal is to provide an accessible learning environment where users can:

**observe → perform → receive feedback → correct → repeat**

The user study suggests that this feedback-driven approach can improve learners' understanding of hand orientation and gesture clarity while encouraging continued practice.

---

## Research

This implementation accompanies the research project:

**Enhancing Sign Language Learning Through Augmented Reality Feedback**

The research explores the design, implementation, and evaluation of a lightweight browser-based BSL learning system combining MediaPipe hand tracking, neural-network classification, DTW-based temporal assessment, and real-time learner feedback.

---

## Acknowledgements

Built using **MediaPipe**, **PyTorch**, **ONNX Runtime Web**, and modern browser technologies, with the aim of making interactive British Sign Language learning more accessible.
