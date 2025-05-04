# FaceTracker Library 📸👤

A Python library for real-time face detection and tracking with support for multiple algorithms (MediaPipe, Haar, LBP, YuNet), camera orientation correction, and display rotation.

---

## Features ✨
- **Multiple Algorithms**: Choose between MediaPipe, Haar, LBP, or YuNet.
- **Orientation Handling**: Correct for camera mounting angles and display rotations.
- **Overlays**: Display FPS, resolution, face offset, and axis labels.
- **Threaded Execution**: Run the tracking loop in the background.
- **Coordinate Transforms**: Accurate face position offsets relative to display orientation.

---

## Installation 🛠️

1. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy
