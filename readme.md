# Video Object Motion and Camera Compensation Tracker

> A Python-based computer vision tool for analyzing object movement in video streams with simultaneous camera motion estimation (using ORB feature matching and RANSAC) and YOLO object tracking.

---

## 🚀 Features

* **Camera Motion Estimation:** Utilizes ORB feature detection, descriptor matching, and RANSAC-based affine estimation (`cv2.estimateAffinePartial2D`) to calculate camera translation (`dx`, `dy`) and scale changes (zoom) per frame[cite: 1].
* **Object Tracking:** Integrates YOLOv8 and ByteTrack to detect, track, and classify objects across video frames[cite: 3].
* **Motion Compensation:** Distinguishes between apparent screen motion and real-world object movement by subtracting camera motion vectors.
* **Time-to-Collision (TTC):** Automatically calculates estimated time-to-collision for approaching objects based on real-world area growth rates.
* **Detailed Reporting:** Generates an annotated output video (`.avi`) and a comprehensive JSON report containing motion statistics and descriptions.

---

## Usage

Clone the repository, open the project folder in your terminal, and run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

Place your input video file in the project directory and update the VIDEO_PATH variable in main.py if necessary (defaults to ./test_video.mp4)

Run the processing pipeline using:
```bash
python main.py
```

The script will output an annotated video (./output_tracked.avi) and a JSON log file (./object_motion_camera_compensated.json) with detailed interval statistics.