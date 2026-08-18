# UAV Object Detection and Navigation using Computer Vision

## Overview

This project was developed as part of an internship at NIT Trichy to investigate vision-based perception and navigation for Unmanned Aerial Vehicles (UAVs). The work explored two different approaches for understanding the environment from RGB video:

1. **ORB-SLAM3** – investigated for visual localization and mapping.
2. **YOLOv8 + Depth Anything V2** – developed for object detection, monocular depth estimation, and depth-aware navigation.

The objective was to determine a practical computer-vision-based approach for UAV scene understanding and obstacle-aware navigation using RGB video.

---

## Approaches Implemented

### 1. ORB-SLAM3 Based Approach

ORB-SLAM3 was implemented and evaluated as a visual SLAM approach for estimating camera motion and building a representation of the surrounding environment from RGB video.

The implementation involved configuring the camera parameters, running visual tracking, generating camera trajectory information, and analysing the resulting localization and mapping behaviour.

### Observed Limitations

Although ORB-SLAM3 provided visual tracking and mapping information, the results were not sufficiently reliable for the intended UAV navigation scenario.

The main limitations observed during testing were:

* Tracking and localization were not consistently reliable for the tested video conditions.
* The SLAM output was primarily focused on camera localization and mapping rather than directly providing obstacle-distance information.
* The resulting information was therefore not sufficient by itself for reliable obstacle-aware navigation.
* Based on these observations, an alternative perception approach using object detection and monocular depth estimation was investigated.

---

## 2. YOLOv8 + Depth Anything V2 Approach

The second approach combined **YOLOv8 object detection** with **Depth Anything V2 monocular depth estimation**.

### YOLOv8

A COCO-pretrained YOLOv8 model was used to detect general objects in the scene. A custom-trained YOLO model was also integrated into the pipeline for application-specific object detection.

### Depth Anything V2

Depth Anything V2 was used to estimate relative depth information from a single RGB frame. The estimated depth was combined with detected objects to provide information about their relative proximity within the scene.

### Navigation

The object detection and depth information were processed by a navigation decision module. Based on the perceived scene, the system generated movement decisions such as:

* `FORWARD`
* `MOVE_LEFT`
* `MOVE_RIGHT`
* `STOP`

This approach provided more useful scene information for the intended obstacle-aware navigation task than the SLAM-based approach tested earlier.

---

## System Pipeline

```text
                    RGB Video
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
         YOLOv8           Depth Anything V2
     Object Detection     Monocular Depth
              │                 │
              └────────┬────────┘
                       │
                       ▼
              Scene Understanding
                       │
                       ▼
             Navigation Decision
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       FORWARD      MOVE LEFT    MOVE RIGHT
                       │
                       ▼
                      STOP
```

---

## Comparison of Approaches

| Approach                   | Primary Purpose                             | Observation                                                                                   |
| -------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------- |
| ORB-SLAM3                  | Visual localization and mapping             | Implemented and evaluated, but not sufficiently reliable for the intended navigation scenario |
| YOLOv8 + Depth Anything V2 | Object detection and depth-aware perception | Provided more useful information for the tested navigation pipeline                           |

The comparison led to the adoption of the **YOLOv8 + Depth Anything V2 approach** as the more practical perception pipeline for the current navigation objective.

---

## Experimental Result

The developed Depth Anything V2 navigation pipeline was tested using RGB video.

For one test video (`test_5.mp4`):

* Resolution: **1248 × 624**
* Input FPS: **30 FPS**
* Total frames: **400**
* Processed frames: **400**
* Skipped frames: **0**
* Total COCO detections: **938**
* Average COCO detections/frame: **2.35**
* Average pipeline FPS on CPU: **0.28 FPS**

Navigation decisions obtained during the test were:

| Decision   | Frames | Percentage |
| ---------- | -----: | ---------: |
| FORWARD    |     67 |      16.8% |
| MOVE_LEFT  |    193 |      48.2% |
| MOVE_RIGHT |    140 |      35.0% |
| STOP       |      0 |       0.0% |

The experiment demonstrated that the complete pipeline could process every frame and generate navigation decisions. However, the low CPU processing rate indicates that further optimization is required for real-time UAV deployment.

---

## Project Structure

```text
uav-object-detection-navigation/
│
├── method_1_orb_slam3/
│   └── README.md
│
├── method_2_depth_yolo/
│   ├── depth_anything_v2/
│   │   ├── dinov2_layers/
│   │   ├── util/
│   │   ├── dinov2.py
│   │   └── dpt.py
│   │
│   ├── scripts/
│   │   └── depth_navigation_updated.py
│   │
│   ├── README.md
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

Large model weights, videos, test datasets, generated outputs, and virtual environments are intentionally excluded from the repository.

---

## Technologies Used

* **Python**
* **YOLOv8**
* **Depth Anything V2**
* **ORB-SLAM3**
* **OpenCV**
* **PyTorch**
* **Computer Vision**
* **Monocular Depth Estimation**
* **Visual SLAM**
* **UAV Navigation**

---

## Limitations

The current implementation has several limitations:

* Depth Anything V2 provides monocular relative depth rather than direct metric depth.
* CPU execution results in low processing speed for the complete pipeline.
* Object detection performance depends on the training data and visual conditions.
* Navigation decisions are based on rule-based scene interpretation.
* ORB-SLAM3 tracking was not sufficiently reliable for the intended navigation scenario.

---

## Future Improvements

Future development can focus on:

* GPU acceleration and runtime optimization.
* TensorRT or model-optimization techniques for improved FPS.
* Improved object detection and obstacle classification.
* More accurate depth-to-distance estimation.
* Integration of RGB-D or stereo cameras for direct depth information.
* Improved navigation logic for more complex UAV environments.
* Real-world UAV flight testing and validation.

---

## Internship Outcome

The internship resulted in the implementation and evaluation of two computer-vision-based approaches for UAV perception and navigation. ORB-SLAM3 was explored for visual localization and mapping, while the YOLOv8 and Depth Anything V2 pipeline provided object detection, depth perception, and navigation-decision generation from RGB video. The evaluation helped identify the more suitable approach for the intended navigation application and highlighted the requirements for future real-time deployment.
