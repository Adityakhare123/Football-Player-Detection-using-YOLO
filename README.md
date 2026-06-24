# Football Player Detection System Using YOLOv8

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:020617,50:22c55e,100:3b82f6&height=220&section=header&text=Football%20Player%20Detection%20System&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=YOLOv8%20%7C%20Ultralytics%20%7C%20Python%20%7C%20Computer%20Vision%20%7C%20Sports%20Analytics&descAlignY=60&descAlign=50&descSize=17" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv8-111827?style=for-the-badge&logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/Ultralytics-000000?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Roboflow-6706CE?style=for-the-badge&logoColor=white" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Sports%20Analytics-22c55e?style=for-the-badge" />
</p>

<p align="center">
  A computer vision project for detecting football players from match videos using YOLO-based object detection.
</p>

---

## Project Overview

**Football Player Detection System** is a computer vision project built using **Python**, **YOLOv8**, **Ultralytics**, and **OpenCV**.

The project focuses on detecting football players from gameplay videos using object detection. It includes a YOLO-format dataset, training files, and an inference script that can process football match videos and save annotated detection outputs.

This project is designed for learning, experimentation, sports analytics practice, and portfolio demonstration.

---

## Key Features

* Football player detection from match videos
* YOLOv8 object detection pipeline
* Video inference support
* Bounding box generation
* Confidence score output
* Annotated video output saving
* Custom dataset support
* YOLO-format training dataset included
* Roboflow dataset structure support
* Training, validation, and test folders included
* Clean GitHub-ready project structure
* Extendable into a complete football analytics system

---

## Tech Stack

<p align="center">
  <img src="https://skillicons.dev/icons?i=python,opencv,pytorch,git,github,vscode" />
</p>

### Core Technologies

* Python
* YOLOv8
* Ultralytics
* OpenCV
* Roboflow
* PyTorch
* Computer Vision
* Deep Learning
* Object Detection

---

## Project Structure

```text
Football-Player-Detection-using-YOLO/
│
├── yolo_inference.py
├── README.md
├── .gitignore
│
├── training/
│   └── football-players-detection-/
│       ├── train/
│       ├── valid/
│       ├── test/
│       ├── data.yaml
│       ├── README.dataset.txt
│       └── README.roboflow.txt
│
├── input_videos/
│   └── 08fd33_4.mp4
│
└── runs/
    └── detect/
        ├── predict/
        ├── predict2/
        └── predict3/
```

> Note: The `input_videos/`, `runs/`, trained weights, video files, and other large generated outputs are ignored in Git to keep the repository lightweight.

---

## Model Information

The project currently uses YOLOv8 for object detection.

Default model used in the inference script:

```text
yolov8l
```

For custom trained model inference, replace the model path in `yolo_inference.py` with your trained weight file:

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

or:

```python
model = YOLO("models/best.pt")
```

Recommended custom class mapping:

```text
0 = Football Player
```

For future versions, the model can be extended to detect:

```text
0 = Player
1 = Football
2 = Goalkeeper
3 = Referee
```

---

## How the System Works

```text
User provides football video input
        ↓
YOLOv8 model is loaded
        ↓
Input video is passed through the detection model
        ↓
Football players are detected frame by frame
        ↓
Bounding boxes and confidence scores are generated
        ↓
Annotated output video is saved
        ↓
Detection results can be reviewed from the runs folder
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Adityakhare123/Football-Player-Detection-using-YOLO.git
cd Football-Player-Detection-using-YOLO
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install ultralytics opencv-python numpy matplotlib
```

---

## Run Detection

Place your input football video inside the `input_videos/` folder.

Example:

```text
input_videos/08fd33_4.mp4
```

Run the inference script:

```bash
python yolo_inference.py
```

YOLO will process the video and save the annotated output inside:

```text
runs/detect/predict/
```

---

## Inference Script

The current inference script loads a YOLOv8 model and runs prediction on a football video.

```python
from ultralytics import YOLO

model = YOLO('yolov8l')

result = model.predict('input_videos/08fd33_4.mp4', save=True)

print(result[0])
print("________________________________________________")

for box in result[0].boxes:
    print(box)
```

The script performs the following steps:

* Loads the YOLOv8 model
* Reads the input video
* Runs object detection
* Saves the annotated output
* Prints detection result information
* Prints bounding box details

---

## Training Dataset

The project includes a YOLO-format football player detection dataset inside the `training/` folder.

Dataset structure:

```text
training/
└── football-players-detection-/
    ├── train/
    ├── valid/
    ├── test/
    └── data.yaml
```

The `data.yaml` file defines the dataset path, train/validation/test folders, number of classes, and class names.

---

## Training

To train a custom YOLO model using the dataset, use the Ultralytics training command:

```bash
yolo task=detect mode=train model=yolov8l.pt data=training/football-players-detection-/data.yaml epochs=50 imgsz=640
```

After training, the best model is usually saved at:

```text
runs/detect/train/weights/best.pt
```

Use this trained model for custom football player detection:

```python
model = YOLO("runs/detect/train/weights/best.pt")
```

---

## requirements.txt

Recommended `requirements.txt`:

```txt
ultralytics
opencv-python
numpy
matplotlib
roboflow
```

For deployment environments, use:

```txt
ultralytics
opencv-python-headless
numpy
matplotlib
roboflow
```

---

## Output

YOLO saves detection results inside the `runs/` directory by default:

```text
runs/detect/predict/
```

Example output files may include:

```text
08fd33_4.avi
08fd33_4.mp4
```

The output folder is ignored from Git because it contains generated detection results.

---

## Current Capability

The current version supports:

* Loading YOLOv8 detection model
* Running inference on football match videos
* Detecting objects in video frames
* Saving annotated detection output
* Printing detection result details
* Printing bounding box information
* Using YOLO-format training dataset
* Supporting future custom-trained model weights

---

## Common Issues

### 1. ModuleNotFoundError: No module named ultralytics

Fix:

```bash
pip install ultralytics
```

### 2. OpenCV import error

Fix:

```bash
pip install opencv-python
```

For deployment environments, use:

```bash
pip install opencv-python-headless
```

### 3. Input video not found

Make sure the video exists in the correct folder:

```text
input_videos/08fd33_4.mp4
```

### 4. Model file not found

If using a custom trained model, make sure the file path is correct:

```text
runs/detect/train/weights/best.pt
```

or:

```text
models/best.pt
```

### 5. Large file push issue on GitHub

Do not push:

```text
.pt model files
.mp4 video files
.avi video files
runs/ output folder
input_videos/ folder
```

These should remain ignored using `.gitignore`.

---

## Recommended .gitignore

```gitignore
# Python
__pycache__/
*.pyc
.env
venv/
.venv/

# YOLO generated outputs
runs/
*.cache

# Large media files
*.mp4
*.avi
*.mov
*.mkv

# Model weights
*.pt
*.onnx
*.engine

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db
```

---

## Future Improvements

* Add Streamlit web application
* Add video upload interface
* Add image upload interface
* Add real-time webcam detection
* Add custom trained football player model
* Add football detection
* Add player tracking
* Add team classification
* Add jersey number recognition
* Add referee detection
* Add goalkeeper detection
* Add player movement trails
* Add ball tracking
* Add possession analysis
* Add pass detection
* Add shot detection
* Add heatmap generation
* Add match analytics dashboard
* Add model evaluation metrics
* Add sample output screenshots
* Deploy project on Streamlit Cloud or Hugging Face Spaces

---

## Current Project Status

The current project includes:

* YOLO inference script
* YOLO-format football detection dataset
* Training, validation, and test folders
* Dataset configuration file
* Git ignore setup
* GitHub-ready folder structure

Areas that can be improved later:

* Add trained custom model weights link
* Add sample output images
* Add model performance metrics
* Add frontend interface
* Add deployment support
* Add football analytics features

---

## Author

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=soft&color=0:020617,100:22c55e&height=120&section=footer&text=Aditya%20Khare&fontSize=36&fontColor=ffffff&animation=fadeIn" />
</p>

**Aditya Khare**

<p>
  <a href="https://github.com/Adityakhare123">
    <img src="https://img.shields.io/badge/GitHub-Adityakhare123-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
  <a href="https://github.com/Adityakhare123/Football-Player-Detection-using-YOLO">
    <img src="https://img.shields.io/badge/Project-Football--Player--Detection-22c55e?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

---

## Support

If you like this project, consider giving it a star on GitHub.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,50:22c55e,100:020617&height=120&section=footer" />
</p>
