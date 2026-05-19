# Real-Time PPE Compliance Detection System

A real-time Personal Protective Equipment (PPE) violation detection system using YOLOv8 + DeepSORT tracking with zone-based safety monitoring.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Flask](https://img.shields.io/badge/Flask-SocketIO-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)

## Overview

This system detects PPE violations (missing hardhats, safety vests) on construction sites in real-time. It combines:
- **YOLOv8n** fine-tuned on 25 PPE-related classes
- **DeepSORT** for persistent multi-object tracking across frames
- **Zone-based monitoring** for restricted area violation alerts
- **Flask + WebSocket** streaming for browser-based real-time visualization

## Architecture

```
Input Video/Stream → YOLOv8n Detection → DeepSORT Tracking → Zone Violation Check → Annotated Output
                                                                                   ↓
                                                                        Flask WebSocket → Browser UI
```

## Results

| Metric | Value |
|--------|-------|
| mAP@0.5 (all 25 classes) | 52.5% |
| Hardhat Detection AP | 67.8% |
| Safety Vest Detection AP | 71.2% |
| Person Detection AP | 71.5% |
| Inference Speed | 3.1ms/frame (T4 GPU) |
| Model Size | 6.2 MB |
| Real-time FPS | 30+ |

## Dataset

- **Source:** [Construction-Site-Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) (Roboflow)
- **Train:** 521 images | **Val:** 114 images
- **Classes:** 25 (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, Gloves, Goggles, etc.)
- **Format:** YOLO (normalized bounding boxes)

## Project Structure

```
├── src/
│   ├── tracker.py          # Main detection + tracking + zone violation engine
│   ├── train.py            # Fine-tuning script
│   └── evaluate.py         # Model evaluation (mAP, precision, recall)
├── app/
│   ├── app.py              # Flask + SocketIO web server
│   └── templates/
│       └── index.html      # Real-time browser dashboard
├── configs/
│   └── ppe_dataset.yaml    # Dataset configuration
├── notebooks/
│   └── train_colab.ipynb   # Google Colab training notebook
├── best.pt                 # Fine-tuned PPE model weights
├── yolov8n.pt              # Base pretrained weights (COCO)
├── Dockerfile              # Container deployment
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Detection (CLI)
```bash
# Fine-tuned PPE mode
python src/tracker.py --source input_video.mp4 --model best.pt --mode ppe --conf 0.4

# Pretrained COCO mode
python src/tracker.py --source input_video.mp4 --model yolov8n.pt --mode pretrained
```

### Run Web App
```bash
python app/app.py
# Open browser at http://localhost:5000
```

### Docker
```bash
docker build -t ppe-detector .
docker run -p 5000:5000 ppe-detector
```

## Training

Fine-tuning was done on Google Colab (Tesla T4 GPU):
```bash
python src/train.py --base_model yolov8n.pt --data configs/ppe_dataset.yaml --epochs 50 --imgsz 640
```

Or use the provided notebook: `notebooks/train_colab.ipynb`

### Training Configuration
- **Base Model:** YOLOv8n (nano — optimized for real-time edge deployment)
- **Epochs:** 50
- **Optimizer:** SGD (lr=0.01, momentum=0.937)
- **Image Size:** 640×640
- **Hardware:** Tesla T4 GPU (Google Colab)

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| YOLOv8**n** (nano) | Real-time safety systems need low latency; 3ms inference enables 30+ FPS on edge devices |
| DeepSORT | Persistent ID tracking prevents double-counting workers across frames |
| Zone-based alerts | Construction sites have restricted areas; spatial context adds safety value |
| Flask + WebSocket | Real-time streaming without polling; lightweight deployment |

## How to Improve

- Scale to YOLOv8s/m for +5-10% mAP with acceptable latency tradeoff
- Augment dataset with mosaic, mixup, copy-paste strategies
- Add more training data (current 521 images is small for production)
- Knowledge distillation from YOLOv8x teacher model
- Active learning loop for edge-case mining

## Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Tracking:** DeepSORT (Kalman Filter + Hungarian Algorithm)
- **Backend:** Flask, Flask-SocketIO
- **Frontend:** HTML5, JavaScript, WebSocket
- **Containerization:** Docker
- **Training:** Google Colab, PyTorch


