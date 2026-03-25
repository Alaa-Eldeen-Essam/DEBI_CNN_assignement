# Face Mask Detector - MobileNetV2 + YOLO + FastAPI + Streamlit
[![CNN_Assignment_DEBI_CI/CD](https://github.com/Alaa-Eldeen-Essam/DEBI_CNN_assignement/actions/workflows/main.yml/badge.svg)](https://github.com/Alaa-Eldeen-Essam/DEBI_CNN_assignement/actions/workflows/main.yml)

This repository contains a face-mask detection demo with two inference paths:

- `MobileNetV2` for whole-image classification
- `YOLO` for per-face detection with bounding boxes on images and progressive video playback

The backend is a FastAPI service and the frontend is a Streamlit app.

Related notebooks:
- [Mask detection using MobileNet notebook](https://colab.research.google.com/drive/1dTv1GSZkwF0OM8sBwFXUcaxVkzFbz04g?usp=sharing)
- [Mask detector using YOLO notebook](https://colab.research.google.com/drive/1BiQw_EBlPL5y71P5ayZqEKyCcCGkOd4g?usp=sharing)

## Project Structure

```text
CNN_assignement/
|-- backend/
|   |-- main.py
|   `-- models/
|       |-- best.pt
|       |-- mobilenetv2_mask.h5
|       |-- mobilenetv2_mask.keras
|       |-- yolo_11_new.pt
|       |-- yolo_11_new_2.pt
|       `-- yolo_11_new_3.pt
|-- frontend/
|   `-- app.py
|-- notebooks/
|-- test_samples/
|-- Makefile
|-- README.md
`-- requirements.txt
```

## Requirements

Recommended environment:

- Python `3.11`
- TensorFlow `2.19.0`
- Keras `3.x`

All committed dependencies live in the root `requirements.txt`. There are no separate `backend/requirements.txt` or `frontend/requirements.txt` files in this repo.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer Conda:

```bash
conda create -n face_mask python=3.11
conda activate face_mask
pip install -r requirements.txt
```

## Run

Start the backend from the repo root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

You can also start the backend with:

```bash
python backend/main.py
```

Start the frontend from the repo root in a second terminal:

```bash
streamlit run frontend/app.py
```

Local URLs:

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

## API Endpoints

| Method | Path                 | Description |
|--------|----------------------|-------------|
| GET    | `/`                  | API status check |
| GET    | `/health`            | Check that the models are loaded |
| POST   | `/predict/mobilenet` | Whole-image classification |
| POST   | `/predict/yolo`      | Face detection with bounding boxes |
| POST   | `/stream/yolo-video` | Create a progressive YOLO video stream session |
| GET    | `/stream/yolo-video/{stream_id}` | Read annotated MJPEG frames for a created session |
| POST   | `/stream/yolo-video/{stream_id}/stop` | Stop an active video stream session |

Prediction endpoints accept `multipart/form-data` with a single `file` field.
- `/predict/mobilenet` and `/predict/yolo` accept image uploads (`jpg`, `jpeg`, `png`).
- `POST /stream/yolo-video` accepts video uploads (`mp4`, `mov`, `avi`, `mkv`) and returns a stream session id plus stream URL.
- `GET /stream/yolo-video/{stream_id}` returns the MJPEG stream of annotated frames.
- `POST /stream/yolo-video/{stream_id}/stop` stops playback and releases temporary files.

### Example request

```bash
curl -X POST http://localhost:8000/predict/mobilenet -F "file=@photo.jpg"
```

### Example MobileNet response

```json
{
  "model": "MobileNetV2",
  "label": "With Mask",
  "confidence": 97.43,
  "probability": {
    "With Mask": 97.43,
    "Without Mask": 2.57
  }
}
```

### Example YOLO response

```json
{
  "model": "YOLO26n",
  "total": 1,
  "with_mask": 1,
  "without_mask": 0,
  "detections": [
    {
      "label": "With Mask",
      "confidence": 96.7,
      "box": [44.2, 18.3, 180.5, 201.7]
    }
  ]
}
```

### Example video streaming session request

```bash
curl -X POST http://localhost:8000/stream/yolo-video -F "file=@clip.mp4"
```

## Notes

- The MobileNet output is interpreted as `0 -> With Mask` and `1 -> Without Mask`.
- The backend loads both models at startup and keeps them in memory.
- MobileNet load order is `mobilenetv2_mask.h5` first, then `mobilenetv2_mask.keras` as fallback.
- The Streamlit app starts a backend video stream session, plays it in the page, and lets you stop it explicitly.
- YOLO model paths and inference thresholds are currently hardcoded in `backend/main.py`.
- Progressive playback is silent and does not generate a downloadable annotated output file.
