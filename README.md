# Face Mask Detector — MobileNetV2 + FastAPI + Streamlit
[![Python Scaffold cloud course](https://github.com/Alaa-Eldeen-Essam/DEBI_CNN_assignement/actions/workflows/main.yml/badge.svg)](https://github.com/Alaa-Eldeen-Essam/DEBI_CNN_assignement/actions/workflows/main.yml)
[Mask detection using Mobilenet notebook](https://colab.research.google.com/drive/1dTv1GSZkwF0OM8sBwFXUcaxVkzFbz04g?usp=sharing)
[Mask detector using YOLO notebook](https://colab.research.google.com/drive/1BiQw_EBlPL5y71P5ayZqEKyCcCGkOd4g?usp=sharing)
## Project Structure

```
face-mask-app/
├── backend/
│   ├── main.py               # FastAPI app (loads MobileNetV2, exposes REST endpoint)
│   ├── requirements.txt
│   └── models/               # ← place your model file here
│       └── mobilenetv2_mask.weights.h5   # weights-only file (recommended)
│       └── mobilenetv2_mask.keras        # full model (requires matching Keras version)
│
├── frontend/
│   ├── app.py                # Streamlit UI
│   └── requirements.txt
│
└── README.md
```

---

## Python & Keras Version Requirements

| Component  | Requirement                        |
|------------|------------------------------------|
| Python     | **3.11+** (recommended) / 3.10 ok with weights-only approach |
| TensorFlow | 2.19.0                             |
| Keras      | 3.13.2 (requires Python 3.11+)     |

> The model was trained in Colab with Keras 3.13.2. Loading the full `.keras` file
> locally requires the same Keras version, which in turn requires Python 3.11+.

---

## Setup

### Option A — Python 3.11 (recommended, full .keras support)

```bash
conda create -n face_mask python=3.11
conda activate face_mask
cd backend
pip install -r requirements.txt
```

### Option B — Python 3.10 (weights-only, no environment upgrade)

Re-save from Colab:
```python
mobilenet_model.save_weights('mobilenetv2_mask.weights.h5')
files.download('mobilenetv2_mask.weights.h5')
```

Place `mobilenetv2_mask.weights.h5` in `backend/models/` and use the
weights-only loader in `main.py` (see comments inside the file).

---

## Run

```bash
# Terminal 1 — backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

- API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

---

## API Endpoints

| Method | Path                    | Description                      |
|--------|-------------------------|----------------------------------|
| GET    | `/`                     | API status check                 |
| GET    | `/health`               | Check model is loaded            |
| POST   | `/predict/mobilenet`    | Predict with MobileNetV2         |

All prediction endpoints accept `multipart/form-data` with a single `file` field (JPEG or PNG).

### Example with curl
```bash
curl -X POST http://localhost:8000/predict/mobilenet \
  -F "file=@photo.jpg"
```

### Example response
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

---

## Notes
- The sigmoid output maps: `0 → WithMask`, `1 → WithoutMask`
- The model is loaded once on startup and kept in memory
- The Streamlit sidebar lets you set the API URL and run a health check
- Labels `.txt` files in the YOLO dataset live alongside images in the same folder,
  not in a separate `labels/` directory — copy them over before training with Ultralytics