# Face Mask Detector — FastAPI + Streamlit

## Project Structure

```
face-mask-app/
├── backend/
│   ├── main.py               # FastAPI app (loads both .h5 models, exposes REST endpoints)
│   ├── requirements.txt
│   └── models/               # ← place your .h5 files here
│       ├── efficientnetb0_mask.h5
│       └── mobilenetv2_mask.h5
│
├── frontend/
│   ├── app.py                # Streamlit UI
│   └── requirements.txt
│
└── README.md
```

---

## Setup & Run

### 1 — Place your models
Copy the two `.h5` files from Colab into `backend/models/`.

### 2 — Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger docs available at: http://localhost:8000/docs

### 3 — Frontend (separate terminal)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

Opens at: http://localhost:8501

---

## API Endpoints

| Method | Path                    | Description                       |
|--------|-------------------------|-----------------------------------|
| GET    | `/health`               | Check API status & loaded models  |
| POST   | `/predict/efficientnet` | Predict with EfficientNetB0       |
| POST   | `/predict/mobilenet`    | Predict with MobileNetV2          |
| POST   | `/predict/both`         | Run both models, compare results  |

All prediction endpoints accept `multipart/form-data` with a single `file` field (JPEG or PNG).

### Example with curl
```bash
curl -X POST http://localhost:8000/predict/both \
  -F "file=@photo.jpg"
```

### Example response (`/predict/both`)
```json
{
  "efficientnet": {
    "model": "EfficientNetB0",
    "label": "With Mask",
    "confidence": 97.43,
    "probability": { "With Mask": 97.43, "Without Mask": 2.57 }
  },
  "mobilenet": {
    "model": "MobileNetV2",
    "label": "With Mask",
    "confidence": 96.11,
    "probability": { "With Mask": 96.11, "Without Mask": 3.89 }
  }
}
```

---

## Notes
- The sigmoid output of both models maps: `0 → WithMask`, `1 → WithoutMask`
  (matches the `class_indices` order from `flow_from_directory` alphabetically).
- Both models are loaded once on startup and kept in memory.
- The frontend sidebar lets you switch between models and check API health without restarting.