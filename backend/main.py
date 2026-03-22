import io
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Face Mask Detector", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE    = 224
CLASS_NAMES = ["With Mask", "Without Mask"]

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading model...")
mobilenet_model = tf.keras.models.load_model("models/mobilenetv2_mask.keras")
print("Model ready.")

# ── Response schema ────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    model:       str
    label:       str
    confidence:  float
    probability: dict[str, float]

# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def run_prediction(model, image_array: np.ndarray) -> PredictionResponse:
    prob_without_mask = float(model.predict(image_array, verbose=0)[0][0])
    prob_with_mask    = 1.0 - prob_without_mask
    label             = "Without Mask" if prob_without_mask >= 0.5 else "With Mask"
    confidence        = prob_without_mask if label == "Without Mask" else prob_with_mask

    return PredictionResponse(
        model="MobileNetV2",
        label=label,
        confidence=round(confidence * 100, 2),
        probability={
            "With Mask":    round(prob_with_mask    * 100, 2),
            "Without Mask": round(prob_without_mask * 100, 2),
        }
    )

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Face Mask Detection API is running."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": "MobileNetV2"}


@app.post("/predict/mobilenet", response_model=PredictionResponse)
async def predict_mobilenet(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPEG / PNG images are supported.")
    image_bytes = await file.read()
    image_array = preprocess(image_bytes)
    return run_prediction(mobilenet_model, image_array)