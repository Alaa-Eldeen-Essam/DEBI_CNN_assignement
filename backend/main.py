import io
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# App setup
app = FastAPI(title="Face Mask Detector", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 224

# Load models once at startup
print("Loading MobileNetV2...")
mobilenet_model = tf.keras.models.load_model("models/mobilenetv2_mask.keras")

print("Loading YOLO26...")
yolo_model = YOLO("models/yolo_11_new_3.pt")  # Swap to best.onnx if preferred.
# yolo_model = YOLO("models/best.onnx")

print("All models ready.")


# Response schemas
class MobileNetResponse(BaseModel):
    model: str
    label: str
    confidence: float
    probability: dict[str, float]


class YOLODetection(BaseModel):
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2] in pixels


class YOLOResponse(BaseModel):
    model: str
    total: int
    with_mask: int
    without_mask: int
    detections: List[YOLODetection]


# Helpers
def preprocess_mobilenet(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def run_mobilenet(image_bytes: bytes) -> MobileNetResponse:
    image_array = preprocess_mobilenet(image_bytes)
    prob_without_mask = float(mobilenet_model.predict(image_array, verbose=0)[0][0])
    prob_with_mask = 1.0 - prob_without_mask
    label = "Without Mask" if prob_without_mask >= 0.5 else "With Mask"
    confidence = prob_without_mask if label == "Without Mask" else prob_with_mask

    return MobileNetResponse(
        model="MobileNetV2",
        label=label,
        confidence=round(confidence * 100, 2),
        probability={
            "With Mask": round(prob_with_mask * 100, 2),
            "Without Mask": round(prob_without_mask * 100, 2),
        },
    )


def run_yolo(image_bytes: bytes) -> YOLOResponse:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = yolo_model.predict(source=img, conf=0.25, iou=0.45, verbose=False)

    detections = []
    with_mask = 0
    without_mask = 0

    class_map = {0: "With Mask", 1: "Without Mask"}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = class_map.get(cls_id, str(cls_id))
        conf = round(float(box.conf[0]) * 100, 2)
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0]]

        detections.append(
            YOLODetection(label=label, confidence=conf, box=[x1, y1, x2, y2])
        )

        if label == "With Mask":
            with_mask += 1
        else:
            without_mask += 1

    return YOLOResponse(
        model="YOLO26n",
        total=len(detections),
        with_mask=with_mask,
        without_mask=without_mask,
        detections=detections,
    )


# Routes
@app.get("/")
def root():
    return {"status": "ok", "message": "Face Mask Detection API is running."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": ["MobileNetV2", "YOLO26n"],
    }


@app.post("/predict/mobilenet", response_model=MobileNetResponse)
async def predict_mobilenet(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPEG / PNG images are supported.")
    return run_mobilenet(await file.read())


@app.post("/predict/yolo", response_model=YOLOResponse)
async def predict_yolo(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPEG / PNG images are supported.")
    return run_yolo(await file.read())
