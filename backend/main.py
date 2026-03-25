import io
import shutil
import tempfile
import time
from pathlib import Path
from threading import Event, Lock
from typing import Any, List
from uuid import uuid4

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# App setup
app = FastAPI(title="Face Mask Detector", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 224
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MOBILENET_MODEL_CANDIDATES = (
    MODEL_DIR / "mobilenetv2_mask.h5",
    MODEL_DIR / "mobilenetv2_mask.keras",
)
YOLO_MODEL_PATH = MODEL_DIR / "yolo_11_new_3.pt"
YOLO_CONFIDENCE = 0.25
YOLO_IOU = 0.45
VIDEO_STREAM_FALLBACK_FPS = 12
VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "application/octet-stream",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


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


class VideoStreamSessionResponse(BaseModel):
    stream_id: str
    stream_url: str


# Helpers
def preprocess_mobilenet(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def load_mobilenet_model() -> tuple[object, str]:
    for model_path in MOBILENET_MODEL_CANDIDATES:
        if model_path.exists():
            print(f"Using MobileNetV2 model file: {model_path.name}")
            model = tf.keras.models.load_model(  # pylint: disable=no-member
                str(model_path)
            )
            return model, model_path.name

    expected_files = ", ".join(path.name for path in MOBILENET_MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find a MobileNetV2 model in {MODEL_DIR}. "
        f"Expected one of: {expected_files}"
    )


def load_yolo_model() -> tuple[YOLO, str]:
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find the YOLO model file: {YOLO_MODEL_PATH.name}"
        )

    print(f"Using YOLO model file: {YOLO_MODEL_PATH.name}")
    return YOLO(str(YOLO_MODEL_PATH)), YOLO_MODEL_PATH.name


def cleanup_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


STREAM_SESSIONS = {}
STREAM_SESSIONS_LOCK = Lock()


def validate_image_upload(file: UploadFile) -> None:
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPEG / PNG images are supported.")


def validate_video_upload(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if file.content_type not in VIDEO_MIME_TYPES and suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(400, "Only MP4, MOV, AVI, and MKV videos are supported.")


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
    results = yolo_model.predict(
        source=img,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        verbose=False,
    )

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


def run_yolo_video(video_bytes: bytes, filename: str) -> tuple[Path, Path]:
    work_dir = Path(tempfile.mkdtemp(prefix="yolo_video_"))
    input_path = work_dir / filename
    input_path.write_bytes(video_bytes)
    return work_dir, input_path


def create_video_stream_session(video_bytes: bytes, filename: str) -> str:
    work_dir, input_path = run_yolo_video(video_bytes, filename)
    stream_id = uuid4().hex
    stop_event = Event()

    with STREAM_SESSIONS_LOCK:
        STREAM_SESSIONS[stream_id] = {
            "work_dir": work_dir,
            "input_path": input_path,
            "stop_event": stop_event,
            "active": False,
        }

    return stream_id


def get_video_stream_session(stream_id: str):
    with STREAM_SESSIONS_LOCK:
        return STREAM_SESSIONS.get(stream_id)


def cleanup_video_stream_session(stream_id: str) -> None:
    with STREAM_SESSIONS_LOCK:
        session = STREAM_SESSIONS.pop(stream_id, None)

    if session is not None:
        cleanup_path(session["work_dir"])


def generate_yolo_video_stream(
    stream_id: str,
    capture: Any,
    stop_event: Event,
    frame_delay: float,
):
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"

    try:
        while True:
            if stop_event.is_set():
                break

            loop_start = time.perf_counter()
            success, frame = capture.read()
            if not success:
                break

            results = yolo_model.predict(
                source=frame,
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                verbose=False,
            )
            annotated_frame = results[0].plot()
            encoded, buffer = cv2.imencode(  # pylint: disable=no-member
                ".jpg", annotated_frame
            )
            if not encoded:
                continue

            yield boundary + buffer.tobytes() + b"\r\n"

            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
    finally:
        capture.release()
        cleanup_video_stream_session(stream_id)


# Load models once at startup
print("Loading MobileNetV2...")
mobilenet_model, mobilenet_model_name = load_mobilenet_model()

print("Loading YOLO26...")
yolo_model, yolo_model_name = load_yolo_model()

print("All models ready.")


# Routes
@app.get("/")
def root():
    return {"status": "ok", "message": "Face Mask Detection API is running."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": ["MobileNetV2", "YOLO26n"],
        "model_files": {
            "mobilenet": mobilenet_model_name,
            "yolo": yolo_model_name,
        },
        "video_inference": True,
        "video_streaming": True,
    }


@app.post("/predict/mobilenet", response_model=MobileNetResponse)
async def predict_mobilenet(file: UploadFile = File(...)):
    validate_image_upload(file)
    return run_mobilenet(await file.read())


@app.post("/predict/yolo", response_model=YOLOResponse)
async def predict_yolo(file: UploadFile = File(...)):
    validate_image_upload(file)
    return run_yolo(await file.read())


@app.post("/stream/yolo-video", response_model=VideoStreamSessionResponse)
async def create_yolo_video_stream(file: UploadFile = File(...)):
    validate_video_upload(file)
    filename = Path(file.filename or "mask_detection.mp4").name
    stream_id = create_video_stream_session(await file.read(), filename)

    return VideoStreamSessionResponse(
        stream_id=stream_id,
        stream_url=f"/stream/yolo-video/{stream_id}",
    )


@app.get("/stream/yolo-video/{stream_id}")
async def stream_yolo_video(stream_id: str):
    session = get_video_stream_session(stream_id)
    if session is None:
        raise HTTPException(404, "Video stream session was not found.")
    if session["active"]:
        raise HTTPException(409, "Video stream session is already in use.")

    capture = cv2.VideoCapture(str(session["input_path"]))  # pylint: disable=no-member
    if not capture.isOpened():
        cleanup_video_stream_session(stream_id)
        raise HTTPException(400, "Could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member
    frame_delay = 1 / fps if fps and fps > 0 else 1 / VIDEO_STREAM_FALLBACK_FPS
    session["active"] = True

    return StreamingResponse(
        generate_yolo_video_stream(
            stream_id,
            capture,
            session["stop_event"],
            frame_delay,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/stream/yolo-video/{stream_id}/stop")
async def stop_yolo_video_stream(stream_id: str):
    session = get_video_stream_session(stream_id)
    if session is None:
        return {"status": "already_stopped"}

    session["stop_event"].set()
    if not session["active"]:
        cleanup_video_stream_session(stream_id)

    return {"status": "stopping"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
