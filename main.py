"""FastAPI backend for chest X-ray inference and Grad-CAM visualization."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from model import (
    ALLOWED_CHECKPOINT_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_GRADCAM_ALPHA,
    ChestXrayInferenceService,
    PredictionResult,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("chest_xray_backend")

PROJECT_ROOT = Path(__file__).resolve().parent
APP_TITLE = "Chest X-ray Inference Backend"
UPLOAD_DIR = PROJECT_ROOT / "uploads" / "checkpoints"
EXPORT_NAMES = {"csv": "chest_xray_results.csv", "pdf": "chest_xray_results.pdf"}
MAX_CHECKPOINT_BYTES = 200 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BODY_BYTES = 210 * 1024 * 1024
ALLOWED_GENDERS = {"unknown", "male", "female", "m", "f", "0", "1"}
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]


@dataclass
class BackendSessionState:
    """Small in-memory backend state."""

    service: ChestXrayInferenceService
    last_result: Optional[PredictionResult] = None


class PredictOptions(BaseModel):
    """Optional preprocessing and metadata controls for sample predictions."""

    bone_suppression: bool = False
    lung_crop: bool = False
    age: Optional[int] = Field(default=None, ge=0, le=120)
    gender: Optional[str] = "unknown"
    gradcam_alpha: float = Field(default=DEFAULT_GRADCAM_ALPHA, ge=0.1, le=0.9)


class PredictSampleRequest(BaseModel):
    """Request body for the sample prediction endpoint."""

    sample_name: str
    options: PredictOptions = Field(default_factory=PredictOptions)


service = ChestXrayInferenceService(default_checkpoint_path=DEFAULT_CHECKPOINT_PATH)
backend_state = BackendSessionState(service=service)

app = FastAPI(title=APP_TITLE, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_gender(value: Optional[str]) -> Optional[str]:
    """Validates and normalizes gender input."""

    if value is None or not value.strip():
        return "unknown"

    normalized = value.strip().lower()
    if normalized not in ALLOWED_GENDERS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported gender value '{value}'. Allowed values: {sorted(ALLOWED_GENDERS)}",
        )
    return normalized


def _validate_checkpoint_extension(filename: str) -> None:
    """Validates checkpoint filename extensions."""

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_CHECKPOINT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported checkpoint extension '{suffix}'. Allowed extensions: {sorted(ALLOWED_CHECKPOINT_EXTENSIONS)}",
        )


def _validate_image_extension(filename: str) -> None:
    """Validates uploaded image filename extensions."""

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension '{suffix}'. Allowed extensions: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )


async def _read_limited_file(upload: UploadFile, max_bytes: int) -> bytes:
    """Reads an uploaded file and enforces a byte limit."""

    data = await upload.read()
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file '{upload.filename}' exceeds the size limit of {max_bytes // (1024 * 1024)} MB.",
        )
    return data


@app.middleware("http")
async def request_logger_and_size_limit(request: Request, call_next):
    """Logs requests and rejects oversized bodies early when content-length is known."""

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request body exceeds {MAX_REQUEST_BODY_BYTES // (1024 * 1024)} MB."},
                )
        except ValueError:
            LOGGER.warning("Received invalid content-length header: %s", content_length)

    LOGGER.info("Request started: %s %s", request.method, request.url.path)
    response = await call_next(request)
    LOGGER.info("Request finished: %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.on_event("startup")
async def startup_event() -> None:
    """Initializes runtime directories and tries to load the default checkpoint."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    backend_state.service.try_load_default_checkpoint()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Returns clean JSON errors for unexpected failures."""

    LOGGER.exception("Unhandled backend error during %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.post("/upload_checkpoint")
async def upload_checkpoint(file: UploadFile = File(...)) -> dict:
    """Uploads a checkpoint file, saves it locally, and activates it."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No checkpoint filename was provided.")

    _validate_checkpoint_extension(file.filename)
    file_bytes = await _read_limited_file(file, max_bytes=MAX_CHECKPOINT_BYTES)

    save_path = UPLOAD_DIR / Path(file.filename).name
    save_path.write_bytes(file_bytes)
    LOGGER.info("Saved uploaded checkpoint to %s", save_path)

    try:
        backend_state.service.load_checkpoint(save_path)
        backend_state.last_result = None
    except Exception as exc:
        LOGGER.exception("Failed to load uploaded checkpoint from %s", save_path)
        raise HTTPException(status_code=400, detail=f"Failed to load checkpoint: {exc}") from exc

    return {"status": "ok", "checkpoint_path": str(save_path.resolve())}


@app.get("/status")
async def status() -> dict:
    """Returns the current backend status."""

    return backend_state.service.status()


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    bone_suppression: bool = Form(False),
    lung_crop: bool = Form(False),
    age: Optional[int] = Form(default=None),
    gender: Optional[str] = Form(default="unknown"),
) -> dict:
    """Runs inference on an uploaded image."""

    if not backend_state.service.checkpoint_loaded:
        raise HTTPException(status_code=400, detail="No checkpoint is loaded. Upload a checkpoint first.")
    if age is not None and not (0 <= age <= 120):
        raise HTTPException(status_code=422, detail="Age must be between 0 and 120.")
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image filename was provided.")

    _validate_image_extension(image.filename)
    normalized_gender = _normalize_gender(gender)
    image_bytes = await _read_limited_file(image, max_bytes=MAX_IMAGE_BYTES)

    try:
        result = backend_state.service.predict_from_bytes(
            image_bytes=image_bytes,
            source_name=image.filename,
            bone_flag=bone_suppression,
            crop_flag=lung_crop,
            age=age,
            gender=normalized_gender,
        )
    except Exception as exc:
        LOGGER.exception("Inference failed for uploaded image %s", image.filename)
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    backend_state.last_result = result
    return result.as_response()


@app.post("/predict_sample")
async def predict_sample(payload: PredictSampleRequest) -> dict:
    """Runs inference on a whitelisted sample image."""

    if not backend_state.service.checkpoint_loaded:
        raise HTTPException(status_code=400, detail="No checkpoint is loaded. Upload a checkpoint first.")

    normalized_gender = _normalize_gender(payload.options.gender)
    try:
        result = backend_state.service.predict_from_sample(
            sample_name=payload.sample_name,
            bone_flag=payload.options.bone_suppression,
            crop_flag=payload.options.lung_crop,
            age=payload.options.age,
            gender=normalized_gender,
            gradcam_alpha=payload.options.gradcam_alpha,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Inference failed for sample image %s", payload.sample_name)
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    backend_state.last_result = result
    return result.as_response()


@app.get("/export_results")
async def export_results(format: Literal["csv", "pdf"] = Query(...)) -> StreamingResponse:
    """Exports the last prediction result as CSV or PDF."""

    if backend_state.last_result is None:
        raise HTTPException(status_code=404, detail="No prediction result is available to export yet.")

    if format == "csv":
        payload = backend_state.service.export_csv_bytes(backend_state.last_result)
        media_type = "text/csv"
    else:
        payload = backend_state.service.export_pdf_bytes(backend_state.last_result)
        media_type = "application/pdf"

    headers = {"Content-Disposition": f"attachment; filename={EXPORT_NAMES[format]}"}
    return StreamingResponse(io.BytesIO(payload), media_type=media_type, headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
