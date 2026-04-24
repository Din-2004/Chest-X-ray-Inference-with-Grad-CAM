# Chest X-ray FastAPI Backend

A production-oriented FastAPI backend for chest X-ray inference with PyTorch model loading, optional preprocessing, Grad-CAM generation, checkpoint uploads, sample-image inference, and CSV/PDF export.

## Features

- FastAPI API for chest X-ray inference
- PyTorch checkpoint loading from upload or local default path
- CUDA auto-selection when available, CPU fallback otherwise
- Torchvision preprocessing with `Resize(448)` and `CenterCrop(448)`
- Optional preprocessing flags:
  - bone suppression stub
  - lung field cropping stub
- Lightweight custom Grad-CAM implementation
- In-memory backend session state:
  - `checkpoint_loaded`
  - `detected_classes`
  - `expected_input_size`
  - `metadata_enabled`
- Strict validation:
  - allowed checkpoint extensions: `.pt`, `.pth`
  - allowed uploaded image extensions: `.jpg`, `.jpeg`, `.png`
  - checkpoint size limit: `200 MB`
- Export of the latest prediction as CSV or PDF
- CORS enabled for local UI development

## Deliverables

- [main.py](main.py): FastAPI application and API routes
- [model.py](model.py): model service, preprocessing, inference, Grad-CAM, export helpers
- [requirements.txt](requirements.txt): backend + ML dependencies

## Python Version

- Python `3.10+`

## Setup with Virtualenv

### Linux / macOS / Git Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Backend

Development server with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Production-style run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open the interactive docs at:

```text
http://localhost:8000/docs
```

## Default Checkpoint Behavior

At startup the backend tries to load the default checkpoint path:

```text
checkpoints/best_resnet50.pt
```

If that file is missing or invalid, the API still starts and `/status` will report `checkpoint_loaded: false`.

## API Endpoints

### `POST /upload_checkpoint`

Multipart upload for a `.pt` or `.pth` checkpoint.

Response:

```json
{"status":"ok","checkpoint_path":"C:/.../uploads/checkpoints/model.pt"}
```

### `GET /status`

Response example:

```json
{
  "checkpoint_loaded": true,
  "detected_classes": 4,
  "expected_input_size": 448,
  "metadata_enabled": false
}
```

### `POST /predict`

Multipart request with:

- `image`: uploaded `jpg`, `jpeg`, or `png`
- `bone_suppression`: `true` or `false`
- `lung_crop`: `true` or `false`
- `age`: optional integer
- `gender`: optional string such as `unknown`, `male`, `female`

Response example:

```json
{
  "scores": {
    "Atelectasis": 0.41,
    "Cardiomegaly": 0.18,
    "Effusion": 0.77,
    "Infiltration": 0.33
  },
  "predicted": "Effusion",
  "predicted_labels": ["Effusion"],
  "gradcam": "data:image/png;base64,...",
  "input_size": 448
}
```

### `POST /predict_sample`

JSON request:

```json
{
  "sample_name": "dataset-chestxray14.webp",
  "options": {
    "bone_suppression": false,
    "lung_crop": false,
    "age": 58,
    "gender": "male",
    "gradcam_alpha": 0.45
  }
}
```

### `GET /export_results?format=csv|pdf`

Exports the latest inference result as a downloadable file attachment.

## Test with curl

### Upload a checkpoint

```bash
curl -X POST "http://localhost:8000/upload_checkpoint" \
  -F "file=@checkpoints/best_resnet50.pt"
```

### Check backend status

```bash
curl "http://localhost:8000/status"
```

### Predict from an uploaded image

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@sample_images/sample_000.png" \
  -F "bone_suppression=false" \
  -F "lung_crop=false" \
  -F "age=55" \
  -F "gender=unknown"
```

### Predict from a sample image

```bash
curl -X POST "http://localhost:8000/predict_sample" \
  -H "Content-Type: application/json" \
  -d "{\"sample_name\":\"dataset-chestxray14.webp\",\"options\":{\"bone_suppression\":false,\"lung_crop\":false,\"age\":60,\"gender\":\"female\",\"gradcam_alpha\":0.45}}"
```

### Export CSV

```bash
curl -L "http://localhost:8000/export_results?format=csv" -o chest_xray_results.csv
```

### Export PDF

```bash
curl -L "http://localhost:8000/export_results?format=pdf" -o chest_xray_results.pdf
```

## Notes

- Uploaded checkpoint files are stored under `uploads/checkpoints/`.
- Sample images are resolved from `sample_images/` and `assets/media/`.
- Metadata currently uses `age` and `gender`; the internal model wrapper pads the missing view feature with a default value for compatibility with metadata-enabled checkpoints.
- The bone suppression and lung crop steps are placeholders intended for later replacement with domain-specific preprocessing.
- The backend does not require Streamlit to run; it is designed to serve any local UI that can call the API.
