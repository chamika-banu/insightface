# InsightFace Face Verification Service — Implementation Guide

## Overview

The InsightFace service is a stateless FastAPI microservice that accepts a selfie image and an ID document image, detects the face in each, and returns whether the two faces belong to the same person. It is always called by the NestJS backend after Step 1 (ID verification) has passed — the frontend never contacts it directly.

- **Model:** InsightFace `buffalo_l` (RetinaFace detection + ArcFace `w600k_r50` recognition)
- **Runtime:** Dynamic (GPU/CPU) via ONNX Runtime — auto-detected at startup
- **Port:** `8002`
- **Framework:** FastAPI + uvicorn

---

## Architecture Position

```
NestJS Backend
    │
    │    (After Step 1 passes — id_image retrieved from Redis)
    │
    ├──► POST http://insightface:8002/verify/face
    │         multipart/form-data: session_id, selfie_image, id_image
    │
    │◄── FaceVerificationResponse
    │         { success, data, error }
    │
    │    If success → mark guardian verified in PostgreSQL
    │    If failure → return rejection to frontend, stop
```

The service is stateless. It does not store images, sessions, or embeddings.

---

## File Structure

```
insightface/
├── main.py                                  # FastAPI app — routes, request handling, logging
├── face_service.py                          # InsightFaceService — model loading and inference
├── schemas.py                               # Pydantic request/response models
├── requirements.txt                         # Python dependencies (onnxruntime installed by Dockerfile)
├── Dockerfile                               # CPU image — python:3.11-slim, onnxruntime (CPU)
├── Dockerfile.gpu                           # GPU image — nvidia/cuda base, onnxruntime-gpu
├── docker-compose.yml                       # Base dev compose — CPU-safe (no GPU device block)
├── docker-compose.gpu.yml                   # GPU overlay — adds Dockerfile.gpu + GPU device access
├── docker-compose.prod.yml                  # Prod override — registry image, CPU
├── docker-compose.prod-gpu.yml              # GPU prod override — adds GPU device reservation
├── Makefile                                 # Build and serve shortcuts (CPU + GPU targets)
├── insightface.postman_collection.json      # Postman collection for this service
└── docs/
    ├── IMPLEMENTATION.md
    ├── LOCAL_DEVELOPMENT.md
    └── DEPLOYMENT.md
```

---

## Dependencies

```
# InsightFace & Inference
insightface==0.7.3
# onnxruntime is installed by each Dockerfile:
#   Dockerfile     → onnxruntime      (CPU-only)
#   Dockerfile.gpu → onnxruntime-gpu  (CUDAExecutionProvider + CPU fallback)

# Scientific stack
# numpy must stay <2.0.0 due to compatibility with ONNX Runtime and InsightFace 0.7.3
numpy>=1.26.4,<2.0
scipy>=1.12.0,<1.13

# Image processing
opencv-python-headless>=4.9.0
Pillow>=10.3.0
pillow-heif>=0.16.0

# API layer
fastapi>=0.111.0
uvicorn>=0.31.1
python-multipart>=0.0.9
```

> `cmake` and `build-essential` are also required as system packages (installed in both Dockerfiles) for compiling InsightFace's C extensions.

---

## CPU vs GPU Dockerfiles

The service ships with two Dockerfiles that are identical in application code but differ in their base image and ONNX Runtime variant:

| | `Dockerfile` (CPU) | `Dockerfile.gpu` (GPU) |
|---|---|---|
| Base image | `python:3.11-slim` | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| ONNX Runtime | `onnxruntime` (CPU-only build) | `onnxruntime-gpu` (CUDA + CPU providers) |
| `docker-compose` | `docker-compose.yml` | `docker-compose.yml` + `docker-compose.gpu.yml` |
| `make` target | `make build` / `make serve` | `make build-gpu` / `make serve-gpu` |

`onnxruntime` is installed **before** `requirements.txt` in each Dockerfile. This means pip sees it as already satisfied and skips it when processing `requirements.txt`.

---

## Versioning & Compatibility

InsightFace versioning is critical for production stability, particularly regarding **embedding stability**.

### 1. Library Versioning
The library version is pinned in `requirements.txt`:
```text
insightface==0.7.3
```

### 2. Model Pack Versioning
The model pack is specified in `face_service.py`:
```python
self.app = FaceAnalysis(name="buffalo_l", providers=providers)
```
**To change packs**: Update the `name` parameter. Popular packs include:
- `buffalo_l`: Large (Current), best accuracy.
- `buffalo_s`: Small, faster inference, lower accuracy.
- `antelopev2`: High-performance alternative.

### 3. Critical: Embedding Compatibility
> **Changing the library version or the model pack will likely change the resulting 512-d embeddings.**

If you have stored face embeddings in a database (e.g., for "known users"):
- **Existing embeddings will become invalid** if the model version changes.
- A match score of `0.70` on version `0.7.3` might become `0.20` on a newer version.
- **Migration Plan**: If you upgrade versions, you must re-process all source images to generate new embeddings.

---

## Model Loading

The model pack is loaded **once at container startup** using FastAPI's `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = InsightFaceService()
    yield
```

`InsightFaceService.__init__()` detects available ONNX Execution Providers and prioritizes them:

```python
available_providers = ort.get_available_providers()
# 1. CUDA (if available — only when using onnxruntime-gpu image on a GPU host)
# 2. CPU (always available as fallback)
providers = []
if "CUDAExecutionProvider" in available_providers:
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")

self.app = FaceAnalysis(
    name="buffalo_l",
    providers=providers,
)
self.app.prepare(ctx_id=0, det_size=(640, 640))
```

The `buffalo_l` pack is **pre-baked** into both Docker images during build to ensure zero-delay startup.

**What `buffalo_l` contains:**

| Model | Task | Architecture |
|---|---|---|
| `det_10g` | Face detection | RetinaFace |
| `w600k_r50` | Face recognition / embedding | ArcFace ResNet-50 |

---

## Inference Pipeline

`verify_faces()` in `face_service.py` runs three sequential steps:

### Step 1 — Selfie Face Detection

`get_primary_face(selfie_path, label="selfie")` runs RetinaFace on the selfie and applies three quality gates:

| Gate | Condition | Failure code |
|---|---|---|
| No face | `len(faces) == 0` | `face_not_detected_selfie` |
| Multiple faces | `len(faces) > 1` (selfie only) | `multiple_faces_in_selfie` |
| Low confidence | `det_score < 0.80` | `face_not_detected_selfie` |
| Face too small | `bbox_width < 80` or `bbox_height < 80` | `face_too_small_selfie` |

### Step 2 — ID Image Face Detection

Same logic as Step 1 with `label="id"`, except the single-face rule is not enforced.

### Step 3 — Embedding Comparison

ArcFace produces a 512-dimensional embedding vector for each detected face. Cosine similarity is computed:

```python
similarity = dot(emb1 / ||emb1||, emb2 / ||emb2||)
```

Both vectors are L2-normalised before the dot product because `insightface 0.7.3` does not guarantee normalised output on `face.embedding`.

**Threshold:** `0.42`. Similarity ≥ 0.42 → `is_match = True`.

**Confidence score:** `min(1.0, 0.5 + abs(similarity - 0.42) * 2)`.

---

## Endpoints

### GET /health

Returns service readiness and the active ONNX execution provider.

**Response — 200 OK (CPU)**
```json
{"status": "ok", "model": "InsightFace buffalo_l (ArcFace w600k_r50)", "device": "CPUExecutionProvider"}
```

**Response — 200 OK (GPU)**
```json
{"status": "ok", "model": "InsightFace buffalo_l (ArcFace w600k_r50)", "device": "CUDAExecutionProvider"}
```

---

### POST /verify/face

**Request — multipart/form-data**

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | ✅ | UUID from NestJS |
| `selfie_image` | file | ✅ | JPEG / PNG / WEBP selfie |
| `id_image` | file | ✅ | JPEG / PNG / WEBP ID document image |

**Response — Check Passed (HTTP 200)**
```json
{
  "success": true,
  "data": {
    "is_match": true,
    "similarity_score": 0.73,
    "confidence": 0.96,
    "failure_reason": null
  },
  "error": null
}
```

---

## Response Envelope

| State | `success` | `data` | `error` | HTTP |
|---|---|---|---|---|
| Model ran, check passed | `true` | populated | `null` | 200 |
| Model ran, check failed | `false` | populated | populated | 200 |
| System error / crash | `false` | `null` | populated | 500 |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ENABLE_SWAGGER` | `true` | Set to `false` to disable Swagger UI (`/docs`) in production |

---

## Pydantic Schemas (`schemas.py`)

```python
class FaceVerificationData(BaseModel):
    is_match: bool
    similarity_score: float
    confidence: float
    failure_reason: str | None

class ErrorDetail(BaseModel):
    code: str
    message: str
    stage: str

class FaceVerificationResponse(BaseModel):
    success: bool
    data: FaceVerificationData | None
    error: ErrorDetail | None
```

---

## Thresholds and Constants

All constants are defined at the top of `face_service.py`:

```python
SIMILARITY_THRESHOLD = 0.42    # Cosine similarity — calibrate against real data
MIN_FACE_SIZE = 80             # Minimum bounding box dimension in pixels
MIN_DETECTION_CONFIDENCE = 0.80
```

---

## Swagger UI

Available at `http://localhost:8002/docs` when `ENABLE_SWAGGER=true`.

ReDoc and the raw OpenAPI JSON schema (`/openapi.json`) are permanently disabled.
