# InsightFace Face Verification Service — Implementation Guide

## Overview

The InsightFace service is a stateless FastAPI microservice that accepts a selfie image and an ID document image, detects the face in each, and returns whether the two faces belong to the same person. It is always called by the NestJS backend after Step 1 (ID verification) has passed — the frontend never contacts it directly.

- **Model:** InsightFace `buffalo_l` (RetinaFace detection + ArcFace `w600k_r50` recognition)
- **Runtime:** Dynamic (GPU/CPU) via ONNX Runtime
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

The service is stateless. It does not store images, sessions, or embeddings. The ID image is held by NestJS in Redis and forwarded here on each request.

---

## File Structure

```
insightface/
├── main.py                                  # FastAPI app — routes, request handling, logging
├── face_service.py                          # InsightFaceService — model loading and inference
├── schemas.py                               # Pydantic request/response models
├── requirements.txt                         # Python dependencies
├── Dockerfile                                # Container definition
├── docker-compose.yml                       # Dev — build locally, --reload, ENABLE_SWAGGER=true
├── docker-compose.prod.yml                  # Prod override — registry image, no reload, ENABLE_SWAGGER=false
├── Makefile                                  # Build and serve shortcuts
├── insightface.postman_collection.json      # Postman collection for this service
└── docs/
    ├── IMPLEMENTATION.md                    
    ├── LOCAL_DEVELOPMENT.md
    └── DEPLOYMENT.md
```


---

## Dependencies

```
insightface==0.7.3
onnxruntime-gpu>=1.19.2,<2.0
opencv-python-headless>=4.9.0
numpy>=1.26.4,<2.0
Pillow>=10.3.0
pillow-heif>=0.16.0
scipy>=1.12.0,<1.13
fastapi>=0.111.0
uvicorn>=0.31.1
python-multipart>=0.0.9
```

> `cmake` and `build-essential` are also required as system packages (installed in the Dockerfile) for compiling InsightFace's C extensions.

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
# 1. CUDA (if available)
# 2. CPU (fallback)
self.app = FaceAnalysis(
    name="buffalo_l",
    providers=providers,
)
self.app.prepare(ctx_id=0, det_size=(640, 640))
```

The `buffalo_l` pack is downloaded from the InsightFace model hub on first run and cached to `/root/.insightface` (mounted as a named Docker volume so it survives rebuilds).

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

If multiple faces are detected, the highest-confidence face is selected before the multiple-faces check. The multiple-faces check is applied after, and only on selfies (not on the ID image).

### Step 2 — ID Image Face Detection

Same logic as Step 1 with `label="id"`, except the single-face rule is not enforced. Only the highest-confidence face is used.

### Step 3 — Embedding Comparison

ArcFace produces a 512-dimensional embedding vector for each detected face. Cosine similarity is computed:

```python
similarity = dot(emb1 / ||emb1||, emb2 / ||emb2||)
```

Both vectors are L2-normalised before the dot product because `insightface 0.7.3` does not guarantee normalised output on `face.embedding`.

**Threshold:** `0.42`. Similarity ≥ 0.42 → `is_match = True`.

**Confidence score:** `min(1.0, 0.5 + abs(similarity - 0.42) * 2)`. Reflects how far the similarity is from the decision boundary — high confidence means a clear match or clear non-match.

---

## Endpoints

### GET /health

Returns service readiness. Docker and NestJS both use this to gate traffic.

**Response — 200 OK**
```json
{
  "status": "ok",
  "model": "InsightFace buffalo_l (ArcFace w600k_r50)",
  "device": "CUDAExecutionProvider"
}
```

---

### POST /verify/face

**Request — multipart/form-data**

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | string | ✅ | UUID from NestJS — passed through for correlation |
| `selfie_image` | file | ✅ | JPEG / PNG / WEBP selfie of the guardian |
| `id_image` | file | ✅ | JPEG / PNG / WEBP ID document image (forwarded by NestJS from Redis) |

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

Every endpoint returns the same three-field envelope. NestJS always reads `success` first.

| State | `success` | `data` | `error` | HTTP |
|---|---|---|---|---|
| Model ran, check passed | `true` | populated | `null` | 200 |
| Model ran, check failed | `false` | populated | populated | 200 |
| System error / crash | `false` | `null` | populated | 500 |

---

## Logging

The service uses standard Uvicorn logging for request tracing. Custom internal logs are kept to a minimum to ensure high signal-to-noise.

| Event | Level | Description |
|---|---|---|
| Request Trace | INFO | Standard Uvicorn request log (Method, Path, Status) |
| `SYSTEM ERROR` | ERROR | High-level exception details for inference failures |

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

These should be tuned based on real-world data before production deployment.

---

## Swagger UI

Available at `http://localhost:8002/docs` when `ENABLE_SWAGGER=true`.

ReDoc and the raw OpenAPI JSON schema (`/openapi.json`) are permanently disabled.
