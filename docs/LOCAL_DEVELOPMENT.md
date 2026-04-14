# InsightFace Face Verification Service — Local Development Guide

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Docker Desktop | 4.x+ | Container runtime |
| Docker Compose | v2 (bundled with Docker Desktop) | Service orchestration |
| make | any | Shortcut commands |
| NVIDIA Container Toolkit | latest | **GPU mode only** — not required for CPU |

No local Python environment is required. Everything runs inside Docker.

---

## Service Layout

This service is fully self-contained. All commands are run from inside the `insightface/` folder.

```
insightface/
├── main.py                                  # FastAPI app — routes, request handling, logging
├── face_service.py                          # InsightFaceService — model loading and inference
├── schemas.py                               # Pydantic request/response models
├── requirements.txt                         # Python dependencies (onnxruntime installed by Dockerfile)
├── Dockerfile                               # CPU image — python:3.11-slim, onnxruntime (CPU)
├── Dockerfile.gpu                           # GPU image — nvidia/cuda base, onnxruntime-gpu
├── docker-compose.yml                       # Base dev compose — CPU, builds Dockerfile
├── docker-compose.gpu.yml                   # GPU overlay — switches to Dockerfile.gpu + GPU devices
├── docker-compose.prod.yml                  # Prod override — registry image, CPU
├── docker-compose.prod-gpu.yml              # GPU prod override — adds GPU device reservation
├── Makefile                                 # Build and serve shortcuts (CPU + GPU targets)
├── insightface.postman_collection.json      # Postman collection for this service
└── docs/
```

---

## First-Time Build & Model Download

The service uses a **pre-baked** model strategy. The **InsightFace buffalo_l** model pack (~330 MB) is downloaded and baked into the Docker image at build time.

### CPU (default — works on any machine)

```bash
make build
# or:
docker compose build
```

### GPU (requires NVIDIA Container Toolkit on host)

```bash
make build-gpu
# or:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
```

> **Note:** The initial build downloads the buffalo_l model pack (~330 MB). Subsequent builds are fast due to layer caching. Both CPU and GPU images include the same model weights.

---

## Running the Service

### CPU (default)

```bash
make serve
# or:
docker compose up
```

### GPU

```bash
make serve-gpu
# or:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

### Rebuild and start (after changing requirements or Dockerfile)

```bash
# CPU
make serve-build

# GPU
make serve-build-gpu
```

The service is ready for inference almost immediately upon startup — there is no runtime model download.

---

## Verifying the Active Device

Check the `/health` endpoint after startup to confirm which execution provider is active:

```bash
curl http://localhost:8002/health
```

**CPU response:**
```json
{"status": "ok", "model": "InsightFace buffalo_l (ArcFace w600k_r50)", "device": "CPUExecutionProvider"}
```

**GPU response:**
```json
{"status": "ok", "model": "InsightFace buffalo_l (ArcFace w600k_r50)", "device": "CUDAExecutionProvider"}
```

---

## Environment Variables

Set in `insightface/docker-compose.yml`:

| Variable | Dev value | Prod value | Description |
|---|---|---|---|
| `ENABLE_SWAGGER` | `true` | `false` | Toggles the `/docs` interactive documentation. |

---

## Testing Endpoints

### Swagger UI

Open `http://localhost:8002/docs` in your browser. Click **"Try it out"** on any endpoint.

> Swagger is only available when `ENABLE_SWAGGER=true` (the default in `docker-compose.yml`).

### Postman

Import `insightface/insightface.postman_collection.json` into Postman.

---

## Reading the Logs

Logs are written to stdout and appear in the `docker compose up` terminal output.

### Diagnosing failures

| Failure code | Meaning |
|---|---|
| `face_not_detected_selfie` | No face found in selfie. |
| `face_not_detected_id` | No face in ID image — photo may be too small or obscured. |
| `multiple_faces_in_selfie` | More than one face detected in the selfie. |
| `face_too_small_selfie` | Face detected but bbox < 80×80 px — take a closer selfie. |
| `face_mismatch` | Faces detected in both images but similarity < 0.42. |
| `SYSTEM ERROR` | Unhandled exception — check logs for traceback. |

---

## Makefile Reference

Run from inside `insightface/`:

| Target | Description |
|---|---|
| `make build` | Build the CPU Docker image from `Dockerfile` |
| `make build-gpu` | Build the GPU Docker image from `Dockerfile.gpu` |
| `make serve` | Start the service (CPU) |
| `make serve-gpu` | Start the service (GPU) |
| `make serve-build` | Rebuild and start (CPU) |
| `make serve-build-gpu` | Rebuild and start (GPU) |
| `make serve-prod` | Start production mode (CPU) |
| `make serve-prod-gpu` | Start production mode (GPU) |
