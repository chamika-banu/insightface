# InsightFace Face Verification Service — Local Development Guide

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Docker Desktop | 4.x+ | Container runtime |
| Docker Compose | v2 (bundled with Docker Desktop) | Service orchestration |
| make | any | Shortcut commands |

No local Python environment is required. Everything runs inside Docker.

---

## Service Layout

This service is fully self-contained. All commands are run from inside the `insightface/` folder.

```
insightface/
├── main.py
├── face_service.py
├── schemas.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml                       # Runs this service only
├── Makefile                                 # Build and serve shortcuts
├── insightface.postman_collection.json      # Postman collection for this service
└── docs/
```

---

## First-Time Build & Model Download

When you build the service for the first time, or after clearing volumes, it will download the **InsightFace buffalo_l** model pack (~300MB).

###  **Build the image**:

```bash
make build
# or
docker compose build
```

###  **Initial Startup**:

The first time you run the service, the model weights are downloaded from the InsightFace hub.

```bash
make serve
# or:
docker compose up
```

The weights are cached in a Docker named volume called `insightface_cache` (mounted to `/root/.insightface` inside the container). This ensuring that subsequent containers start instantly without re-downloading the model.

## Running the Service

### Start the service

```bash
make serve
# or:
docker compose up
```

### Rebuild and start (after changing requirements or Dockerfile)

```bash
make serve-build
# or:
docker compose up --build
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

The recommended way to test endpoints locally is via the interactive Swagger documentation.

1.  Open `http://localhost:8002/docs` in your browser.
2.  Click **"Try it out"** on any endpoint.
3.  Upload images and click **"Execute"**.

> Swagger is only available when `ENABLE_SWAGGER=true` (the default in `docker-compose.yml`).

### Postman

Import `insightface/insightface.postman_collection.json` into Postman. The collection covers the endpoints with example responses.

---

## Reading the Logs

Logs are written to stdout and appear in the `docker compose up` terminal output. The service uses standard Uvicorn logging.

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
| `make build` | Build the Docker image |
| `make serve` | Start the service |
| `make serve-build` | Rebuild and start |
