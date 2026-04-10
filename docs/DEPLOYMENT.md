# InsightFace Face Verification Service — Deployment Guide

## Overview

This guide covers deploying the InsightFace service to a production environment. It is written in cloud-agnostic terms since the target platform has not yet been decided. The core concepts apply to any container-capable hosting environment.

---

## What Gets Deployed

The InsightFace service is a single Docker container:

- **Image:** built from `insightface/Dockerfile`
- **Base:** `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (GPU support)
- **Port:** `8002` (internal), mapped to whatever port the load balancer or gateway exposes
- **Process:** `python3 -m uvicorn main:app --host 0.0.0.0 --port 8002` (no `--reload`)
- **Approximate image size:** ~2.8 GB (CUDA + ONNX Runtime + pre-baked buffalo_l model pack)

---

## Production vs. Development Differences

| Aspect | Development | Production |
|---|---|---|
| `ENABLE_SWAGGER` | `true` | `false` |
| `--reload` flag | enabled | **removed** |
| Swagger UI (`/docs`) | enabled | **disabled** |
| Model cache | **N/A (Pre-baked)** | **N/A (Pre-baked)** |
| Logging | stdout (Docker) | stdout → log aggregator |

---

## Environment Variables

Set these in your hosting platform's environment/secrets configuration:

| Variable | Default | Notes |
|---|---|---|
| `ENABLE_SWAGGER` | `true` | Set to `false` to disable Swagger UI (`/docs`) |

---

The `buffalo_l` model pack (~330 MB) is **pre-baked** into the Docker image during the build process.

**Benefits:**
- **Zero-delay startup**: The service is ready as soon as the container starts.
- **Air-gap compatible**: No internet connection is required at runtime to download weights.
- **Immutable**: The exact model weights are version-locked within the image.

---

## Dockerfile — Production CMD

The current `CMD` in the Dockerfile:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

InsightFace with ONNX Runtime on CPU is significantly lighter than Florence-2. Multiple workers are feasible:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
```

Each worker loads its own copy of the buffalo_l model pack into memory (~500 MB). Tune based on available RAM.

---

## Health Check

Configure your hosting platform to poll:

```
GET /health  →  HTTP 200
```

Expected response (on GPU-enabled host):
```json
{"status": "ok", "model": "InsightFace buffalo_l (ArcFace w600k_r50)", "device": "CUDAExecutionProvider"}
```

**Startup grace period:** The buffalo_l model loads in a few seconds. A 15–30 second initial delay is sufficient.

**Recommended health check settings:**

| Setting | Value |
|---|---|
| Path | `/health` |
| Protocol | HTTP |
| Port | 8002 |
| Initial delay | 30s |
| Interval | 30s |
| Timeout | 10s |
| Unhealthy threshold | 3 consecutive failures |

---

## Networking

The InsightFace service should **not** be publicly accessible. It should only be reachable by the NestJS backend within the same private network.

**Access pattern:**
- NestJS → InsightFace: internal network only, after Step 1 (Florence-2) has passed
- Frontend → NestJS: public HTTPS
- Frontend → InsightFace: **never — blocked by network rules**

**Internal URL from NestJS:**

| Environment | URL |
|---|---|
| Same Docker Compose network | `http://insightface:8002` |
| Same private cloud network / VPC | `http://<internal-ip-or-dns>:8002` |
| Kubernetes (same namespace) | `http://insightface-service:8002` |

---

## Resource Requirements

These are approximate figures for CPU-only inference with InsightFace buffalo_l (ONNX Runtime):

| Resource | Minimum | Recommended | Notes |
|---|---|---|---|
| GPU | Optional | NVIDIA T4 / L4 / A10G | Required for hardware acceleration |
| vCPU | 1 | 2 | Higher if running multiple workers |
| RAM | 4 GB | 8 GB | Higher for multiple workers / large images |
| Disk (image) | 3 GB | 5 GB | Includes base OS, CUDA libraries, and weights |
| Network | inbound | — | Internal API only; baked models eliminate hub download |

**Inference time per request (CPU):**
- RetinaFace detection (both images): 1–3 seconds
- ArcFace embedding (both faces): 1–2 seconds
- Total per `/verify/face` call: 2–5 seconds

This is substantially faster than the Florence-2 service. NestJS HTTP timeout of 30 seconds is sufficient.

---

## Scaling

InsightFace is stateless. Each request is fully independent — no sessions, no stored embeddings.

- Run multiple container instances behind a load balancer
- Each instance holds its own copy of the model in memory (~500 MB per instance)
- No session affinity required
- No shared state between instances

**Note:** The two-image upload in `/verify/face` means the same container processes both the selfie and the ID image within a single request. NestJS is responsible for fetching the ID image from Redis before calling this service — the service itself receives both images in the same multipart form.

---

## Logging

The service writes structured logs to stdout. Route stdout to your platform's log aggregator.

Log format:
```
YYYY-MM-DDTHH:MM:SS [LEVEL] insightface.verify — [verify/face] EVENT  key=value ...
```

**Key fields to index in your log aggregator:**

| Field | Purpose |
|---|---|
| `session_id` | Correlate with NestJS and Florence-2 logs for a single verification flow |
| `failure_reason` | Track rejection rates by type |
| `similarity` | Monitor score distribution over time — drift may indicate model or image quality issues |
| `confidence` | Reflects certainty of the match/non-match decision |
| Log level `WARNING` | All verification rejections |
| Log level `ERROR` | System failures — trigger alerts |

**Recommended monitoring queries:**

- `failure_reason = face_mismatch` rate: should reflect genuine mismatches
- `failure_reason = face_not_detected_*` rate: high rate suggests image quality issues upstream
- `similarity` score distribution: watch for unexpected shifts after NestJS or frontend changes that affect how images are captured or cropped

---

## GPU Prerequisites

To use hardware acceleration in production:

1.  **Host Drivers**: The host machine must have NVIDIA drivers installed.
2.  **NVIDIA Container Toolkit**: The host must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed to expose GPUs to Docker.
3.  **Docker Compose**: The service must be started with the `deploy.resources.reservations.devices` section (included in `docker-compose.prod.yml`).

If these prerequisites are not met, the service will dynamically detect their absence and fallback to **CPU-only** mode automatically.

---

## Security Considerations

- **No image storage:** Both uploaded images are written to temp files and deleted immediately in a `finally` block — images are never persisted beyond the duration of a single request.
- **No authentication on the service itself:** Authentication and session validation are handled by NestJS. The InsightFace port should never be exposed to the public internet.
- **Embedding privacy:** ArcFace 512-d embeddings are computed in memory and discarded after each request. They are never logged or stored.
- **Model weights:** Downloaded from the InsightFace model hub. Pin the model pack version (`buffalo_l`) in `face_service.py` and verify checksums if supply chain integrity is a requirement.
- **Image format validation:** Consider adding file type validation (check magic bytes) in `main.py` before writing to temp if stricter input control is needed.

---

## Similarity Threshold Calibration

The current threshold is `SIMILARITY_THRESHOLD = 0.42` in `face_service.py`. This value should be calibrated against a real dataset before production:

1. Collect a labelled dataset of matching and non-matching selfie/ID pairs
2. Run all pairs through `/verify/face` and record `similarity_score`
3. Plot an ROC curve across threshold values
4. Select the threshold that gives an acceptable trade-off between false accept rate (FAR) and false reject rate (FRR) for the intended use case

A threshold that is too low increases the risk of false accepts. A threshold that is too high increases user friction from false rejections.

---

## Rolling Deploys

The buffalo_l model loads quickly (a few seconds), making rolling deploys straightforward:

1. Start the new container
2. Wait for `/health` to return 200
3. Route traffic to the new instance
4. Drain and terminate the old instance

Any platform that supports health-check-gated traffic shifting will work correctly.

---

## Docker Compose

The service ships with two Compose files:

| File | Purpose |
|---|---|
| `docker-compose.yml` | Development — builds locally, `--reload`, `ENABLE_SWAGGER=true` |
| `docker-compose.prod.yml` | Production override — registry image, no reload, `ENABLE_SWAGGER=false` |

`docker-compose.prod.yml` is a **Compose override file** — it merges with the base file, only overriding the specified keys. Run them together:

```bash
# Using make:
make serve-prod

# Or directly:
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

Before running in production, set the registry image:

```bash
export INSIGHTFACE_IMAGE=your-registry/insightface:latest
make serve-prod
```

If `INSIGHTFACE_IMAGE` is not set, it falls back to `insightface:latest` (a locally built image).

**What the prod override changes vs. dev:**

| Setting | Dev (`docker-compose.yml`) | Prod (`docker-compose.prod.yml`) |
|---|---|---|
| Image source | `build: .` (local) | `image: $INSIGHTFACE_IMAGE` (registry) |
| `ENABLE_SWAGGER` | `true` | `false` |
| uvicorn command | `--reload` | No reload, `--workers 1` |

> If deploying to a managed container platform (ECS, GCP Cloud Run, Fly.io, etc.), the platform typically has its own service definition format. In that case, use the Dockerfile directly and configure env vars through the platform — `docker-compose.prod.yml` is optional in those environments.

---

- [ ] `ENABLE_SWAGGER=false` is set in the production environment
- [ ] Swagger UI at `/docs` returns `404`
- [ ] `/health` returns `200` after startup
- [ ] InsightFace port (`8002`) is not reachable from the public internet
- [x] Model weights are pre-baked into the Docker image
- [ ] NestJS HTTP timeout for calls to this service is ≥ 30 seconds
- [ ] Health check initial delay is ≥ 30 seconds
- [ ] Logs are flowing to your log aggregator
- [ ] Alerts are configured for `ERROR` level log events
- [ ] Similarity threshold has been calibrated against a real dataset before enabling production traffic
