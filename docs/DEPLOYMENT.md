# InsightFace Face Verification Service — Deployment Guide

## Overview

This guide covers deploying the InsightFace service to a production environment. It is written in cloud-agnostic terms since the target platform has not yet been decided. The core concepts apply to any container-capable hosting environment.

---

## What Gets Deployed

The InsightFace service is a single Docker container:

- **Image:** built from `insightface/Dockerfile` (CPU) or `insightface/Dockerfile.gpu` (GPU)
- **Port:** `8002` (internal), mapped to whatever port the load balancer or gateway exposes
- **Process:** `python3 -m uvicorn main:app --host 0.0.0.0 --port 8002` (no `--reload`)
- **Approximate image size:**
  - CPU image: ~1.5 GB
  - GPU image: ~2.8 GB (CUDA + cuDNN + onnxruntime-gpu + pre-baked buffalo_l model pack)

---

## CPU vs GPU Deployment

The service ships with two Dockerfiles that share the same application code and model weights.

| | `Dockerfile` (CPU) | `Dockerfile.gpu` (GPU) |
|---|---|---|
| Base image | `python:3.11-slim` | `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` |
| ONNX Runtime | `onnxruntime` | `onnxruntime-gpu` |
| Execution provider | `CPUExecutionProvider` | `CUDAExecutionProvider` + CPU fallback |
| Host requirements | None | NVIDIA drivers + NVIDIA Container Toolkit |
| Inference time | 2–5 seconds | < 1 second |

`face_service.py` auto-detects available ONNX providers at startup and selects the best one — no code change is needed between CPU and GPU deployments.

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

## Health Check

Configure your hosting platform to poll:

```
GET /health  →  HTTP 200
```

Expected response (CPU host):
```json
{"status": "ok", "model": "InsightFace buffalo_l (ArcFace w600k_r50)", "device": "CPUExecutionProvider"}
```

Expected response (GPU host):
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

| Resource | Minimum (CPU) | Recommended (CPU) | GPU |
|---|---|---|---|
| GPU | — | — | NVIDIA T4 / L4 / A10G |
| vCPU | 1 | 2 | 2 |
| RAM | 2 GB | 4 GB | 4 GB |
| Disk (image) | 2 GB | 3 GB (CPU) | 5 GB (GPU) |
| Network | inbound | — | — |

**Inference time per request:**
- CPU: 2–5 seconds
- GPU: < 1 second

NestJS HTTP timeout of 30 seconds is sufficient for either mode.

---

## Scaling

InsightFace is stateless. Each request is fully independent — no sessions, no stored embeddings.

- Run multiple container instances behind a load balancer
- Each instance holds its own copy of the model in memory (~500 MB per instance)
- No session affinity required

---

## GPU Prerequisites

To use GPU hardware acceleration in production:

1. **Host Drivers**: The host machine must have NVIDIA drivers installed.
2. **NVIDIA Container Toolkit**: Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to expose GPUs to Docker containers.
3. **GPU compose overlay**: Use `docker-compose.gpu.yml` (dev) or `docker-compose.prod-gpu.yml` (prod) — see the Docker Compose section below.
4. **GPU image**: Build from `Dockerfile.gpu` (done automatically by the GPU Makefile targets).

If these prerequisites are not met and the GPU image is accidentally used on a CPU host, `onnxruntime-gpu` will not find a CUDA execution provider and will silently fall back to `CPUExecutionProvider`. The service will still work correctly.

---

## Security Considerations

- **No image storage:** Both uploaded images are written to temp files and deleted immediately in a `finally` block.
- **No authentication on the service itself:** Authentication is handled by NestJS. The InsightFace port should never be exposed to the public internet.
- **Embedding privacy:** ArcFace 512-d embeddings are computed in memory and discarded after each request.

---

## Rolling Deploys

The buffalo_l model loads quickly (a few seconds), making rolling deploys straightforward:

1. Start the new container
2. Wait for `/health` to return 200
3. Route traffic to the new instance
4. Drain and terminate the old instance

---

## Docker Compose

The service ships with four Compose files:

| File | Purpose |
|---|---|
| `docker-compose.yml` | Base / Dev — CPU, builds locally, `--reload`, `ENABLE_SWAGGER=true` |
| `docker-compose.gpu.yml` | GPU overlay — switches build to `Dockerfile.gpu`, adds GPU device access |
| `docker-compose.prod.yml` | Production override — registry image, no reload, `ENABLE_SWAGGER=false` |
| `docker-compose.prod-gpu.yml` | GPU + Production overlay — adds GPU device access for GPU prod hosts |

### CPU (default)

```bash
# Dev
make serve
# or:
docker compose up

# Prod
export INSIGHTFACE_IMAGE=your-registry/insightface:latest
make serve-prod
# or:
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

### GPU

```bash
# Dev (GPU)
make serve-gpu
# or:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up

# Prod (GPU)
export INSIGHTFACE_IMAGE=your-registry/insightface:gpu-latest
make serve-prod-gpu
# or:
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.prod-gpu.yml up
```

> If deploying to a managed container platform (ECS, GCP Cloud Run, Fly.io, etc.), the platform typically has its own service definition format. In that case, use the Dockerfile directly and configure env vars through the platform — the compose files are optional in those environments.

---

## Makefile Reference

| Target | Description |
|---|---|
| `make build` | Build the CPU image from `Dockerfile` |
| `make build-gpu` | Build the GPU image from `Dockerfile.gpu` |
| `make serve` | Start the service (CPU) |
| `make serve-gpu` | Start the service (GPU) |
| `make serve-build` | Rebuild and start (CPU) |
| `make serve-build-gpu` | Rebuild and start (GPU) |
| `make serve-prod` | Start production (CPU) |
| `make serve-prod-gpu` | Start production (GPU) |
| `make build-prod` | Build production CPU image |
| `make build-prod-gpu` | Build production GPU image |
