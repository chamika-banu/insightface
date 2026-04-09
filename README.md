# InsightFace Face Verification Service

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

A production-ready microservice for high-precision face detection and verification. This service leverages Microsoft's **Buffalo_L** model pack and **ArcFace** embeddings to compare live selfies against faces extracted from ID documents, optimized for KYC and identity verification workflows.

---

## 🚀 Key Features

*   **ArcFace Embeddings**: Uses the `w600k_r50` model for state-of-the-art face recognition accuracy.
*   **Intelligent Quality Gates**: Automatically rejects blurry images or faces that are too small.
*   **Format Versatility**: Native support for **HEIF/HEIC**, JPEG, and PNG formats.
*   **EXIF Intelligence**: Corrects image rotation automatically using metadata to ensure detection reliability.
*   **GPU Accelerated**: Optimized for NVIDIA CUDA with automatic failover to CPU.
*   **Containerized**: Fully orchestrated with Docker and Docker Compose for seamless deployment.

---

## 🛠 Tech Stack

*   **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
*   **Core Model**: [InsightFace Buffalo_L](https://github.com/deepinsight/insightface)
*   **Inference**: ONNX Runtime (CUDA/CPU)
*   **Processing**: OpenCV, Pillow-HEIF, NumPy
*   **Server**: Uvicorn

---

## ⚙️ Workflow

The service follows a rigorous 4-stage pipeline to ensure extraction reliability:

1.  **Image Acquisition**: Loads images while handling EXIF orientation and converting HEIC/HEIF formats to BGR.
2.  **Quality Gate**: 
    *   **Detection**: Runs `SCRFD` face detection to find all faces in the image.
    *   **Validation**: Rejects images if the face detection score is `< 0.80` or if the face size is `< 80x80` pixels.
    *   **Singularity**: Ensures only one face is present in the selfie upload.
3.  **Face Recognition**: Extracts high-dimensional (512-d) feature embeddings using ArcFace.
4.  **Similarity Analysis**:
    *   **Match**: Calculates Cosine Similarity; a score `≥ 0.42` constitutes a positive match.
    *   **Confidence**: Normalizes the similarity score into a human-readable confidence percentage.

---

## 📡 API Endpoints

### `POST /verify/face`
Main endpoint for face verification. Accepts a Form-data file upload.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `session_id` | string | Unique identifier for tracking the request. |
| `selfie_image` | file | Live selfie image (JPEG/PNG/HEIF). |
| `id_image` | file | ID document photo (JPEG/PNG/HEIF). |

**Example Response:**
```json
{
  "success": true,
  "data": {
    "is_match": true,
    "similarity_score": 0.823451,
    "confidence": 0.99,
    "failure_reason": null
  },
  "error": null
}
```

### `GET /health`
Returns service status, active model pack, and the current execution provider (e.g., `CUDAExecutionProvider`).

---

## 📦 Getting Started

### Prerequisites
*   Docker and Docker Compose
*   NVIDIA Container Toolkit (Optional, for GPU support)

### Deployment with Docker

```bash
# Build and start the service
docker-compose up --build
```
The service will be available at `http://localhost:8002`. Access the Interactive Swagger UI at `http://localhost:8002/docs`.

---

## 🔧 Configuration

Environment variables can be set in `docker-compose.yml`:

*   `ENABLE_SWAGGER`: Toggle Swagger UI (default: `true`).
*   `PYTHONUNBUFFERED`: Standard python logging mode (default: `1`).

---

## 👨💻 Development

*   **Testing**: Use the provided `insightface.postman_collection.json` for rapid API testing.
*   **Model Caching**: Models are stored in a Docker volume (`insightface_cache`) to avoid re-downloading (~330MB) on every boot.
*   **Makefile**:
    *   `make build`: Build the Docker image.
    *   `make run`: Start the container locally.
