FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies — cmake and build-essential are required to compile
# InsightFace's C extensions (e.g. the face detection post-processing kernels).
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ONNX Runtime (CPU-only build).
# onnxruntime is installed separately so each Dockerfile can pin the correct
# variant — CPU here, onnxruntime-gpu in Dockerfile.gpu.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "onnxruntime>=1.19.2,<2.0"

# Install remaining Python dependencies.
# onnxruntime is already satisfied above; pip will skip it in requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the buffalo_l model pack (~330 MB) to bake it into the image.
# This eliminates the runtime download delay and makes the service air-gap safe.
RUN python3 -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=0, det_size=(640, 640))"

# Copy service and application files
COPY face_service.py .
COPY main.py .
COPY schemas.py .

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
