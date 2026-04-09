from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from face_service import InsightFaceService
from schemas import (
    ErrorDetail,
    FaceVerificationData,
    FaceVerificationResponse,
    HealthResponse,
)

logger = logging.getLogger("uvicorn.error")

service: InsightFaceService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the buffalo_l model pack once at container startup."""
    global service
    service = InsightFaceService()
    yield

SWAGGER_ENABLED = os.getenv("ENABLE_SWAGGER", "true").lower() == "true"

app = FastAPI(
    lifespan=lifespan,
    title="InsightFace Face Verification Service",
    description=(
        "Detects and matches faces using InsightFace ArcFace (buffalo_l / w600k_r50). "
        "Compares a selfie against a face extracted from an ID document.\n"        
    ),
    version="1.0.0",
    docs_url="/docs" if SWAGGER_ENABLED else None,
    redoc_url=None,
    openapi_url="/openapi.json" if SWAGGER_ENABLED else None,
)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service health check",
)
def health():    
    return {
        "status": "ok",
        "model": "InsightFace buffalo_l (ArcFace w600k_r50)",
        "device": service.get_active_provider() if service else "not_initialized",
    }

@app.post(
    "/verify/face",
    response_model=FaceVerificationResponse,
    tags=["Verification"],
    summary="Verify face match between selfie and ID document",
    responses={
        200: {"description": "Model ran — check `success` field for pass/fail"},
        500: {"description": "Unhandled system error during inference"},
    },
)
async def verify_face(
    session_id: str = Form(...),
    selfie_image: UploadFile = File(...),
    id_image: UploadFile = File(...),
):    
    selfie_suffix = os.path.splitext(selfie_image.filename or "selfie.jpg")[1] or ".jpg"
    id_suffix = os.path.splitext(id_image.filename or "id.jpg")[1] or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=selfie_suffix) as stmp:
        shutil.copyfileobj(selfie_image.file, stmp)
        selfie_path = stmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=id_suffix) as itmp:
        shutil.copyfileobj(id_image.file, itmp)
        id_path = itmp.name

    try:
        result = service.verify_faces(selfie_path, id_path)
        
        data = FaceVerificationData(
            is_match=result.is_match,
            similarity_score=result.similarity_score,
            confidence=result.confidence,
            failure_reason=result.failure_reason,
        )

        if result.is_match:
            return FaceVerificationResponse(success=True, data=data, error=None)

        stage_map = {
            "image_quality_insufficient": "image_quality",
            "face_not_detected_selfie": "face_detection_selfie",
            "face_not_detected_id": "face_detection_id",
            "multiple_faces_in_selfie": "face_detection_selfie",
            "face_too_small_selfie": "face_detection_selfie",
            "face_too_small_id": "face_detection_id",
            "face_mismatch": "face_matching",
        }
        code = result.failure_reason or "inference_error"
        stage = stage_map.get(code, "face_matching")

        message_map = {
            "image_quality_insufficient": "Selfie image is too blurry for face detection.",
            "face_not_detected_selfie": "No face detected in the selfie image.",
            "face_not_detected_id": "No face detected in the ID document image.",
            "multiple_faces_in_selfie": "More than one face was detected in the selfie.",
            "face_too_small_selfie": "The face in the selfie is too small (below 80×80 px).",
            "face_too_small_id": "The face on the ID document is too small (below 80×80 px).",
            "face_mismatch": "The selfie does not match the face on the provided ID document.",
        }
        message = message_map.get(code, "Face verification failed.")

        return FaceVerificationResponse(
            success=False,
            data=data,
            error=ErrorDetail(code=code, message=message, stage=stage),
        )

    except Exception as exc:
        logger.exception(f"SYSTEM ERROR session_id={session_id}")
        return JSONResponse(
            status_code=500,
            content=FaceVerificationResponse(
                success=False,
                data=None,
                error=ErrorDetail(
                    code="inference_error",
                    message=f"Model inference failed unexpectedly. ({exc})",
                    stage="face_detection_selfie",
                ),
            ).model_dump(),
        )

    finally:
        os.unlink(selfie_path)
        os.unlink(id_path)
