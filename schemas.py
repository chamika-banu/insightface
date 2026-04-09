# These models are used by FastAPI to generate accurate OpenAPI/Swagger docs

from pydantic import BaseModel


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


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
