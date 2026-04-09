from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()
import onnxruntime as ort
from insightface.app import FaceAnalysis

SIMILARITY_THRESHOLD = 0.42
MIN_FACE_SIZE = 80
MIN_DETECTION_CONFIDENCE = 0.80

@dataclass
class FaceVerificationResult:
    is_match: bool
    similarity_score: float
    confidence: float
    failure_reason: str | None

class InsightFaceService:
    """InsightFace buffalo_l face verification service (CPU, ONNX Runtime)."""

    def __init__(self) -> None:
        available_providers = ort.get_available_providers()
        print(f"[InsightFaceService] Available ONNX providers: {available_providers}")
        
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        print(f"[InsightFaceService] Using providers: {providers}")
        
        print("[InsightFaceService] Downloading/Loading models... (This is ~330MB and will take a moment if not cached!)")
        
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("[InsightFaceService] Model ready.")

    def get_active_provider(self) -> str:
        """Return the name of the execution provider currently in use."""
        try:                                    
            return self.app.models["detection"].session.get_providers()[0]
        except (AttributeError, KeyError, IndexError):
            return "Unknown"

    def _load_image(self, image_path: str) -> np.ndarray:        
        """Load an image, handling EXIF orientation (rotation) metadata."""
        try:
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                rgb_img = np.array(pil_img.convert("RGB"))
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                return bgr_img
        except Exception as e:
            raise ValueError(f"Failed to load image from '{image_path}': {e}")

    def detect_faces(self, image_path: str) -> list:
        """Detect all faces in an image and return the raw InsightFace Face list."""
        img = self._load_image(image_path)
        return self.app.get(img)

    def get_primary_face(
        self,
        image_path: str,
        label: str,
    ) -> tuple:
        """Return the highest-confidence face that passes all quality checks."""
        faces = self.detect_faces(image_path)

        # No faces found
        if len(faces) == 0:
            return None, f"face_not_detected_{label}"

        # Selfies must contain exactly one face
        if len(faces) > 1 and label == "selfie":
            return None, "multiple_faces_in_selfie"

        # Pick the face with the highest detection score
        face = max(faces, key=lambda f: f.det_score)

        # Confidence gate
        if face.det_score < MIN_DETECTION_CONFIDENCE:
            return None, f"face_not_detected_{label}"

        # Size gate — bounding box is [x1, y1, x2, y2]
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            return None, f"face_too_small_{label}"

        return face, None

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two ArcFace embeddings.

        insightface 0.7.3 does not guarantee L2-normalisation on the raw
        face.embedding output, so we normalise here before the dot product.
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1 / norm1, emb2 / norm2))

    def verify_faces(
        self,
        selfie_path: str,
        id_image_path: str,
    ) -> FaceVerificationResult:
        """Full face verification pipeline: detect → embed → compare."""
        selfie_face, selfie_error = self.get_primary_face(selfie_path, "selfie")        
        if selfie_face is None:
            return FaceVerificationResult(
                is_match=False,
                similarity_score=0.0,
                confidence=0.0,
                failure_reason=selfie_error,
            )

        id_face, id_error = self.get_primary_face(id_image_path, "id")        
        if id_face is None:
            return FaceVerificationResult(
                is_match=False,
                similarity_score=0.0,
                confidence=0.0,
                failure_reason=id_error,
            )

        similarity = self.cosine_similarity(selfie_face.embedding, id_face.embedding)        

        is_match = similarity >= SIMILARITY_THRESHOLD
        confidence = min(1.0, 0.5 + abs(similarity - SIMILARITY_THRESHOLD) * 2)

        return FaceVerificationResult(
            is_match=is_match,
            similarity_score=round(similarity, 6),
            confidence=round(confidence, 6),
            failure_reason=None if is_match else "face_mismatch",
        )
