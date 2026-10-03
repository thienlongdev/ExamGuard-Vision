"""
Stage 2 End-to-End Orchestration Package
=======================================
"""

from src.orchestration.model_registry import ModelRegistry, compute_sha256
from src.orchestration.phone_associator import PhoneAssociator, TrackPhoneAssociation
from src.orchestration.crop_scheduler import CropScheduler
from src.orchestration.evidence_manager import IntegratedEvidenceManager
from src.orchestration.stage2_pipeline import Stage2Pipeline, Stage2FrameResult, Stage2FrameMetrics

__all__ = [
    "ModelRegistry",
    "compute_sha256",
    "PhoneAssociator",
    "TrackPhoneAssociation",
    "CropScheduler",
    "IntegratedEvidenceManager",
    "Stage2Pipeline",
    "Stage2FrameResult",
    "Stage2FrameMetrics",
]
