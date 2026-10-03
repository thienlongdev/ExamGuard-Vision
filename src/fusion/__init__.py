"""
V4D Temporal & Multi-Cue Fusion Package
=======================================
State-of-the-art temporal buffer, capability gating, evidence reliability weighting,
multi-cue fusion, event state machine, and risk aggregation.
"""

from src.fusion.types import (
    ObservationStatus,
    PostureClass,
    EventFamily,
    RiskLevel,
    EventLifecycleState,
    HeadPoseSupportStatus,
    PhoneAssociationStatus,
    CameraProfile,
    PostureCue,
    HeadPoseCue,
    PhoneCue,
    MacroBehaviorCue,
    TrackingState,
    UnifiedTrackUpdate,
    FusedEvent,
    SourceOrigin,
)

from src.fusion.capability import CapabilityGate
from src.fusion.reliability import ReliabilityModel
from src.fusion.temporal_buffer import TemporalBuffer, TrackObservationBuffer
from src.fusion.cue_state import PerTrackCueState
from src.fusion.fusion_engine import MultiCueFusionEngine
from src.fusion.event_engine import EventEngine, TrackEventStateMachine
from src.fusion.risk_aggregator import RiskAggregator
from src.fusion.replay import ReplayEngine

__all__ = [
    "ObservationStatus",
    "PostureClass",
    "EventFamily",
    "RiskLevel",
    "EventLifecycleState",
    "HeadPoseSupportStatus",
    "PhoneAssociationStatus",
    "CameraProfile",
    "PostureCue",
    "HeadPoseCue",
    "PhoneCue",
    "MacroBehaviorCue",
    "TrackingState",
    "UnifiedTrackUpdate",
    "FusedEvent",
    "SourceOrigin",
    "CapabilityGate",
    "ReliabilityModel",
    "TemporalBuffer",
    "TrackObservationBuffer",
    "PerTrackCueState",
    "MultiCueFusionEngine",
    "EventEngine",
    "TrackEventStateMachine",
    "RiskAggregator",
    "ReplayEngine",
]
