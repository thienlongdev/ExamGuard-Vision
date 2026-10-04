"""
Stage 2 Crop Scheduler & Batched Inference Engine
=================================================
Implements deterministic cadence scheduling (e.g. Posture 10 Hz, Head-Pose 6 Hz)
and vectorized GPU crop batching for active tracks.

Guarantees:
- Strict track-to-batch index mapping preservation (no track assignment corruption)
- Capability gating (missing or small head -> UNAVAILABLE, never yaw = 0)
- Single GPU inference pass per scheduled modality per frame
- torch.inference_mode() execution
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch

from src.tracking.tracker import Track
from src.detection.types import BBox
from src.fusion.types import (
    PostureCue,
    HeadPoseCue,
    ObservationStatus,
    HeadPoseSupportStatus,
)
from src.orchestration.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class TrackCadenceState:
    """Tracks timing and last inference results for a single track ID."""
    track_id: int
    last_posture_timestamp: float = -1.0
    last_headpose_timestamp: float = -1.0
    latest_posture_cue: PostureCue = field(default_factory=PostureCue)
    latest_headpose_cue: HeadPoseCue = field(default_factory=HeadPoseCue)
    is_attention: bool = False
    last_attention_timestamp: float = -1.0
    total_posture_evaluations: int = 0
    total_headpose_evaluations: int = 0


class CropScheduler:
    """Manages per-track crop extraction, cadence intervals, and GPU batching."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.registry = model_registry
        self.config = config or {}

        # Static / legacy cadence config
        cadence_cfg = self.config.get("cadence", {})
        self.posture_hz = float(cadence_cfg.get("posture_hz", 10.0))
        self.headpose_hz = float(cadence_cfg.get("headpose_hz", 6.0))
        self.posture_interval = 1.0 / max(0.1, self.posture_hz)
        self.headpose_interval = 1.0 / max(0.1, self.headpose_hz)

        # Adaptive scheduling & performance config
        perf_cfg = self.config.get("performance", {})
        self.adaptive_scheduling = bool(perf_cfg.get("adaptive_scheduling", True))
        self.normal_posture_hz = float(perf_cfg.get("normal_posture_hz", 4.0))
        self.attention_posture_hz = float(perf_cfg.get("attention_posture_hz", 8.0))
        self.normal_headpose_hz = float(perf_cfg.get("normal_headpose_hz", 4.0))
        self.attention_headpose_hz = float(perf_cfg.get("attention_headpose_hz", 8.0))
        self.max_starvation_sec = float(perf_cfg.get("max_starvation_interval_sec", 0.35))
        self.attention_stickiness_sec = float(perf_cfg.get("attention_stickiness_sec", 1.0))

        batch_cfg = self.config.get("batching", {})
        self.max_posture_batch = int(batch_cfg.get("max_posture_batch_size", 32))
        self.max_headpose_batch = int(batch_cfg.get("max_headpose_batch_size", 32))
        self.crop_padding = float(batch_cfg.get("crop_padding_ratio", 0.05))
        self.min_person_h = float(batch_cfg.get("min_person_height_px", 120.0))
        self.low_res_person_h = float(batch_cfg.get("posture_low_resolution_min_height_px", 60.0))
        self.min_head_dim = float(batch_cfg.get("min_head_dimension_px", 25.0))

        # Resolution config
        posture_model_cfg = self.config.get("models", {}).get("posture", {})
        self.posture_res = int(posture_model_cfg.get("primary_resolution", 224))
        self.device = self.registry.device

        # Per-track scheduling state
        self._track_states: Dict[int, TrackCadenceState] = {}

    def _get_track_state(self, track_id: int) -> TrackCadenceState:
        if track_id not in self._track_states:
            self._track_states[track_id] = TrackCadenceState(track_id=track_id)
        return self._track_states[track_id]

    def cleanup_expired_tracks(self, active_track_ids: List[int]) -> None:
        """Prune tracks that are no longer active to prevent memory growth."""
        active_set = set(active_track_ids)
        stale_ids = [t_id for t_id in self._track_states if t_id not in active_set]
        for t_id in stale_ids:
            self._track_states.pop(t_id, None)

    def reset(self) -> None:
        """Reset all per-track scheduling state."""
        self._track_states.clear()

    def _preprocess_crop(self, crop_bgr: np.ndarray, target_size: int) -> torch.Tensor:
        """Resize, convert BGR->RGB, normalize with ImageNet mean/std, to (3, H, W) tensor."""
        if crop_bgr.shape[0] != target_size or crop_bgr.shape[1] != target_size:
            resized = cv2.resize(crop_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            resized = crop_bgr
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).contiguous().float()
        return tensor

    def extract_person_crop(self, frame: np.ndarray, bbox: Any) -> Tuple[Optional[np.ndarray], str, float]:
        """Extract padded person crop clamped to frame boundaries with scale eligibility check.
        
        Returns:
            Tuple of (crop, scale_status, reliability_scale)
            - Height >= 120 px: (crop, "POSTURE_PRIMARY_224_ELIGIBLE", 1.0)
            - 60 <= Height < 120 px: (crop, "POSTURE_LOW_RESOLUTION", 0.60)
            - Height < 60 px: (None, "NOT_ELIGIBLE", 0.0)
        """
        if not isinstance(bbox, BBox):
            if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
                bbox = BBox(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            else:
                return None, "NOT_ELIGIBLE", 0.0

        h, w = frame.shape[:2]
        pad_x = bbox.width * self.crop_padding
        pad_y = bbox.height * self.crop_padding

        x1 = max(0, int(bbox.x1 - pad_x))
        y1 = max(0, int(bbox.y1 - pad_y))
        x2 = min(w, int(bbox.x2 + pad_x))
        y2 = min(h, int(bbox.y2 + pad_y))

        crop_h = y2 - y1
        crop_w = x2 - x1

        if crop_w < 10 or crop_h < self.low_res_person_h:
            return None, "NOT_ELIGIBLE", 0.0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, "NOT_ELIGIBLE", 0.0

        if crop_h < self.min_person_h:
            # Low resolution range: 60 <= h < 120. Reduced reliability, not full confidence.
            return crop, "POSTURE_LOW_RESOLUTION", 0.60
        else:
            return crop, "POSTURE_PRIMARY_224_ELIGIBLE", 1.0

    def extract_head_crop(self, frame: np.ndarray, bbox: BBox) -> Tuple[Optional[np.ndarray], HeadPoseSupportStatus]:
        """Extract top head region from person bounding box with capability check."""
        h, w = frame.shape[:2]
        person_w = bbox.width
        person_h = bbox.height

        # Head region is top 30% vertically, middle 70% horizontally
        x1_head = max(0, int(bbox.x1 + 0.15 * person_w))
        x2_head = min(w, int(bbox.x2 - 0.15 * person_w))
        y1_head = max(0, int(bbox.y1))
        y2_head = min(h, int(bbox.y1 + 0.32 * person_h))

        head_w = x2_head - x1_head
        head_h = y2_head - y1_head

        if head_w < self.min_head_dim or head_h < self.min_head_dim:
            return None, HeadPoseSupportStatus.FACE_UNRESOLVABLE

        crop = frame[y1_head:y2_head, x1_head:x2_head]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None, HeadPoseSupportStatus.FACE_UNRESOLVABLE

        return crop, HeadPoseSupportStatus.WITHIN_PRIMARY_SUPPORT

    def schedule_and_infer(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        timestamp_sec: float,
        attention_track_ids: Optional[Any] = None,
    ) -> Tuple[Dict[int, PostureCue], Dict[int, HeadPoseCue], Dict[str, float]]:
        """
        Determine which tracks require inference based on cadence & attention state,
        batch crops, execute GPU inference with exact track index preservation,
        and return per-track cues.
        """
        posture_results: Dict[int, PostureCue] = {}
        headpose_results: Dict[int, HeadPoseCue] = {}

        if not tracks:
            return posture_results, headpose_results, {
                "crop_preprocess_ms": 0.0,
                "posture_ms": 0.0,
                "headpose_ms": 0.0,
                "crop_extraction_ms": 0.0,
                "posture_preprocess_ms": 0.0,
                "posture_inference_ms": 0.0,
                "headpose_preprocess_ms": 0.0,
                "headpose_inference_ms": 0.0,
            }

        # Normalize attention set
        attn_set = set(attention_track_ids) if attention_track_ids is not None else set()

        # 1. Determine scheduled candidates and measure extraction / prep
        t_prep_start = time.perf_counter()
        t_crop_extract_total = 0.0
        t_pos_prep_total = 0.0
        t_hp_prep_total = 0.0
        posture_scheduled_tracks: List[Tuple[int, torch.Tensor, str, float]] = []
        headpose_scheduled_tracks: List[Tuple[int, torch.Tensor, HeadPoseSupportStatus]] = []

        for track in tracks:
            state = self._get_track_state(track.track_id)

            # Determine attention state
            if attention_track_ids is not None:
                has_external_attn = track.track_id in attn_set
                recent_sticky_attn = (
                    state.last_attention_timestamp >= 0.0 and
                    (timestamp_sec - state.last_attention_timestamp) < self.attention_stickiness_sec
                )
                is_attn = has_external_attn or recent_sticky_attn
            else:
                cue_attn = False
                if state.latest_posture_cue.status == ObservationStatus.AVAILABLE:
                    if (
                        state.latest_posture_cue.predicted_class in ("TURN_HEAD_CLEAR", "HEAD_REST_SLEEP") and
                        state.latest_posture_cue.confidence >= 0.35
                    ):
                        cue_attn = True
                if (
                    state.latest_headpose_cue.status == ObservationStatus.AVAILABLE and
                    state.latest_headpose_cue.yaw_deg is not None
                ):
                    if abs(state.latest_headpose_cue.yaw_deg) >= 22.0:
                        cue_attn = True

                recent_sticky_attn = (
                    state.last_attention_timestamp >= 0.0 and
                    (timestamp_sec - state.last_attention_timestamp) < self.attention_stickiness_sec
                )
                is_attn = cue_attn or recent_sticky_attn

            if is_attn:
                state.is_attention = True
                state.last_attention_timestamp = timestamp_sec
            else:
                state.is_attention = False

            # Determine effective target cadence
            if self.adaptive_scheduling:
                pos_hz = self.attention_posture_hz if is_attn else self.normal_posture_hz
                hp_hz = self.attention_headpose_hz if is_attn else self.normal_headpose_hz
                pos_interval = 1.0 / max(0.1, pos_hz)
                hp_interval = 1.0 / max(0.1, hp_hz)
            else:
                pos_interval = self.posture_interval
                hp_interval = self.headpose_interval

            # Check posture cadence with anti-starvation floor
            time_since_pos = timestamp_sec - state.last_posture_timestamp
            run_pos = (
                state.last_posture_timestamp < 0.0 or
                time_since_pos >= (pos_interval - 1e-4) or
                (self.adaptive_scheduling and time_since_pos >= self.max_starvation_sec)
            )

            if run_pos:
                t_c0 = time.perf_counter()
                p_crop, scale_status, rel_scale = self.extract_person_crop(frame, track.bbox)
                t_crop_extract_total += (time.perf_counter() - t_c0)
                if p_crop is not None:
                    t_p0 = time.perf_counter()
                    tensor = self._preprocess_crop(p_crop, self.posture_res)
                    t_pos_prep_total += (time.perf_counter() - t_p0)
                    posture_scheduled_tracks.append((track.track_id, tensor, scale_status, rel_scale))
                else:
                    # Crop sub-resolution: mark UNAVAILABLE and record cadence timestamp
                    post_cue = PostureCue(
                        status=ObservationStatus.UNAVAILABLE,
                        crop_quality="SUB_RESOLUTION",
                        reliability_weight=0.0,
                    )
                    posture_results[track.track_id] = post_cue
                    state.last_posture_timestamp = timestamp_sec
                    state.latest_posture_cue = post_cue

            # Check headpose cadence with anti-starvation floor
            time_since_hp = timestamp_sec - state.last_headpose_timestamp
            run_hp = (
                state.last_headpose_timestamp < 0.0 or
                time_since_hp >= (hp_interval - 1e-4) or
                (self.adaptive_scheduling and time_since_hp >= self.max_starvation_sec)
            )

            if run_hp:
                t_h0 = time.perf_counter()
                h_crop, hp_status = self.extract_head_crop(frame, track.bbox)
                t_crop_extract_total += (time.perf_counter() - t_h0)
                if h_crop is not None and hp_status == HeadPoseSupportStatus.WITHIN_PRIMARY_SUPPORT:
                    t_hp0 = time.perf_counter()
                    tensor = self._preprocess_crop(h_crop, 224)
                    t_hp_prep_total += (time.perf_counter() - t_hp0)
                    headpose_scheduled_tracks.append((track.track_id, tensor, hp_status))
                else:
                    # Capability gating: missing headpose is UNAVAILABLE, never yaw = 0
                    hp_cue = HeadPoseCue(
                        status=ObservationStatus.UNAVAILABLE,
                        yaw_deg=None,
                        support_status=hp_status,
                    )
                    headpose_results[track.track_id] = hp_cue
                    state.last_headpose_timestamp = timestamp_sec
                    state.latest_headpose_cue = hp_cue
        t_prep_end = time.perf_counter()

        # 2. Batched Posture Inference
        if posture_scheduled_tracks:
            t_pos_start = time.perf_counter()
            # Batch slicing up to max_posture_batch
            for start_idx in range(0, len(posture_scheduled_tracks), self.max_posture_batch):
                batch_slice = posture_scheduled_tracks[start_idx : start_idx + self.max_posture_batch]
                t_ids = [item[0] for item in batch_slice]
                scale_statuses = [item[2] for item in batch_slice]
                rel_scales = [item[3] for item in batch_slice]
                tensors = torch.stack([item[1] for item in batch_slice], dim=0).to(self.device)

                try:
                    with torch.inference_mode():
                        outputs = self.registry.posture_predictor.predict_tensor(
                            tensor_batch=tensors,
                            student_ids=t_ids,
                        )
                except Exception as e:
                    logger.warning(f"Posture batch prediction failed: {e}")
                    outputs = []

                # Map outputs back to tracks with exact index preservation
                for i, out in enumerate(outputs):
                    t_id = t_ids[i]
                    sc_status = scale_statuses[i]
                    r_scale = rel_scales[i]
                    probs = {
                        "NORMAL_UPRIGHT": out.normal_upright_score,
                        "NORMAL_READ_WRITE": out.normal_read_write_score,
                        "HEAD_REST_SLEEP": out.head_rest_sleep_score,
                        "TURN_HEAD_CLEAR": out.turn_head_clear_score,
                    }
                    crop_q = "POSTURE_LOW_RESOLUTION" if sc_status == "POSTURE_LOW_RESOLUTION" else out.crop_quality
                    cue = PostureCue(
                        status=ObservationStatus.AVAILABLE,
                        probabilities=probs,
                        predicted_class=out.predicted_class,
                        confidence=out.confidence,
                        crop_quality=crop_q,
                        resolution_used=self.posture_res,
                        reliability_weight=float(r_scale),
                    )
                    posture_results[t_id] = cue
                    state = self._get_track_state(t_id)
                    state.last_posture_timestamp = timestamp_sec
                    state.latest_posture_cue = cue
                    state.total_posture_evaluations += 1

                # Handle any tracks where inference failed
                if not outputs:
                    for t_id in t_ids:
                        posture_results[t_id] = PostureCue(status=ObservationStatus.UNAVAILABLE)

            t_pos_end = time.perf_counter()
            t_pos_ms = (t_pos_end - t_pos_start) * 1000.0
        else:
            t_pos_ms = 0.0

        # 3. Batched Head-Pose Inference
        if headpose_scheduled_tracks:
            t_hp_start = time.perf_counter()
            for start_idx in range(0, len(headpose_scheduled_tracks), self.max_headpose_batch):
                batch_slice = headpose_scheduled_tracks[start_idx : start_idx + self.max_headpose_batch]
                t_ids = [item[0] for item in batch_slice]
                tensors = torch.stack([item[1] for item in batch_slice], dim=0).to(self.device)

                try:
                    with torch.inference_mode():
                        outputs = self.registry.headpose_predictor.predict_tensor(
                            tensor_batch=tensors,
                            student_ids=t_ids,
                        )
                except Exception as e:
                    logger.warning(f"Headpose batch prediction failed: {e}")
                    outputs = []

                for i, out in enumerate(outputs):
                    t_id = t_ids[i]
                    cue = HeadPoseCue(
                        status=ObservationStatus.AVAILABLE,
                        yaw_deg=out.yaw_deg,
                        pitch_deg=out.pitch_deg,
                        roll_deg=out.roll_deg,
                        reliability_weight=out.pose_confidence,
                        source_model="HopeNet-Yaw",
                        support_status=HeadPoseSupportStatus.WITHIN_PRIMARY_SUPPORT,
                    )
                    headpose_results[t_id] = cue
                    state = self._get_track_state(t_id)
                    state.last_headpose_timestamp = timestamp_sec
                    state.latest_headpose_cue = cue
                    state.total_headpose_evaluations += 1

                if not outputs:
                    for t_id in t_ids:
                        headpose_results[t_id] = HeadPoseCue(
                            status=ObservationStatus.UNAVAILABLE,
                            yaw_deg=None,
                            support_status=HeadPoseSupportStatus.FACE_UNRESOLVABLE,
                        )
            t_hp_end = time.perf_counter()
            t_hp_ms = (t_hp_end - t_hp_start) * 1000.0
        else:
            t_hp_ms = 0.0

        # 4. Fill in skipped / un-scheduled tracks from recent cached state
        for track in tracks:
            t_id = track.track_id
            state = self._get_track_state(t_id)

            if t_id not in posture_results:
                # Use latest posture cue with cadence bridged status
                cached_cue = state.latest_posture_cue
                if cached_cue.status == ObservationStatus.AVAILABLE:
                    posture_results[t_id] = cached_cue
                else:
                    posture_results[t_id] = PostureCue(status=ObservationStatus.NOT_EVALUATED)

            if t_id not in headpose_results:
                cached_hp = state.latest_headpose_cue
                if cached_hp.status == ObservationStatus.AVAILABLE:
                    headpose_results[t_id] = cached_hp
                else:
                    headpose_results[t_id] = HeadPoseCue(status=ObservationStatus.NOT_EVALUATED)

        timings = {
            "crop_extraction_ms": round(t_crop_extract_total * 1000.0, 3),
            "posture_preprocess_ms": round(t_pos_prep_total * 1000.0, 3),
            "posture_inference_ms": round(t_pos_ms, 3),
            "headpose_preprocess_ms": round(t_hp_prep_total * 1000.0, 3),
            "headpose_inference_ms": round(t_hp_ms, 3),
            # Backward-compatibility mappings
            "crop_preprocess_ms": round((t_crop_extract_total + t_pos_prep_total + t_hp_prep_total) * 1000.0, 3),
            "posture_ms": round(t_pos_ms, 3),
            "headpose_ms": round(t_hp_ms, 3),
        }
        return posture_results, headpose_results, timings
