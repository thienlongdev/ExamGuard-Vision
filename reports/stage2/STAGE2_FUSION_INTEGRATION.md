# Stage 2 Multi-Cue Fusion Integration
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative Fusion Integration Specification  
**Date**: 2026-10-03  

---

## 1. UnifiedTrackUpdate Data Contract

To prevent hidden state coupling, all perception, tracking, and spatial association outputs are unified into a canonical `UnifiedTrackUpdate` per track before crossing into the V4D fusion boundary:

```python
@dataclass
class UnifiedTrackUpdate:
    track_id: int
    timestamp_sec: float
    camera_id: str = "cam_0"
    tracking: Optional[TrackingState] = None
    posture: PostureCue = field(default_factory=PostureCue)
    headpose: HeadPoseCue = field(default_factory=HeadPoseCue)
    phone: PhoneCue = field(default_factory=PhoneCue)
    macro_behavior: MacroBehaviorCue = field(default_factory=MacroBehaviorCue)
```

---

## 2. MultiCueFusionEngine Integration

The Stage 2 pipeline feeds `UnifiedTrackUpdate` directly into `src.fusion.fusion_engine.MultiCueFusionEngine.update_track()`. Fusion logic is maintained strictly within `src/fusion/`, preserving a single source of truth.

### Key Fusion Mechanisms
1. **Sliding Window Buffer**: Ingests updates chronologically, enforcing a 30.0s time horizon and 300 sample cap per track.
2. **Competing Evidence Veto**: `NORMAL_READ_WRITE` scores $\ge 0.50$ actively veto `HEAD_REST_SLEEP` candidates, eliminating writing false alarms.
3. **Correlated Cue Discounting**: Yaw orientation is combined with posture turn evidence using reliability weighting ($w_{yaw} = 0.35$), preventing correlated artificial confidence inflation.
4. **Missing Cue Safety**: Unavailable headpose or phone cues receive weight 0.0 without penalizing posture cues.

---

## 3. Ambiguity-Aware Phone Fusion

Phone association status is explicitly evaluated by `PhoneAssociator`:
- `CLEAR_ASSOCIATION`: Phone unambiguously associated with track -> contributes positive evidence.
- `AMBIGUOUS_ASSOCIATION`: Multiple students equidistant to phone -> marked ambiguous, vetoes positive phone event generation.
- `NO_PHONE`: No phone in proximity.
- `NOT_EVALUATED`: Gated or sub-resolution.

This structure eliminates false positive phone accusations in dense exam rows.
