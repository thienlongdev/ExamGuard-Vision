# EVIDENCE PRIVACY & GOVERNANCE POLICY

## 1. Core Ethical & Legal Mandate
The Suspicious Behavior Detection system is deployed exclusively as a **decision-support tool for certified human invigilators**. To protect student privacy, civil liberties, and institutional integrity, the system enforces strict architectural guardrails:

1. **Zero Biometric Identity Profiling**:
   - Facial recognition models, facial embeddings, face database lookups, and biometric student naming are **strictly prohibited** and physically excluded from the codebase.
   - All student tracking operates exclusively on anonymous numerical identifiers (`track_id: 1`, `track_id: 2`) assigned by ByteTrack for the duration of track visibility.
2. **Prohibition of Guilt Verdicts**:
   - The system NEVER declares `CHEATING`, `CHEATER`, `GUILTY`, or `FRAUD` as model output, event taxonomy, API response, dashboard notification, or database record.
   - The system records factual, observable behavioral cues (e.g., `SUSTAINED_HEAD_REST`, `SUSTAINED_LATERAL_HEAD_ORIENTATION`, `PHONE_ASSOCIATED`, `DISCUSSION_CANDIDATE`, `STANDING`).
   - Risk categories (`LOW`, `MEDIUM`, `HIGH`) represent engineered evidence-confidence heuristics, NOT legal or moral probabilities of guilt.
3. **No Continuous Full-Room Archival**:
   - The pilot evidence manager disables continuous full-room recording by default. If continuous footage is legally required, it must be managed through the school's existing certified CCTV infrastructure with independent access controls.
   - Media storage is strictly bounded to short, event-triggered snapshots and rolling clips (pre-event and post-event buffers).

---

## 2. Evidence Recording Modes
The system supports four explicit evidence capture modes configured in `configs/pilot/privacy_policy_template.yaml`:

| Mode | Behavior | Privacy Level | Storage Impact |
| :--- | :--- | :--- | :--- |
| `DISABLED` | Only JSON event metadata is published to WebSocket. Zero JPEG snapshots or MP4 clips written to disk. | Highest Privacy | Zero media disk usage |
| `EVENT_SNAPSHOT_ONLY` | Captures 1 opening JPEG snapshot and up to 1 escalation snapshot if risk escalates to HIGH. | High Privacy | Minimal (~150 KB per event) |
| `EVENT_CLIP` | Captures opening snapshot and an MP4 video clip spanning only the active event duration. | Balanced | Moderate (~2-5 MB per event) |
| `EVENT_CLIP_WITH_PREBUFFER` | Captures opening snapshot, escalation snapshot, and rolling MP4 clip with 3.0s pre-buffer and 3.0s post-buffer. | Recommended for Pilot | Controlled (~5-12 MB per event) |

---

## 3. Evidence Manifest Specification
Every evidence record stored under `storage/evidence/manifests/<event_id>_manifest.json` contains comprehensive provenance metadata to ensure auditability and prevent tampering:

```json
{
  "event_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "camera_id": "cam_hall_01_front",
  "track_id": 4,
  "room_id": "hall_101",
  "event_type": "SUSTAINED_LATERAL_HEAD_ORIENTATION",
  "risk_level": "MEDIUM",
  "risk_score": 58.4,
  "start_timestamp_sec": 1727931405.120,
  "end_timestamp_sec": 1727931408.350,
  "duration_sec": 3.230,
  "model_hashes": {
    "yolo26m.pt": "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7",
    "v4_posture_best.pt": "529a23f96ebec6051ad29279fba3cd5279e7aa8fe2e46292ab4bb9c9523ca180",
    "v4_headpose_yaw_best.pt": "5d15eec5941cfc8d2468d09c27bff8eca2014e62bb12fc9a95c0c6239fbbca55"
  },
  "fusion_config_version": "4.0.0-v4d",
  "camera_profile_version": "1.0.0-pilot",
  "open_snapshot_path": "storage/evidence/snapshots/9b1deb4d_open.jpg",
  "clip_path": "storage/evidence/clips/9b1deb4d.mp4",
  "operator_review_status": "NEW",
  "reviewer_notes": null,
  "reviewed_at": null
}
```

---

## 4. Institutional Access Control
- Access to the dashboard and evidence directory is restricted to authorized exam invigilators and proctoring supervisors.
- Export of raw evidence clips outside the examination review environment requires formal dual-key authorization and audit logging.
- Video files are encrypted at rest where host OS encryption (e.g. BitLocker) is active.
