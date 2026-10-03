# Stage 2 FastAPI & WebSocket API Validation
**Phase**: Stage 2 Full End-to-End Orchestration  
**Status**: Authoritative API Specification  
**Date**: 2026-10-03  

---

## 1. API Endpoints & Contract Preservation

Stage 2 preserves all existing FastAPI endpoints while extending them with model provenance and lifecycle metadata:

| Endpoint | Method | Response Schema | Description |
| :--- | :---: | :--- | :--- |
| `/health` | GET | `HealthResponse` | Liveness health check, returns status and timestamp |
| `/api/cameras` | GET | `List[CameraInfo]` | Active camera streams, resolution, and nominal FPS |
| `/api/events` | GET | `List[EventResponse]` | Filterable event history (`risk_level`, `status`, `limit`) |
| `/api/events/{event_id}` | GET | `EventResponse` | Detailed event inspection record with evidence paths |
| `/api/events/{event_id}` | PATCH | `EventResponse` | Invigilator status update (`reviewed`, `confirmed`, `dismissed`) |
| `/api/system/status` | GET | `SystemStatusResponse`| System metrics, pipeline version, and active model registry |
| `/api/system/models` | GET | `Dict[str, Any]` | Full provenance metadata and SHA-256 digests for all active models |
| `/ws/events` | WS | Real-time JSON stream | Real-time event lifecycle stream (`EVENT_OPEN`, `EVENT_UPDATE`, `EVENT_CLOSE`) |

---

## 2. Observable Event Taxonomy

In strict compliance with governance rules, API payloads use observable factual descriptions:
- `SUSTAINED_HEAD_REST`
- `SUSTAINED_LATERAL_HEAD_ORIENTATION`
- `PHONE_ASSOCIATED`
- `DISCUSSION_CANDIDATE`
- `STANDING`
- `MULTI_CUE_ATTENTION_SHIFT`

Subjective terms (`CHEATING`, `CHEATER`, `GUILTY`, `FRAUD`) are completely prohibited across all API responses.

---

## 3. WebSocket Real-Time Broadcasting

The `/ws/events` endpoint streams event lifecycle transitions without flooding clients with redundant per-frame updates:

```json
{
  "type": "EVENT_OPEN",
  "action": "OPEN",
  "event": {
    "event_id": "9d8e7c6b-5a4f-4321-9876-abcdef012345",
    "track_id": 4,
    "camera_id": "cam_exam_hall_01",
    "event_type": "SUSTAINED_HEAD_REST",
    "start_timestamp": 12.50,
    "last_update_timestamp": 14.50,
    "duration": 2.00,
    "risk_level": "LOW",
    "risk_score": 25.0,
    "evidence_summary": {
      "initial_evidence_score": 0.88,
      "candidate_duration_sec": 2.0,
      "open_snapshot_path": "storage/evidence/snapshots/9d8e..._open.jpg"
    },
    "status": "active"
  }
}
```
All API and WebSocket contracts are validated by `test_stage2_api_contract.py`.
