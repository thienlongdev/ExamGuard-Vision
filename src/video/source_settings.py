"""Local source settings and CurrentUser DPAPI protection for optional RTSP credentials."""
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]


def protect_rtsp_url(url: str) -> str:
    from src.security.key_provider import _dpapi_protect
    path = ROOT / "storage/security" / f"camera-{uuid.uuid4().hex}.dpapi"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_dpapi_protect(url.encode("utf-8")))
    return f"dpapi:{path.name}"


def resolve_rtsp_url(reference: str) -> str:
    if not reference.startswith("dpapi:"):
        return reference
    from src.security.key_provider import _dpapi_unprotect
    name = reference.removeprefix("dpapi:")
    if Path(name).name != name or not name.startswith("camera-") or not name.endswith(".dpapi"):
        raise ValueError("Invalid camera secret reference")
    return _dpapi_unprotect((ROOT / "storage/security" / name).read_bytes()).decode("utf-8")
