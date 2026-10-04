"""Embedded dashboard UI loader for ExamGuard Vision."""

from pathlib import Path

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "dashboard.html"
_LOGIN_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "login.html"
_SETUP_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "setup.html"


def load_dashboard_html() -> str:
    """Load production dashboard HTML template from disk."""
    if _TEMPLATE_PATH.is_file():
        return _TEMPLATE_PATH.read_text(encoding="utf-8")
    return (
        "<!DOCTYPE html><html><head>"
        "<title>ExamGuard Vision — Exam Suspicious Behavior Monitoring</title>"
        "</head><body><script>function connectWebSocket(){}</script></body></html>"
    )


def load_login_html() -> str:
    """Load login HTML template from disk."""
    if _LOGIN_TEMPLATE_PATH.is_file():
        return _LOGIN_TEMPLATE_PATH.read_text(encoding="utf-8")
    return "<h1>ExamGuard Vision — Đăng nhập</h1>"


def load_setup_html() -> str:
    """Load setup HTML template from disk."""
    if _SETUP_TEMPLATE_PATH.is_file():
        return _SETUP_TEMPLATE_PATH.read_text(encoding="utf-8")
    return "<h1>ExamGuard Vision — Thiết lập quản trị viên</h1>"


DASHBOARD_HTML = load_dashboard_html()
LOGIN_HTML = load_login_html()
SETUP_HTML = load_setup_html()
