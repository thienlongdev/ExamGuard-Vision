"""Review tracked text or exact staged blobs without printing secret values.

Pattern matches require human review: synthetic test credentials can match too.
Ignored runtime data and untracked files are never traversed.
"""

import argparse
from pathlib import Path
import re
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
PATTERNS = (
    ("credential_literal", re.compile(
        r"(?:api[_-]?key|secret[_-]?key|private[_-]?key|auth[_-]?token|password)"
        r"\s*[:=]\s*[\"']([^\"'\n]{8,})[\"']", re.IGNORECASE)),
    ("github_token", re.compile(r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("provider_token", re.compile(r"(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|tskey-[A-Za-z0-9_-]{16,})")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
)
TEXT_SUFFIXES = {".py", ".js", ".css", ".html", ".json", ".yaml", ".yml",
                 ".md", ".txt", ".sh", ".bat", ".ps1", ".toml", ".ini",
                 ".cfg", ".env", ".example", ".template"}


def audit_secrets(repo_root=None, *, staged=False):
    """Return the finding count; output contains only path, line and rule name."""
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    args = ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if staged else ["ls-files", "-z"]
    paths = subprocess.check_output(["git", "-C", str(root), *args]).decode("utf-8").split("\0")
    count = 0
    for relative in filter(None, paths):
        path = Path(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES and not path.name.startswith(".env"):
            continue
        if staged:
            raw = subprocess.check_output(["git", "-C", str(root), "show", f":{relative}"])
        else:
            local = root / path
            if not local.is_file():
                continue
            raw = local.read_bytes()
        for line_number, line in enumerate(raw.decode("utf-8-sig", errors="replace").splitlines(), 1):
            for name, pattern in PATTERNS:
                if pattern.search(line):
                    count += 1
                    print(f"{relative}:{line_number}: {name} (value redacted)")
    print(f"Secret audit: {count} pattern matches; review fixtures and rotate any real exposed credentials.")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Scan exact staged blobs, excluding deletions")
    options = parser.parse_args()
    raise SystemExit(1 if audit_secrets(staged=options.staged) else 0)
