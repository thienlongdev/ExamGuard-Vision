import os
import re
from pathlib import Path

def audit_secrets():
    repo_root = Path(__file__).resolve().parent.parent
    patterns = [
        re.compile(r'(api[_-]?key|secret[_-]?key|private[_-]?key|auth[_-]?token|password)\s*[:=]\s*["\']([^"\']{8,})["\']', re.IGNORECASE),
        re.compile(r'ghp_[A-Za-z0-9]{30,}', re.IGNORECASE),
        re.compile(r'github_pat_[A-Za-z0-9_]{30,}', re.IGNORECASE),
        re.compile(r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}', re.IGNORECASE)
    ]

    skip_dirs = {'.git', '.venv', '__pycache__', 'datasets', 'storage', '.pytest_cache'}
    found_issues = []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in {'.py', '.json', '.yaml', '.yml', '.md', '.txt', '.sh', '.bat', '.ps1'}:
                continue
            file_path = Path(root) / file
            rel_path = file_path.relative_to(repo_root)
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                for i, line in enumerate(content.splitlines(), 1):
                    for pat in patterns:
                        match = pat.search(line)
                        if match:
                            # Filter out harmless mock/test keys
                            val = match.group(0)
                            if any(mock in val.lower() for mock in ['dummy', 'mock', 'placeholder', 'none', 'null', 'false', 'true']):
                                continue
                            found_issues.append((str(rel_path), i, line.strip()))
            except Exception as e:
                print(f"Error reading {rel_path}: {e}")

    print(f"Secret Audit Results: {len(found_issues)} suspicious findings.")
    for path, line_no, text in found_issues:
        print(f"  {path}:{line_no}: {text[:100]}")
    return len(found_issues)

if __name__ == '__main__':
    audit_secrets()
