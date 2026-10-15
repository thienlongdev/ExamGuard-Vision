"""Security tooling must scan the intended Git content without disclosing it."""
import subprocess

import pytest

from tools.dev import audit_repo_secrets as audit


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    return tmp_path


def stage(repo, name):
    subprocess.run(["git", "-C", str(repo), "add", "--", name], check=True)


def test_uses_repository_root_and_redacts_findings(repo, monkeypatch, capsys):
    token = "ghp_" + "A" * 36
    (repo / "app.py").write_text(f'token = "{token}"\n', encoding="utf-8")
    stage(repo, "app.py")
    monkeypatch.setattr(audit, "REPO_ROOT", repo)
    monkeypatch.chdir(repo.parent)
    assert audit.audit_secrets() == 1
    output = capsys.readouterr().out
    assert "app.py:1: github_token" in output
    assert token not in output


def test_staged_scan_uses_index_even_when_working_file_is_clean(repo, capsys):
    token = "ghp_" + "B" * 36
    (repo / "app.py").write_text(token, encoding="utf-8")
    stage(repo, "app.py")
    (repo / "app.py").write_text("# cleaned but not staged\n", encoding="utf-8")
    assert audit.audit_secrets(repo, staged=True) == 1
    assert token not in capsys.readouterr().out


def test_staged_scan_ignores_unstaged_and_ignored_data(repo, capsys):
    (repo / ".gitignore").write_text("storage/\n", encoding="utf-8")
    (repo / "app.py").write_text("# safe staged source\n", encoding="utf-8")
    stage(repo, "app.py")
    token = "ghp_" + "C" * 36
    (repo / "app.py").write_text(token, encoding="utf-8")
    (repo / "storage").mkdir()
    (repo / "storage/private.env").write_text(token, encoding="utf-8")
    assert audit.audit_secrets(repo, staged=True) == 0
    assert "private.env" not in capsys.readouterr().out


def test_staged_scan_handles_deleted_files(repo):
    (repo / "obsolete.py").write_text("# staged then removed\n", encoding="utf-8")
    stage(repo, "obsolete.py")
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test Fixture",
         "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "rm", "--cached", "--", "obsolete.py"], check=True, capture_output=True)
    assert audit.audit_secrets(repo, staged=True) == 0
