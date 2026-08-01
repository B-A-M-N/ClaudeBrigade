import subprocess
from workspace_fingerprint import fingerprint, git_has_head


def test_fingerprint_with_head(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)

    # Initial commit
    file1 = repo / "file1.txt"
    file1.write_text("initial content")
    subprocess.run(["git", "add", "file1.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)

    assert git_has_head(repo) is True
    fp1 = fingerprint(repo)
    assert len(fp1) == 64

    # Mutate file
    file1.write_text("modified content")
    fp2 = fingerprint(repo)
    assert fp1 != fp2

    # Revert modification
    file1.write_text("initial content")
    assert fingerprint(repo) == fp1


def test_fingerprint_without_head(tmp_path):
    repo = tmp_path / "new_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)

    assert git_has_head(repo) is False
    # Should calculate fingerprint without throwing bad revision HEAD error
    fp1 = fingerprint(repo)
    assert len(fp1) == 64

    # Add untracked file
    untracked = repo / "new_file.txt"
    untracked.write_text("hello")
    fp2 = fingerprint(repo)
    assert fp1 != fp2
