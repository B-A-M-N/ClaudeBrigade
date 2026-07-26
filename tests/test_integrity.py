import hashlib
import pathlib
import pytest


def test_sha256sums_manifest_parity():
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest_path = root / "SHA256SUMS"
    if not manifest_path.exists():
        pytest.skip("SHA256SUMS manifest does not exist")

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        expected_hash, rel_path = parts[0], parts[1].lstrip("*").strip()

        # SHA256SUMS should not include itself
        assert rel_path != "SHA256SUMS"

        file_path = root / rel_path
        assert file_path.exists(), f"Manifest file missing: {rel_path}"
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
