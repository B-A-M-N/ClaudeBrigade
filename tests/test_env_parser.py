"""Comprehensive tests for router/enhanced_router/env_parser.py."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from enhanced_router.env_parser import EnvParseError, parse_env_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    """Write text to *path* and set mode 0o640 (owner rw, group r, others none)."""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o640)


def _tmpfile(content: str) -> Path:
    """Create a temporary file with *content* and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    # Set restrictive permissions
    os.chmod(f.name, 0o640)
    return Path(f.name)


def _tmpfile_raw(content: bytes) -> Path:
    """Create a temporary file with raw *bytes* and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".env", delete=False
    )
    f.write(content)
    f.close()
    os.chmod(f.name, 0o640)
    return Path(f.name)


# ---------------------------------------------------------------------------
# 1. Basic parsing
# ---------------------------------------------------------------------------


def test_basic_parsing():
    path = _tmpfile("API_KEY=abc123\nMODEL_ID=opus\n")
    try:
        result = parse_env_file(path)
        assert result == {"API_KEY": "abc123", "MODEL_ID": "opus"}
    finally:
        path.unlink()


def test_empty_file():
    path = _tmpfile("")
    try:
        result = parse_env_file(path)
        assert result == {}
    finally:
        path.unlink()


def test_single_key_no_value():
    path = _tmpfile("EMPTY=\n")
    try:
        result = parse_env_file(path)
        assert result == {"EMPTY": ""}
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# 2. Quoted values (single and double)
# ---------------------------------------------------------------------------


def test_double_quoted_value():
    path = _tmpfile('KEY="hello world"\n')
    try:
        result = parse_env_file(path)
        assert result == {"KEY": "hello world"}
    finally:
        path.unlink()


def test_single_quoted_value():
    path = _tmpfile("KEY='hello world'\n")
    try:
        result = parse_env_file(path)
        assert result == {"KEY": "hello world"}
    finally:
        path.unlink()


def test_escaped_quotes_in_double_quoted():
    path = _tmpfile('KEY="he said \\"hi\\""\n')
    try:
        result = parse_env_file(path)
        assert result == {"KEY": 'he said "hi"'}
    finally:
        path.unlink()


def test_escaped_backslash_in_double_quoted():
    path = _tmpfile('KEY="path\\\\to\\\\file"\n')
    try:
        result = parse_env_file(path)
        assert result == {"KEY": "path\\to\\file"}
    finally:
        path.unlink()


def test_unquoted_value_with_spaces():
    path = _tmpfile("KEY=hello world\n")
    try:
        result = parse_env_file(path)
        assert result == {"KEY": "hello world"}
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# 3. Comments and blank lines
# ---------------------------------------------------------------------------


def test_comments_and_blank_lines():
    path = _tmpfile(
        "# This is a comment\n"
        "\n"
        "KEY1=value1\n"
        "  # indented comment\n"
        "\n"
        "KEY2=value2\n"
    )
    try:
        result = parse_env_file(path)
        assert result == {"KEY1": "value1", "KEY2": "value2"}
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# 4. Duplicate key rejection
# ---------------------------------------------------------------------------


def test_duplicate_key_rejected():
    path = _tmpfile("KEY=value1\nKEY=value2\n")
    try:
        with pytest.raises(EnvParseError, match="duplicate"):
            parse_env_file(path)
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# 5. Invalid variable name rejection
# ---------------------------------------------------------------------------


def test_invalid_variable_name_numbers():
    path = _tmpfile("1INVALID=value\n")
    try:
        with pytest.raises(EnvParseError, match="invalid variable name"):
            parse_env_file(path)
    finally:
        path.unlink()


def test_invalid_variable_name_hyphen():
    path = _tmpfile("MY-KEY=value\n")
    try:
        with pytest.raises(EnvParseError, match="invalid variable name"):
            parse_env_file(path)
    finally:
        path.unlink()


def test_invalid_variable_name_space():
    path = _tmpfile("MY KEY=value\n")
    try:
        with pytest.raises(EnvParseError, match="invalid variable name"):
            parse_env_file(path)
    finally:
        path.unlink()


def test_valid_variable_name_with_underscore():
    path = _tmpfile("MY_API_KEY=value\n")
    try:
        result = parse_env_file(path)
        assert result == {"MY_API_KEY": "value"}
    finally:
        path.unlink()


def test_valid_variable_name_leading_underscore():
    path = _tmpfile("_PRIVATE=value\n")
    try:
        result = parse_env_file(path)
        assert result == {"_PRIVATE": "value"}
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# 6. Null byte rejection
# ---------------------------------------------------------------------------


def test_null_byte_rejected():
    path = _tmpfile_raw(b"KEY=value\x00more\n")
    try:
        with pytest.raises(EnvParseError, match="null byte"):
            parse_env_file(path)
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# 7. Allowed keys filtering
# ---------------------------------------------------------------------------


def test_allowed_keys_filtering_keeps_approved():
    path = _tmpfile("ANTHROPIC_API_KEY=sk-123\nLONGCAT_API_KEY=lc-456\n")
    try:
        allowed = {"ANTHROPIC_API_KEY", "LONGCAT_API_KEY"}
        result = parse_env_file(path, allowed_keys=allowed)
        assert result == {
            "ANTHROPIC_API_KEY": "sk-123",
            "LONGCAT_API_KEY": "lc-456",
        }
    finally:
        path.unlink()


def test_allowed_keys_filtering_rejects_unapproved():
    path = _tmpfile("ANTHROPIC_API_KEY=sk-123\nUNSAFE_SECRET=hacked\n")
    try:
        allowed = {"ANTHROPIC_API_KEY"}
        with pytest.raises(EnvParseError, match="unapproved"):
            parse_env_file(path, allowed_keys=allowed)
    finally:
        path.unlink()


def test_allowed_keys_none_returns_all():
    path = _tmpfile("KEY1=val1\nKEY2=val2\n")
    try:
        result = parse_env_file(path, allowed_keys=None)
        assert result == {"KEY1": "val1", "KEY2": "val2"}
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# 8. File ownership validation (mock for tests)
# ---------------------------------------------------------------------------


def test_symlink_refused(tmp_path: Path) -> None:
    """Symlinks must be refused even if they point to a valid file."""
    target = tmp_path / "real.env"
    _write(target, "KEY=value\n")
    link = tmp_path / "link.env"
    link.symlink_to(target)
    with pytest.raises(EnvParseError, match="symlink"):
        parse_env_file(link)


def test_non_regular_file_refused(tmp_path: Path) -> None:
    """Directories must be rejected, not treated as regular files."""
    target = tmp_path / "dir.env"
    target.mkdir()
    with pytest.raises(EnvParseError, match="not a regular file"):
        parse_env_file(target)


def test_excessive_permissions_refused(tmp_path: Path) -> None:
    """Files with group or other write permissions must be rejected."""
    target = tmp_path / "bad.env"
    target.write_text("KEY=value\n")
    target.chmod(0o666)  # group write
    with pytest.raises(EnvParseError, match="excessive permissions"):
        parse_env_file(target)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EnvParseError, match="Cannot stat"):
        parse_env_file(tmp_path / "nonexistent.env")


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


def test_value_with_equals_sign():
    path = _tmpfile("KEY=a=b=c\n")
    try:
        result = parse_env_file(path)
        assert result == {"KEY": "a=b=c"}
    finally:
        path.unlink()


def test_whitespace_around_key():
    path = _tmpfile("  KEY  =value\n")
    try:
        result = parse_env_file(path)
        assert result == {"KEY": "value"}
    finally:
        path.unlink()


def test_line_missing_equals():
    """A line without '=' that isn't a comment or blank should fail."""
    path = _tmpfile("JUSTAKEY\n")
    try:
        with pytest.raises(EnvParseError, match="expected KEY=VALUE"):
            parse_env_file(path)
    finally:
        path.unlink()


def test_unicode_value():
    path = _tmpfile("KEY=\u043f\u0440\u0438\u0432\u0435\u0442\n")
    try:
        result = parse_env_file(path)
        assert result == {"KEY": "\u043f\u0440\u0438\u0432\u0435\u0442"}
    finally:
        path.unlink()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
