"""Tests for hooks/_shared.py."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "hooks"))

from _shared import fail_open_main  # noqa: E402


def test_fail_open_main_returns_the_wrapped_return_value():
    assert fail_open_main(lambda: 0) == 0


def test_fail_open_main_swallows_an_unhandled_exception_and_returns_zero(capsys):
    def _boom() -> int:
        raise RuntimeError("boom")

    assert fail_open_main(_boom) == 0
    captured = capsys.readouterr()
    assert "boom" in captured.err
