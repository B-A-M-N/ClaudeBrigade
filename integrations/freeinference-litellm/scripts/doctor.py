#!/usr/bin/env python3
"""Compatibility wrapper for the documented doctor.py entry point."""

from fi_litellm.cli import doctor


if __name__ == "__main__":
    doctor()
