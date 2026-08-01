#!/usr/bin/env python3
"""Compatibility wrapper for the documented sync_models.py entry point."""

from fi_litellm.cli import sync


if __name__ == "__main__":
    sync()
