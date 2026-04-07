#!/usr/bin/env python3
"""Thin script entry point for checkpoint sampling."""

from __future__ import annotations

from ndae.cli.sample import run_sample_cli


if __name__ == "__main__":
    raise SystemExit(run_sample_cli())
