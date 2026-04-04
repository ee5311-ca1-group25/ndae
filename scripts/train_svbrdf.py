#!/usr/bin/env python3
"""Thin script entry point for Lecture 1 dry-run execution."""

from __future__ import annotations

from ndae.cli.train import run_train_cli


if __name__ == "__main__":
    raise SystemExit(run_train_cli())
