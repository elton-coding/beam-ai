"""Shared filesystem locations for the direct_network package."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPO_ROOT = ROOT.parent
REPORTS = REPO_ROOT / "docs" / "reports"
