"""Configuration loader and canonical repository paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "targets.yaml"
SEED_PARQUET_PATH = REPO_ROOT / "data" / "processed" / "darkstore_14d_seed.parquet"
LIVE_PARQUET_PATH = REPO_ROOT / "data" / "raw" / "live_observations.parquet"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def load_targets_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load target pincodes and everyday items from YAML configuration."""
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_active_dataset(prefer_live: bool = False) -> Path:
    """Return live collected Parquet if available and requested, else 14-day seed Parquet."""
    if prefer_live and LIVE_PARQUET_PATH.exists():
        return LIVE_PARQUET_PATH
    return SEED_PARQUET_PATH
