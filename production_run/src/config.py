from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AppConfig:
    raw: Dict[str, Any]

    @property
    def frequency_minutes(self) -> int:
        return int(self.raw.get("project", {}).get("frequency_minutes", 1))

    @property
    def output_dir(self) -> Path:
        base = self.raw.get("output", {}).get("base_dir", "outputs")
        return Path(base)


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(raw=data)
