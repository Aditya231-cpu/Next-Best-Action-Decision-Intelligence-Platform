"""Small YAML-backed settings loader for reproducible local paths and seeds."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class ProjectSettings:
    root: Path
    random_seed: int
    raw_data: Path
    processed_data: Path
    reports: Path
    artifacts: Path

def load_settings(root: Path = Path.cwd()) -> ProjectSettings:
    """Load project-relative locations from configs/data.yaml."""
    config_path = root / "configs" / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ProjectSettings(root=root, random_seed=int(config.get("random_seed", 0)),
        raw_data=root / config["raw_path"], processed_data=root / config["processed_path"],
        reports=root / config.get("report_dir", "reports/generated"), artifacts=root / "artifacts")
