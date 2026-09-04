"""加载与管理项目配置。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """从 YAML 文件加载配置，缺失项使用内置默认值。"""
    path = config_path or DEFAULT_CONFIG_PATH
    defaults: dict[str, Any] = {
        "match_threshold": 0.55,
        "min_face_size": 40,
        "det_size": 640,
        "model_name": "buffalo_l",
        "use_gpu": True,
        "scan_workers": 0,
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
        "database_path": "data/facesort.db",
        "export_dir": "output",
        "server_port": 7860,
    }
    if path.exists():
        with path.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    return defaults


def db_path(config: dict[str, Any] | None = None) -> Path:
    """返回 SQLite 数据库的绝对路径。"""
    cfg = config or load_config()
    return PROJECT_ROOT / cfg["database_path"]


def export_path(config: dict[str, Any] | None = None) -> Path:
    """返回默认导出目录的绝对路径。"""
    cfg = config or load_config()
    return PROJECT_ROOT / cfg["export_dir"]
