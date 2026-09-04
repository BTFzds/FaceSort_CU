"""递归扫描本地文件夹，收集图片路径。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator


def normalize_extensions(extensions: list[str]) -> set[str]:
    """统一扩展名为小写带点形式。"""
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def iter_image_files(root: str | Path, extensions: list[str]) -> Iterator[Path]:
    """递归遍历目录，产出图片文件路径。"""
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"目录不存在: {root_path}")

    allowed = normalize_extensions(extensions)
    for dirpath, _, filenames in os.walk(root_path):
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in allowed:
                yield path


def file_fingerprint(path: Path) -> tuple[str, float]:
    """返回 (md5_hash, mtime) 用于增量扫描判断。"""
    stat = path.stat()
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest(), stat.st_mtime
