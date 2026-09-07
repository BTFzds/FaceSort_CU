"""跨编码安全读图（修复 Windows 中文路径下 cv2.imread 失败）。"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def imread_bgr(path: str | Path) -> np.ndarray | None:
    """
    读取 BGR 图像。优先用 np.fromfile + imdecode，兼容中文/特殊字符路径。
    """
    path = Path(path)
    if not path.is_file():
        logger.warning("文件不存在: %s", path)
        return None

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except OSError as exc:
        logger.warning("fromfile 失败 %s: %s", path, exc)

    # Pillow 回退（部分 webp / 特殊编码）
    try:
        with Image.open(path) as pil:
            rgb = pil.convert("RGB")
            arr = np.asarray(rgb)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except OSError as exc:
        logger.warning("无法读取图片 %s: %s", path, exc)
        return None


def short_filename(name: str, max_len: int = 24) -> str:
    """固定展示长度，避免 UI 因文件名长短抖动。"""
    name = name.replace("\n", " ")
    if len(name) <= max_len:
        return name.ljust(max_len)
    keep = max_len - 1
    left = keep // 2
    right = keep - left
    return name[:left] + "…" + name[-right:]
