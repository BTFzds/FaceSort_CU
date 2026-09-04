"""将匹配结果导出到本地文件夹。"""

from __future__ import annotations

import shutil
from pathlib import Path


def export_photos(
    matches: list[dict],
    export_root: Path,
    person_name: str,
    mode: str = "copy",
) -> dict[str, int]:
    """
    导出照片到 export_root/person_name/。
    mode: copy | hardlink
    """
    dest_dir = export_root / _safe_name(person_name)
    dest_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    for m in matches:
        src = Path(m["path"])
        if not src.exists():
            skipped += 1
            continue
        dest = dest_dir / src.name
        if dest.exists():
            dest = dest_dir / f"{src.stem}_{exported}{src.suffix}"

        if mode == "hardlink":
            try:
                dest.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dest)
        else:
            shutil.copy2(src, dest)
        exported += 1

    return {"exported": exported, "skipped": skipped, "dest": str(dest_dir)}


def _safe_name(name: str) -> str:
    """去除路径非法字符。"""
    forbidden = '<>:"/\\|?*'
    cleaned = "".join(c if c not in forbidden else "_" for c in name.strip())
    return cleaned or "未命名"
