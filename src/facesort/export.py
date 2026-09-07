"""将匹配结果导出到本地文件夹。"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from facesort.tags import folder_person_tag, safe_name


def export_photos(
    matches: list[dict],
    export_root: Path,
    person_name: str,
    mode: str = "copy",
) -> dict[str, int | str]:
    """
    按「文件夹_人名」标签导出：
    output/源文件夹_人名/原文件名
    """
    export_root.mkdir(parents=True, exist_ok=True)
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        tag = m.get("person_name") or folder_person_tag(m["path"], person_name)
        by_tag[safe_name(tag)].append(m)

    exported = 0
    skipped = 0
    dest_dirs: list[str] = []

    for tag, items in by_tag.items():
        dest_dir = export_root / tag
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_dirs.append(str(dest_dir))
        for m in items:
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

    summary = dest_dirs[0] if len(dest_dirs) == 1 else f"{export_root}（{len(dest_dirs)} 个子文件夹）"
    return {
        "exported": exported,
        "skipped": skipped,
        "dest": summary,
        "folders": len(dest_dirs),
    }
