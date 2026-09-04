"""扫描编排：遍历目录、检测人脸、写入索引。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from facesort.detector import FaceDetector
from facesort.indexer import Indexer
from facesort.scanner import file_fingerprint, iter_image_files

logger = logging.getLogger(__name__)


def scan_directory(
    root: str | Path,
    indexer: Indexer,
    detector: FaceDetector,
    extensions: list[str],
    *,
    incremental: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """
    扫描目录并将人脸写入索引。
    incremental=True 时跳过 hash 未变的已索引文件。
    """
    root = Path(root).expanduser().resolve()
    files = list(iter_image_files(root, extensions))
    scanned = 0
    skipped = 0
    faces_total = 0
    errors = 0

    iterator: tqdm | list[Path]
    iterator = tqdm(files, desc="扫描照片", unit="张")

    for path in iterator:
        path_str = str(path)
        try:
            file_hash, mtime = file_fingerprint(path)
            if incremental:
                existing = indexer.get_photo_by_path(path_str)
                if existing and existing["file_hash"] == file_hash:
                    skipped += 1
                    continue

            faces = detector.detect(path_str)
            photo_id = indexer.upsert_photo(path_str, file_hash, mtime, len(faces))
            indexer.replace_faces(photo_id, faces)
            scanned += 1
            faces_total += len(faces)

            if progress_callback:
                progress_callback(f"已处理: {path.name} ({len(faces)} 张人脸)")

        except Exception as exc:  # noqa: BLE001 — 单张失败不中断整批
            errors += 1
            logger.exception("处理失败 %s: %s", path_str, exc)

    return {
        "total_files": len(files),
        "scanned": scanned,
        "skipped": skipped,
        "faces": faces_total,
        "errors": errors,
    }
