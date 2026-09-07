"""扫描编排：遍历目录、检测人脸、写入索引。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from facesort.detector import FaceDetector
from facesort.image_io import imread_bgr, short_filename
from facesort.indexer import Indexer
from facesort.scanner import file_fingerprint, iter_image_files

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int, str], None]


def scan_directory(
    root: str | Path,
    indexer: Indexer,
    detector: FaceDetector,
    extensions: list[str],
    *,
    incremental: bool = True,
    progress_callback: ProgressCb | None = None,
) -> dict[str, int]:
    """
    扫描目录并将人脸写入索引。
    incremental=True 时跳过 hash 未变的已索引文件。
    progress_callback(current, total, message)
    """
    root = Path(root).expanduser().resolve()
    files = list(iter_image_files(root, extensions))
    total = len(files)
    scanned = 0
    skipped = 0
    faces_total = 0
    errors = 0
    retained_faces = 0
    unread = 0
    no_face = 0

    for index, path in enumerate(files, start=1):
        path_str = str(path)
        short = short_filename(path.name, 22)
        try:
            file_hash, mtime = file_fingerprint(path)
            if incremental:
                existing = indexer.get_photo_by_path(path_str)
                if existing and existing["file_hash"] == file_hash:
                    skipped += 1
                    retained_faces += int(existing["face_count"] or 0)
                    if progress_callback:
                        progress_callback(
                            index,
                            total,
                            f"跳过 {short} 保留{int(existing['face_count'] or 0)}脸",
                        )
                    continue

            # 先探测能否读取（中文路径等）
            probe = imread_bgr(path_str)
            if probe is None:
                unread += 1
                errors += 1
                if progress_callback:
                    progress_callback(index, total, f"无法读取 {short}")
                continue

            faces = detector.detect(path_str, image_bgr=probe)
            photo_id = indexer.upsert_photo(path_str, file_hash, mtime, len(faces))
            indexer.replace_faces(photo_id, faces)
            scanned += 1
            faces_total += len(faces)
            if not faces:
                no_face += 1

            if progress_callback:
                progress_callback(index, total, f"检测 {short} → {len(faces)}脸")

        except Exception as exc:  # noqa: BLE001 — 单张失败不中断整批
            errors += 1
            logger.exception("处理失败 %s: %s", path_str, exc)
            if progress_callback:
                progress_callback(index, total, f"失败 {short}")

    return {
        "total_files": total,
        "scanned": scanned,
        "skipped": skipped,
        "faces": faces_total,
        "retained_faces": retained_faces,
        "errors": errors,
        "unread": unread,
        "no_face": no_face,
    }
