"""人脸相似度匹配。"""

from __future__ import annotations

from typing import Any

import numpy as np

from facesort.indexer import Indexer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个 L2 归一化向量的余弦相似度。"""
    return float(np.dot(a, b))


def find_matches(
    indexer: Indexer,
    reference: np.ndarray,
    threshold: float,
    person_name: str,
) -> list[dict[str, Any]]:
    """
    在索引库中查找与参考 embedding 相似的人脸。
    同一张照片只保留最高相似度的一条记录。
    """
    best_by_photo: dict[int, dict[str, Any]] = {}

    for face_id, photo_id, embedding, photo_path in indexer.iter_all_faces():
        sim = cosine_similarity(reference, embedding)
        if sim < threshold:
            continue
        existing = best_by_photo.get(photo_id)
        if existing is None or sim > existing["similarity"]:
            best_by_photo[photo_id] = {
                "face_id": face_id,
                "photo_id": photo_id,
                "path": photo_path,
                "similarity": sim,
                "person_name": person_name,
            }

    matches = list(best_by_photo.values())
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches
