"""InsightFace 人脸检测与特征提取（100% 本地运行）。"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from facesort.gpu import resolve_execution_backend

logger = logging.getLogger(__name__)


class FaceDetector:
    """封装 InsightFace，提供检测与 embedding 提取。"""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        use_gpu: bool = True,
        det_size: int = 640,
        min_face_size: int = 40,
    ) -> None:
        backend = resolve_execution_backend(use_gpu)
        self.device_label: str = backend["device_label"]
        self.using_gpu: bool = backend["using_gpu"]

        self.app = FaceAnalysis(name=model_name, providers=backend["providers"])
        self.app.prepare(ctx_id=backend["ctx_id"], det_size=(det_size, det_size))
        self.min_face_size = min_face_size

        active = self._active_providers()
        logger.info(
            "FaceDetector 已加载模型 %s | 模式=%s | ONNX=%s",
            model_name,
            self.device_label,
            active,
        )

    def _active_providers(self) -> list[str]:
        """读取 InsightFace 内部 session 实际使用的 provider。"""
        providers: set[str] = set()
        for model in self.app.models.values():
            session = getattr(model, "session", None)
            if session is not None:
                providers.update(session.get_providers())
        return sorted(providers)

    def detect(self, image_path: str) -> list[dict[str, Any]]:
        """检测单张图片中的所有人脸，返回 bbox + embedding。"""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("无法读取图片: %s", image_path)
            return []

        faces = self.app.get(img)
        results: list[dict[str, Any]] = []
        for face in faces:
            bbox = face.bbox.astype(float).tolist()
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if min(w, h) < self.min_face_size:
                continue
            embedding = face.normed_embedding
            if embedding is None:
                continue
            results.append(
                {
                    "bbox": bbox,
                    "embedding": np.asarray(embedding, dtype=np.float32),
                    "det_score": float(getattr(face, "det_score", 0.0)),
                }
            )
        return results

    def detect_from_array(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """从 numpy 数组（参考脸上传）检测人脸。"""
        faces = self.app.get(image_bgr)
        results: list[dict[str, Any]] = []
        for face in faces:
            embedding = face.normed_embedding
            if embedding is None:
                continue
            results.append(
                {
                    "bbox": face.bbox.astype(float).tolist(),
                    "embedding": np.asarray(embedding, dtype=np.float32),
                    "det_score": float(getattr(face, "det_score", 0.0)),
                }
            )
        return results

    @staticmethod
    def reference_embedding(faces: list[dict[str, Any]]) -> np.ndarray | None:
        """多张参考脸取 embedding 均值（L2 归一化）。"""
        if not faces:
            return None
        stack = np.stack([f["embedding"] for f in faces], axis=0)
        mean = stack.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-6:
            return None
        return (mean / norm).astype(np.float32)
