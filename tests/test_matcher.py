"""matcher 模块单元测试。"""

import numpy as np

from facesort.matcher import cosine_similarity


def test_cosine_similarity_identical() -> None:
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(cosine_similarity(a, b)) < 1e-6
