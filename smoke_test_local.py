"""本地冒烟测试：扫描 testdata 并匹配 PersonA（不上传、仅本地）。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from facesort.config import db_path, load_config
from facesort.detector import FaceDetector
from facesort.indexer import Indexer
from facesort.matcher import find_matches
from facesort.pipeline import scan_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    album = ROOT / "testdata" / "album"
    ref = ROOT / "testdata" / "ref" / "PersonA.jpg"
    if not album.is_dir() or not ref.is_file():
        raise SystemExit("缺少 testdata，请先准备测试图片")

    cfg = load_config()
    # 使用独立测试库，避免污染正式索引
    test_db = ROOT / "data" / "smoke_test.db"
    if test_db.exists():
        test_db.unlink()

    indexer = Indexer(test_db)
    detector = FaceDetector(
        model_name=cfg["model_name"],
        use_gpu=cfg["use_gpu"],
        det_size=cfg["det_size"],
        min_face_size=cfg["min_face_size"],
    )
    print(f"设备: {detector.device_label}")

    result = scan_directory(album, indexer, detector, cfg["image_extensions"], incremental=False)
    print("扫描结果:", result)

    import cv2

    img = cv2.imread(str(ref))
    faces = detector.detect_from_array(img)
    reference = detector.reference_embedding(faces)
    if reference is None:
        raise SystemExit("参考图未检测到人脸")

    matches = find_matches(indexer, reference, threshold=0.45, person_name="PersonA")
    print(f"匹配数量: {len(matches)}")
    for m in matches:
        print(f"  {Path(m['path']).name}  similarity={m['similarity']:.3f}")

    if len(matches) < 1:
        raise SystemExit("匹配失败：期望至少命中 PersonA 相关照片")
    print("冒烟测试通过")


if __name__ == "__main__":
    main()
