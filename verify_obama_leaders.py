"""验证：奥巴马查询图不在相册内，且仍能从相册中找回奥巴马。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2

from facesort.config import load_config
from facesort.detector import FaceDetector
from facesort.indexer import Indexer
from facesort.matcher import find_matches
from facesort.pipeline import scan_directory
from facesort.tags import apply_folder_person_tags


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    manifest_path = ROOT / "testdata" / "leaders" / "manifest.json"
    if not manifest_path.exists():
        print("请先运行: python prepare_leaders_dataset.py")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    album = Path(manifest["album_root"])
    query = Path(manifest["obama_query"])

    assert query.exists(), "查询图不存在"
    assert album.exists(), "相册不存在"
    assert not query.is_relative_to(album), "查询图落在相册内！"

    qhash = md5_file(query)
    collisions = [p for p in album.rglob("*.*") if p.is_file() and md5_file(p) == qhash]
    if collisions:
        print("FAIL: 查询图内容与相册重复:", collisions)
        return 1
    print(f"隔离通过: 查询图 md5={qhash[:16]}… 未出现在相册 {sum(1 for _ in album.rglob('*.*'))} 个文件中")

    cfg = load_config()
    db = ROOT / "data" / "leaders_obama_test.db"
    if db.exists():
        db.unlink()

    indexer = Indexer(db)
    detector = FaceDetector(
        model_name=cfg["model_name"],
        use_gpu=cfg["use_gpu"],
        det_size=int(cfg["det_size"]),
        min_face_size=int(cfg["min_face_size"]),
        det_thresh=float(cfg.get("det_thresh", 0.35)),
    )
    print("扫描 leaders/album …", detector.device_label)
    result = scan_directory(
        album,
        indexer,
        detector,
        cfg["image_extensions"],
        incremental=False,
    )
    print("扫描:", result)

    img = cv2.imread(str(query))
    faces = detector.detect_from_array(img)
    reference = detector.reference_embedding(faces)
    if reference is None:
        print("FAIL: 查询图未检测到人脸")
        return 1

    matches = find_matches(indexer, reference, 0.40, "Obama")
    matches = apply_folder_person_tags(matches, "Obama")
    names = [Path(m["path"]).name for m in matches]
    print(f"命中 {len(matches)} 张:")
    for m in matches[:20]:
        print(f"  {m['person_name']}  {Path(m['path']).name}  sim={m['similarity']:.3f}")

    obama_hits = [n for n in names if "obama" in n.lower()]
    if len(obama_hits) < 1:
        print("FAIL: 未能从相册中找回奥巴马（查询图未参与检索）")
        return 1

    # 确认命中路径都不等于查询图
    if any(Path(m["path"]).resolve() == query.resolve() for m in matches):
        print("FAIL: 命中结果包含查询图本身")
        return 1

    print("PASS: 查询图不在数据集中，且成功找回相册内奥巴马照片")
    print(f"建议标注示例: {matches[0]['person_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
