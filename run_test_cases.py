"""自动跑本地测试用例（读 testdata/cases/expected.json，不上传）。"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from facesort.config import load_config
from facesort.detector import FaceDetector
from facesort.indexer import Indexer
from facesort.matcher import find_matches
from facesort.pipeline import scan_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def case_pass(case: dict, matches: list[dict]) -> tuple[bool, str]:
    names = [Path(m["path"]).name for m in matches]
    joined = " | ".join(names[:20])

    min_matches = case.get("min_matches")
    max_matches = case.get("max_matches")
    subs = case.get("must_match_substrings") or []

    if min_matches is not None and len(matches) < min_matches:
        return False, f"匹配数 {len(matches)} < 期望至少 {min_matches}; 样本: {joined}"

    if max_matches is not None and len(matches) > max_matches:
        return False, f"匹配数 {len(matches)} > 期望最多 {max_matches}; 样本: {joined}"

    if subs:
        hit_subs = [sub for sub in subs if any(sub in n for n in names)]
        required = int(case.get("min_substring_hits", len(subs)))
        if len(hit_subs) < required:
            return (
                False,
                f"关键词命中 {len(hit_subs)}/{required} {hit_subs}; 样本: {joined}",
            )

    return True, f"OK matches={len(matches)} | {joined}"


def main() -> int:
    expected_path = ROOT / "testdata" / "cases" / "expected.json"
    if not expected_path.exists():
        print("缺少 expected.json，请先运行: python prepare_test_cases.py")
        return 2

    data = json.loads(expected_path.read_text(encoding="utf-8"))
    album = Path(data["album_root"])
    cfg = load_config()

    db = ROOT / "data" / "test_cases.db"
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
    print(f"设备: {detector.device_label}")
    print("扫描相册…")
    result = scan_directory(
        album,
        indexer,
        detector,
        cfg["image_extensions"],
        incremental=False,
        progress_callback=lambda c, t, m: print(f"\r{c}/{t} {m[:60]}", end="") if c % 10 == 0 or c == t else None,
    )
    print()
    print("扫描结果:", result)

    passed = 0
    failed = 0
    for case in data["auto_cases"]:
        ref = Path(case["reference"])
        if not ref.exists():
            print(f"[SKIP] {case['id']} 缺少参考脸 {ref}")
            failed += 1
            continue

        import cv2

        img = cv2.imread(str(ref))
        faces = detector.detect_from_array(img) if img is not None else []
        reference = detector.reference_embedding(faces)
        if reference is None:
            print(f"[FAIL] {case['id']} 参考脸未检测到人脸")
            failed += 1
            continue

        matches = find_matches(
            indexer,
            reference,
            float(case["threshold"]),
            case["person_name"],
        )
        ok, detail = case_pass(case, matches)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}: {detail}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 50)
    print(f"通过 {passed} / 失败 {failed} / 合计 {passed + failed}")
    print(f"手工步骤见: {ROOT / 'testdata' / 'cases' / 'MANUAL_TEST_GUIDE.txt'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
