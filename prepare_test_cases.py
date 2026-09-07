"""
准备本地测试用例素材（仅写入 testdata/，已 gitignore，勿上传）。

场景：
1. 同人多图（公开示例人物）
2. 合照合成（把目标脸贴进多人背景，验证合照检索）
3. 干扰项 / 负例（相册中不存在的人）
4. 缩放与压缩变体
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent
ALBUM = ROOT / "testdata" / "album"
REF = ROOT / "testdata" / "ref"
CASES = ROOT / "testdata" / "cases"
IDENTITY = ALBUM / "case_identity"
GROUPS = ALBUM / "case_groups"
DISTRACT = ALBUM / "case_distract"
VARIANT = ALBUM / "case_variants"

UA = {"User-Agent": "FaceSortLocalTest/0.2 (offline QA only)"}

# 公开示例图（多人脸识别教程常用，非内部资料）
PUBLIC_URLS = {
    "obama_01.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
    "obama_02.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
    "obama_03.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama_small.jpg",
    "biden_01.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
    "two_people.jpg": "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/two_people.jpg",
}


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 2000:
        return True
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
        if len(data) < 1500:
            return False
        dest.write_bytes(data)
        print(f"OK {dest.relative_to(ROOT)}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {dest.name}: {exc}")
        return False


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def paste_face_on_background(
    bg: Image.Image,
    face: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """把人脸缩放到 box 并贴到背景上，模拟合照中的小人脸。"""
    out = bg.copy()
    x, y, w, h = box
    face_r = face.resize((w, h), Image.Resampling.LANCZOS)
    # 轻微羽化边缘
    mask = Image.new("L", (w, h), 255)
    out.paste(face_r, (x, y), mask)
    return out


def make_group_composites(face_path: Path, tag: str, count: int = 8) -> list[str]:
    """基于现有合照背景 + 目标脸，生成带 ground-truth 的合成合照。"""
    backgrounds = sorted((ALBUM / "groups").glob("*.*"))
    if not backgrounds:
        backgrounds = sorted(IDENTITY.glob("*.jpg"))[:3]
    if not backgrounds:
        return []

    face = load_rgb(face_path)
    # 裁中心区域当脸（示例图多为半身/头像）
    fw, fh = face.size
    crop = face.crop((int(fw * 0.2), int(fh * 0.05), int(fw * 0.8), int(fh * 0.75)))

    names: list[str] = []
    rng = random.Random(42 + hash(tag) % 1000)
    for i in range(count):
        bg = load_rgb(backgrounds[i % len(backgrounds)])
        bw, bh = bg.size
        # 小人脸尺寸：背景宽度的 6%~14%
        face_w = max(40, int(bw * rng.uniform(0.06, 0.14)))
        face_h = int(face_w * 1.2)
        x = rng.randint(10, max(11, bw - face_w - 10))
        y = rng.randint(10, max(11, bh - face_h - 10))
        composed = paste_face_on_background(bg, crop, (x, y, face_w, face_h))
        # 部分加模糊/降亮度，模拟难例
        if i % 3 == 1:
            composed = composed.filter(ImageFilter.GaussianBlur(radius=0.8))
        if i % 3 == 2:
            composed = ImageEnhance.Brightness(composed).enhance(0.85)

        out = GROUPS / f"synth_{tag}_{i:02d}.jpg"
        composed.save(out, quality=88)
        names.append(out.name)
        print(f"合成合照 {out.name}")
    return names


def make_variants(src: Path, prefix: str) -> list[str]:
    """同图缩放/压缩变体，应仍能匹配。"""
    img = load_rgb(src)
    names: list[str] = []
    specs = [
        ("small", (180, 180), 70),
        ("medium", (480, 480), 85),
        ("large", (900, 900), 92),
    ]
    for label, size, quality in specs:
        v = img.copy()
        v.thumbnail(size, Image.Resampling.LANCZOS)
        out = VARIANT / f"{prefix}_{label}.jpg"
        v.save(out, quality=quality)
        names.append(out.name)
    return names


def ensure_distract(n: int = 30) -> None:
    """干扰肖像：已有 portraits 则复制一批到 distract。"""
    DISTRACT.mkdir(parents=True, exist_ok=True)
    portraits = sorted((ALBUM / "portraits").glob("*.jpg"))
    if not portraits:
        return
    for i, src in enumerate(portraits[:n]):
        dest = DISTRACT / f"distract_{i:03d}.jpg"
        if not dest.exists():
            dest.write_bytes(src.read_bytes())


def write_expected(payload: dict) -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    path = CASES / "expected.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {path}")


def write_guide(payload: dict) -> None:
    lines = [
        "FaceSort 本地手工测试指南（素材仅在本机，勿 git add）",
        "=" * 50,
        "",
        "相册目录统一选：",
        f"  {ALBUM}",
        "",
        "建议：取消勾选「增量扫描」后全量扫描一次。",
        "",
    ]
    for case in payload["manual_cases"]:
        lines.append(f"【{case['id']}】{case['title']}")
        lines.append(f"  参考脸: {case['reference']}")
        lines.append(f"  标签名: {case['person_name']}")
        lines.append(f"  建议阈值: {case['threshold']}")
        lines.append(f"  期望: {case['expect']}")
        lines.append(f"  说明: {case['note']}")
        lines.append("")
    guide = CASES / "MANUAL_TEST_GUIDE.txt"
    guide.write_text("\n".join(lines), encoding="utf-8")
    print(f"写入 {guide}")


def main() -> None:
    for d in (IDENTITY, GROUPS, DISTRACT, VARIANT, REF, CASES):
        d.mkdir(parents=True, exist_ok=True)

    print("1) 下载公开同人多图…")
    for name, url in PUBLIC_URLS.items():
        download(url, IDENTITY / name)

    obama_ref = REF / "PersonObama.jpg"
    biden_ref = REF / "PersonBiden.jpg"
    if (IDENTITY / "obama_01.jpg").exists():
        obama_ref.write_bytes((IDENTITY / "obama_01.jpg").read_bytes())
    if (IDENTITY / "biden_01.jpg").exists():
        biden_ref.write_bytes((IDENTITY / "biden_01.jpg").read_bytes())

    # PersonA：沿用已有参考；若不存在则用肖像
    person_a_ref = REF / "PersonA.jpg"
    if not person_a_ref.exists():
        portraits = sorted((ALBUM / "portraits").glob("portrait_women_*.jpg"))
        if portraits:
            person_a_ref.write_bytes(portraits[0].read_bytes())

    # 负例参考：单独下载一张不放入 album 身份区的脸，只放 ref
    absent_ref = REF / "PersonAbsent.jpg"
    download(
        "https://randomuser.me/api/portraits/men/97.jpg",
        absent_ref,
    )
    # 确保 distract 里不要用 97 号同文件名干扰——用不同编号即可

    print("2) 生成合照合成用例…")
    synth_obama = []
    synth_a = []
    if obama_ref.exists():
        synth_obama = make_group_composites(obama_ref, "Obama", count=10)
    if person_a_ref.exists():
        synth_a = make_group_composites(person_a_ref, "PersonA", count=10)

    print("3) 生成缩放压缩变体…")
    variants = []
    if (IDENTITY / "obama_02.jpg").exists():
        variants = make_variants(IDENTITY / "obama_02.jpg", "obama")

    print("4) 准备干扰项…")
    ensure_distract(40)

    expected = {
        "album_root": str(ALBUM),
        "auto_cases": [
            {
                "id": "TC03_obama_variants",
                "person_name": "PersonObama",
                "reference": str(obama_ref),
                "threshold": 0.40,
                "must_match_substrings": ["obama_small", "obama_medium", "obama_large"],
                "min_substring_hits": 2,
                "min_matches": 2,
                "note": "缩放压缩变体应能匹配至少 2 种",
            },
            {
                "id": "TC04_biden",
                "person_name": "PersonBiden",
                "reference": str(biden_ref),
                "threshold": 0.40,
                "must_match_substrings": ["biden_", "two_people"],
                "min_substring_hits": 1,
                "min_matches": 1,
                "note": "Biden 单人/双人图",
            },
            {
                "id": "TC05_absent_negative",
                "person_name": "PersonAbsent",
                "reference": str(absent_ref),
                "threshold": 0.55,
                "must_match_substrings": [],
                "max_matches": 3,
                "note": "相册中不存在该人，高阈值下匹配应很少或为 0",
            },
            {
                "id": "TC06_personA_synth",
                "person_name": "PersonA",
                "reference": str(person_a_ref),
                "threshold": 0.40,
                "must_match_substrings": ["synth_PersonA_", "personA_"],
                "min_substring_hits": 1,
                "min_matches": 3,
                "note": "PersonA 原图 + 合成合照",
            },
            {
                "id": "TC01_obama_identity",
                "person_name": "PersonObama",
                "reference": str(obama_ref),
                "threshold": 0.40,
                "must_match_substrings": ["obama_"],
                "min_substring_hits": 1,
                "min_matches": 2,
                "note": "同人多图至少命中 2 张 obama_*",
            },
            {
                "id": "TC02_obama_group_synth",
                "person_name": "PersonObama",
                "reference": str(obama_ref),
                "threshold": 0.35,
                "must_match_substrings": ["synth_Obama_"],
                "min_substring_hits": 1,
                "min_matches": 5,
                "note": "合成合照中应能找回多数 Obama 小人脸",
            },
        ],
        "manual_cases": [
            {
                "id": "M1",
                "title": "同人多图召回",
                "reference": str(obama_ref),
                "person_name": "PersonObama",
                "threshold": "0.40~0.50",
                "expect": "至少出现 obama_01 / obama_02 等",
                "note": "验证基础识别是否正常",
            },
            {
                "id": "M2",
                "title": "合照找人（合成）",
                "reference": str(obama_ref),
                "person_name": "PersonObama",
                "threshold": "0.35~0.45",
                "expect": "右侧出现 synth_Obama_*.jpg 若干张",
                "note": "专测从小脸合照中找回目标",
            },
            {
                "id": "M3",
                "title": "双人图",
                "reference": str(biden_ref),
                "person_name": "PersonBiden",
                "threshold": "0.40",
                "expect": "命中 biden_01 或 two_people",
                "note": "一张图多人时仍应命中含目标的照片",
            },
            {
                "id": "M4",
                "title": "负例（不应乱配）",
                "reference": str(absent_ref),
                "person_name": "PersonAbsent",
                "threshold": "0.55~0.65",
                "expect": "匹配很少或为 0",
                "note": "阈值过高更严；若误报多可再提高阈值",
            },
            {
                "id": "M5",
                "title": "难例：模糊/偏暗合成图",
                "reference": str(obama_ref),
                "person_name": "PersonObama",
                "threshold": "0.35",
                "expect": "synth_Obama_ 中带模糊/降亮的也能部分命中",
                "note": "漏检属正常，观察召回是否明显差于清晰图",
            },
            {
                "id": "M6",
                "title": "你自己的合照",
                "reference": "自备 1 张清晰正脸（勿用工作照提交 Git）",
                "person_name": "PersonX",
                "threshold": "0.40~0.50",
                "expect": "在自有相册合照中找出该人",
                "note": "这是最终业务验收；素材勿入库",
            },
        ],
        "synth_files": {"obama": synth_obama, "personA": synth_a, "variants": variants},
    }
    write_expected(expected)
    write_guide(expected)
    total = len(list(ALBUM.rglob("*.*")))
    print("=" * 40)
    print(f"相册文件约 {total} 个（均在 testdata，不上传）")
    print("下一步: .venv\\Scripts\\python run_test_cases.py")


if __name__ == "__main__":
    main()
