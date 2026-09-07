"""下载公开测试素材到 testdata/（仅本地，已 gitignore，勿上传）。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALBUM = ROOT / "testdata" / "album"
REF = ROOT / "testdata" / "ref"
GROUP = ROOT / "testdata" / "album" / "groups"
PORTRAIT = ROOT / "testdata" / "album" / "portraits"


def download(url: str, dest: Path, timeout: int = 40) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FaceSortLocalTest/0.1 (local offline testing)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 800:
            return False
        dest.write_bytes(data)
        print(f"OK {dest.name} ({len(data)} bytes)")
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"FAIL {dest.name}: {exc}")
        return False


def fetch_wikimedia_group(limit: int = 80) -> int:
    """从 Wikimedia Commons 拉取合照类图片。"""
    query = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": "group photograph OR group portrait OR team photo",
        "gsrnamespace": "6",
        "gsrlimit": str(min(limit, 50)),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "1024",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(query)
    req = urllib.request.Request(url, headers={"User-Agent": "FaceSortLocalTest/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Wikimedia 查询失败: {exc}")
        return 0

    pages = payload.get("query", {}).get("pages", {})
    count = 0
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        img_url = info.get("thumburl") or info.get("url")
        if not img_url:
            continue
        ext = Path(urllib.parse.urlparse(img_url).path).suffix.lower() or ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        dest = GROUP / f"group_{page.get('pageid', count)}{ext}"
        if download(img_url, dest):
            count += 1
        time.sleep(0.15)
    return count


def fetch_portraits(count: int = 60) -> int:
    """公开头像素材（随机用户肖像）。"""
    ok = 0
    for i in range(count):
        gender = "women" if i % 2 == 0 else "men"
        idx = (i * 3) % 99
        url = f"https://randomuser.me/api/portraits/{gender}/{idx}.jpg"
        dest = PORTRAIT / f"portrait_{gender}_{idx:02d}_{i:03d}.jpg"
        if download(url, dest):
            ok += 1
        time.sleep(0.05)
    return ok


def fetch_picsum_peopleish(count: int = 40) -> int:
    """备用大图素材（风景/人物混合，用于压测扫描进度）。"""
    ok = 0
    for i in range(count):
        # seed 固定，便于复现
        url = f"https://picsum.photos/seed/facesort{i}/800/600.jpg"
        dest = ALBUM / f"misc_{i:03d}.jpg"
        if download(url, dest):
            ok += 1
        time.sleep(0.08)
    return ok


def ensure_reference() -> None:
    REF.mkdir(parents=True, exist_ok=True)
    src_candidates = sorted(PORTRAIT.glob("portrait_women_*.jpg"))
    if not src_candidates:
        src_candidates = sorted(ALBUM.glob("personA_*.jpg"))
    if src_candidates:
        target = REF / "PersonA.jpg"
        if not target.exists():
            target.write_bytes(src_candidates[0].read_bytes())
            print(f"参考脸 -> {target}")


def main() -> None:
    ALBUM.mkdir(parents=True, exist_ok=True)
    GROUP.mkdir(parents=True, exist_ok=True)
    PORTRAIT.mkdir(parents=True, exist_ok=True)

    print("下载肖像…")
    p = fetch_portraits(70)
    print("下载合照…")
    g = fetch_wikimedia_group(80)
    print("下载补充大图…")
    m = fetch_picsum_peopleish(40)
    ensure_reference()

    total = len(list(ALBUM.rglob("*.*")))
    print("=" * 40)
    print(f"肖像成功 {p}｜合照成功 {g}｜补充 {m}")
    print(f"testdata/album 下共 {total} 个文件（本地，不上传）")
    print(f"参考脸: {REF / 'PersonA.jpg'}")


if __name__ == "__main__":
    main()
