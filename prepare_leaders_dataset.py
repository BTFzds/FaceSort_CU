"""
准备「各国领导人」测试相册（仅本地 testdata/leaders，已 gitignore）。

规则：
- 相册全是子文件夹（按国家/地区）
- 专门用于查找 Obama
- 查询用参考图绝不放入相册数据集
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALBUM = ROOT / "testdata" / "leaders" / "album"
REF = ROOT / "testdata" / "leaders" / "ref"
META = ROOT / "testdata" / "leaders" / "manifest.json"

UA = {"User-Agent": "FaceSortLocalTest/0.3 (offline QA; leaders dataset)"}

# 国家子文件夹 -> Wikimedia 检索关键词（公开资料）
LEADER_QUERIES: dict[str, list[str]] = {
    "USA": [
        "Barack Obama official portrait",
        "Joe Biden official portrait",
        "Donald Trump official portrait",
        "George W Bush portrait",
        "Bill Clinton portrait",
    ],
    "France": [
        "Emmanuel Macron portrait",
        "Francois Hollande portrait",
        "Nicolas Sarkozy portrait",
    ],
    "Germany": [
        "Angela Merkel portrait",
        "Olaf Scholz portrait",
        "Frank-Walter Steinmeier portrait",
    ],
    "UK": [
        "Boris Johnson portrait",
        "Theresa May portrait",
        "Rishi Sunak portrait",
        "Keir Starmer portrait",
    ],
    "Canada": [
        "Justin Trudeau portrait",
    ],
    "India": [
        "Narendra Modi portrait",
    ],
    "Brazil": [
        "Lula da Silva portrait",
        "Jair Bolsonaro portrait",
    ],
    "Japan": [
        "Fumio Kishida portrait",
        "Shinzo Abe portrait",
    ],
    "Australia": [
        "Anthony Albanese portrait",
        "Scott Morrison portrait",
    ],
    "SouthAfrica": [
        "Cyril Ramaphosa portrait",
        "Nelson Mandela portrait",
    ],
    "Italy": [
        "Giorgia Meloni portrait",
        "Sergio Mattarella portrait",
    ],
    "Spain": [
        "Pedro Sanchez portrait",
    ],
    "Mexico": [
        "Andres Manuel Lopez Obrador portrait",
    ],
    "SouthKorea": [
        "Yoon Suk Yeol portrait",
        "Moon Jae-in portrait",
    ],
}

# 相册中的 Obama：固定若干公开图（不含查询图）
OBAMA_ALBUM_URLS = [
    # ageitgey 示例（将进入 USA/）
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
        "obama_album_01.jpg",
    ),
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
        "obama_album_02.jpg",
    ),
    (
        "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama_small.jpg",
        "obama_album_03.jpg",
    ),
]

# 查询专用 Obama：只用这张，绝不写入 album/
# Wikimedia：2009 官方肖像（与上面示例图不同源）
OBAMA_QUERY_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
)
OBAMA_QUERY_NAME = "Obama_QUERY_ONLY.jpg"


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def download(url: str, dest: Path, timeout: int = 50) -> bytes | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 3000:
        return dest.read_bytes()
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 2000:
            print(f"太小跳过 {dest.name}")
            return None
        dest.write_bytes(data)
        print(f"OK {dest.relative_to(ROOT)} ({len(data)} bytes)")
        return data
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"FAIL {dest.name}: {exc}")
        return None


def commons_search(query: str, limit: int = 4) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": "800",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"搜索失败 [{query}]: {exc}")
        return []

    urls: list[str] = []
    for page in payload.get("query", {}).get("pages", {}).values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        img = infos[0].get("thumburl") or infos[0].get("url")
        if img:
            urls.append(img)
    return urls


def fetch_country_folders(per_query: int = 3) -> dict[str, int]:
    counts: dict[str, int] = {}
    for country, queries in LEADER_QUERIES.items():
        folder = ALBUM / country
        folder.mkdir(parents=True, exist_ok=True)
        n = 0
        for q in queries:
            for i, img_url in enumerate(commons_search(q, limit=per_query)):
                ext = Path(urllib.parse.urlparse(img_url).path).suffix.lower() or ".jpg"
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    ext = ".jpg"
                safe_q = "".join(c if c.isalnum() else "_" for c in q)[:40]
                dest = folder / f"{safe_q}_{i}{ext}"
                if download(img_url, dest):
                    n += 1
                time.sleep(0.12)
        counts[country] = n
        print(f"[{country}] {n} 张")
    return counts


def fetch_obama_album_and_query() -> tuple[str, list[str]]:
    """下载 Obama 相册图 + 独立查询图，并校验查询图不在相册中。"""
    usa = ALBUM / "USA"
    usa.mkdir(parents=True, exist_ok=True)

    album_hashes: set[str] = set()
    album_files: list[str] = []

    for url, name in OBAMA_ALBUM_URLS:
        dest = usa / name
        data = download(url, dest)
        if data:
            album_hashes.add(md5_bytes(data))
            album_files.append(str(dest))

    # 再从 Commons 拉几张 Obama 进相册（排除官方那张查询 URL）
    for i, img_url in enumerate(commons_search("Barack Obama smiling photograph", limit=6)):
        if "President_Barack_Obama.jpg" in img_url:
            continue  # 跳过与查询图同源文件
        dest = usa / f"obama_album_wiki_{i}.jpg"
        data = download(img_url, dest)
        if data:
            album_hashes.add(md5_bytes(data))
            album_files.append(str(dest))
        time.sleep(0.12)

    query_path = REF / OBAMA_QUERY_NAME
    query_data = download(OBAMA_QUERY_URL, query_path)
    if query_data is None:
        raise SystemExit("奥巴马查询图下载失败")

    qhash = md5_bytes(query_data)
    if qhash in album_hashes:
        raise SystemExit("安全校验失败：查询图哈希出现在相册中，已中止")

    # 文件名隔离：查询目录不得位于 album 下
    if ALBUM in query_path.parents or query_path.is_relative_to(ALBUM):
        raise SystemExit("安全校验失败：查询图路径落在相册内")

    print(f"查询图 OK: {query_path}  md5={qhash[:12]}…")
    print(f"相册 Obama 文件数: {len(album_files)}")
    return str(query_path), album_files


def write_readme(counts: dict[str, int], query_path: str, album_obama: list[str]) -> None:
    total = sum(1 for _ in ALBUM.rglob("*.*"))
    text = f"""各国领导人测试相册（本地，勿提交 Git）
========================================

相册根目录（请在 FaceSort 中选择这个）：
  {ALBUM}

查询奥巴马专用参考图（不在相册内）：
  {query_path}

使用步骤：
  1. FaceSort → 浏览 → 选上面的 album 目录
  2. 取消「增量扫描」→ 开始扫描
  3. 人名填：Obama
  4. 参考照选：{query_path}
  5. 相似度建议 0.40~0.50
  6. 开始查找 → 应命中 USA 下 obama_album_* 等
  7. 标注格式：USA_Obama

隔离校验：
  - 查询图目录: testdata/leaders/ref/
  - 数据集目录: testdata/leaders/album/
  - 查询图文件名: {OBAMA_QUERY_NAME}
  - 相册内 Obama 示例数: {len(album_obama)}

子文件夹照片数：
{chr(10).join(f'  - {k}: {v}' for k, v in sorted(counts.items()))}
  - 合计文件约: {total}
"""
    (ROOT / "testdata" / "leaders" / "README_LOCAL.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ALBUM.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)

    print("1) 下载各国领导人肖像到子文件夹…")
    counts = fetch_country_folders(per_query=2)

    print("2) 准备 Obama 相册图 + 独立查询图…")
    query_path, album_obama = fetch_obama_album_and_query()

    # 统计各国实际文件数
    real_counts = {p.name: len(list(p.glob("*.*"))) for p in ALBUM.iterdir() if p.is_dir()}

    manifest = {
        "album_root": str(ALBUM),
        "obama_query": query_path,
        "obama_query_in_album": False,
        "obama_album_files": album_obama,
        "country_counts": real_counts,
    }
    META.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(real_counts, query_path, album_obama)
    print(f"清单: {META}")


if __name__ == "__main__":
    main()
