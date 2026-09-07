"""标签与导出路径工具：标注格式为「文件夹_人名」。"""

from __future__ import annotations

from pathlib import Path


def safe_name(name: str) -> str:
    """去除路径非法字符。"""
    forbidden = '<>:"/\\|?*'
    cleaned = "".join(c if c not in forbidden else "_" for c in name.strip())
    cleaned = cleaned.strip(" .")
    return cleaned or "未命名"


def folder_person_tag(photo_path: str | Path, person_name: str) -> str:
    """
    生成标注名：照片所在文件夹名_人名。
    例：groups_PersonA、case_identity_PersonObama
    """
    folder = Path(photo_path).parent.name or "root"
    return f"{safe_name(folder)}_{safe_name(person_name)}"


def apply_folder_person_tags(matches: list[dict], person_name: str) -> list[dict]:
    """为匹配结果写入 folder_person 格式的 person_name。"""
    out: list[dict] = []
    for m in matches:
        item = dict(m)
        item["person_name"] = folder_person_tag(item["path"], person_name)
        item["source_folder"] = Path(item["path"]).parent.name
        item["display_name"] = person_name
        out.append(item)
    return out
