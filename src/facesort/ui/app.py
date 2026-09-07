"""Gradio 简易界面 — 仅绑定本机，数据不出本地。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import gradio as gr

from facesort.config import db_path, export_path, load_config
from facesort.detector import FaceDetector
from facesort.gpu import gpu_status_message
from facesort.export import export_photos
from facesort.indexer import Indexer
from facesort.matcher import find_matches
from facesort.pipeline import scan_directory

logger = logging.getLogger(__name__)

# 全局单例，避免重复加载模型
_detector: FaceDetector | None = None
_indexer: Indexer | None = None
_last_matches: list[dict[str, Any]] = []


def _get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        cfg = load_config()
        _detector = FaceDetector(
            model_name=cfg["model_name"],
            use_gpu=cfg["use_gpu"],
            det_size=cfg["det_size"],
            min_face_size=cfg["min_face_size"],
        )
    return _detector


def _get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer(db_path())
    return _indexer


def _stats_text() -> str:
    cfg = load_config()
    stats = _get_indexer().stats()
    gpu_line = gpu_status_message(cfg["use_gpu"])
    if _detector is not None:
        gpu_line = f"**加速模式：{_detector.device_label}**"
    return (
        f"{gpu_line}  \n"
        f"已索引 **{stats['photos']}** 张照片，检测到 **{stats['faces']}** 张人脸，"
        f"已标注 **{stats['tags']}** 条。"
    )


def do_scan(folder: str, incremental: bool) -> tuple[str, str]:
    """步骤1：扫描文件夹。"""
    folder = folder.strip()
    if not folder:
        return "请先输入照片文件夹路径。", _stats_text()

    root = Path(folder)
    if not root.is_dir():
        return f"目录不存在: {folder}", _stats_text()

    cfg = load_config()
    result = scan_directory(
        root,
        _get_indexer(),
        _get_detector(),
        cfg["image_extensions"],
        incremental=incremental,
    )
    msg = (
        f"扫描完成！共发现 {result['total_files']} 张图片，"
        f"新扫描 {result['scanned']} 张，跳过 {result['skipped']} 张，"
        f"检测到 {result['faces']} 张人脸。"
    )
    if result["errors"]:
        msg += f"（{result['errors']} 张处理失败，详见日志）"
    return msg, _stats_text()


def do_match(
    reference_files: list[str] | None,
    person_name: str,
    threshold: float,
) -> tuple[str, list[tuple[str, str]], str]:
    """步骤2：上传参考脸，查找匹配照片。"""
    global _last_matches

    person_name = person_name.strip()
    if not person_name:
        return "请输入人物姓名。", [], _stats_text()
    if not reference_files:
        return "请上传至少一张参考照片（含该人物清晰正脸）。", [], _stats_text()

    detector = _get_detector()
    ref_faces: list[dict[str, Any]] = []
    for file_path in reference_files:
        img = cv2.imread(file_path)
        if img is None:
            continue
        ref_faces.extend(detector.detect_from_array(img))

    reference = detector.reference_embedding(ref_faces)
    if reference is None:
        return "参考照片中未检测到人脸，请换一张更清晰的照片。", [], _stats_text()

    matches = find_matches(_get_indexer(), reference, threshold, person_name)
    _last_matches = matches

    if not matches:
        return f"未找到相似度 ≥ {threshold:.2f} 的照片，可尝试降低阈值或补充参考脸。", [], _stats_text()

    gallery = [(m["path"], f"相似度 {m['similarity']:.3f}") for m in matches[:200]]
    msg = f"找到 **{len(matches)}** 张可能属于「{person_name}」的照片（显示前 200 张）。"
    return msg, gallery, _stats_text()


def do_tag(person_name: str) -> tuple[str, str]:
    """将当前匹配结果写入标签库。"""
    global _last_matches
    if not _last_matches:
        return "请先完成「查找人物」步骤。", _stats_text()
    if not person_name.strip():
        return "请输入人物姓名。", _stats_text()

    for m in _last_matches:
        m["person_name"] = person_name.strip()
    added = _get_indexer().add_tags(_last_matches)
    return f"已标注 {added} 条记录。", _stats_text()


def do_export(person_name: str, mode: str) -> str:
    """步骤3：导出到文件夹。"""
    name = person_name.strip()
    if not name:
        return "请输入要导出的人物姓名。"

    tagged = _get_indexer().get_tagged_photos(name)
    if not tagged:
        if _last_matches and all(m.get("person_name") == name for m in _last_matches):
            matches = _last_matches
        else:
            return f"没有找到「{name}」的标注记录，请先查找并标注。"
    else:
        matches = [{"path": t["path"]} for t in tagged]

    cfg = load_config()
    dest_root = export_path(cfg)
    result = export_photos(matches, dest_root, name, mode="hardlink" if mode == "硬链接" else "copy")
    return f"已导出 {result['exported']} 张到: {result['dest']}"


def build_app() -> gr.Blocks:
    """构建 Gradio 应用。"""
    cfg = load_config()

    with gr.Blocks(title="FaceSort 本地人脸照片整理", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # FaceSort — 本地人脸照片整理工具
            **100% 本地运行，照片不会上传到任何服务器。**

            ### 使用步骤
            1. **扫描照片** — 输入文件夹路径，建立人脸索引
            2. **查找人物** — 上传参考脸，自动找出所有相关照片
            3. **标注 & 导出** — 保存标签，复制到分类文件夹
            """
        )

        stats = gr.Markdown(_stats_text())

        with gr.Tab("① 扫描照片"):
            folder_input = gr.Textbox(
                label="照片文件夹路径",
                placeholder=r"例如 D:\Photos\2024 或 E:\相册",
            )
            incremental = gr.Checkbox(label="增量扫描（跳过未变化的文件）", value=True)
            scan_btn = gr.Button("开始扫描", variant="primary")
            scan_result = gr.Textbox(label="扫描结果", lines=3)

            scan_btn.click(do_scan, [folder_input, incremental], [scan_result, stats])

        with gr.Tab("② 查找人物"):
            person_input = gr.Textbox(label="人物姓名", placeholder="例如：PersonA")
            ref_upload = gr.File(
                label="参考照片（可上传多张，选最清晰的正脸）",
                file_count="multiple",
                file_types=["image"],
            )
            threshold = gr.Slider(
                minimum=0.3,
                maximum=0.9,
                value=cfg["match_threshold"],
                step=0.05,
                label="相似度阈值（越高越严格，建议 0.45~0.65）",
            )
            match_btn = gr.Button("开始查找", variant="primary")
            match_result = gr.Markdown()
            gallery = gr.Gallery(label="匹配结果预览", columns=4, height=400)

            match_btn.click(
                do_match,
                [ref_upload, person_input, threshold],
                [match_result, gallery, stats],
            )

        with gr.Tab("③ 标注 & 导出"):
            tag_name = gr.Textbox(label="人物姓名（与查找时一致）")
            tag_btn = gr.Button("保存标注到索引")
            tag_result = gr.Textbox(label="标注结果")

            export_mode = gr.Radio(
                choices=["复制", "硬链接"],
                value="复制",
                label="导出方式（硬链接不占额外空间，但不可跨磁盘）",
            )
            export_btn = gr.Button("导出到文件夹", variant="primary")
            export_result = gr.Textbox(label="导出结果")

            tag_btn.click(do_tag, [tag_name], [tag_result, stats])
            export_btn.click(do_export, [tag_name, export_mode], [export_result])

        gr.Markdown(
            """
            ---
            **隐私说明**：本工具仅在您的电脑上运行。首次使用会自动下载 InsightFace 模型到本地。
            """
        )

    return app


def launch() -> None:
    """启动本地 Web 界面。"""
    cfg = load_config()
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=cfg["server_port"],
        share=False,
        show_error=True,
    )
