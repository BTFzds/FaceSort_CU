"""FaceSort 桌面小程序界面（CustomTkinter，不依赖浏览器）。"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
import cv2
from PIL import Image

from facesort.config import db_path, export_path, load_config
from facesort.detector import FaceDetector
from facesort.export import export_photos
from facesort.indexer import Indexer
from facesort.matcher import find_matches
from facesort.pipeline import scan_directory
from facesort.tags import apply_folder_person_tags

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

ACCENT = "#2DD4BF"
ACCENT_HOVER = "#14B8A6"
CARD = "#1A1F2E"
BG = "#0F1219"
MUTED = "#94A3B8"
TEXT = "#F1F5F9"


class FaceSortApp(ctk.CTk):
    """单窗口工作流：扫描 → 查找 → 预览 → 导出。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("FaceSort 本地人脸整理")
        self.geometry("1220x780")
        self.minsize(1000, 680)
        self.configure(fg_color=BG)

        self.cfg = load_config()
        self.indexer = Indexer(db_path(self.cfg))
        self.detector: FaceDetector | None = None
        self.reference_path: str | None = None
        self.matches: list[dict[str, Any]] = []
        self._thumb_refs: list[ctk.CTkImage] = []
        self._ref_image: ctk.CTkImage | None = None
        self._busy = False

        self._build_ui()
        self._refresh_stats()
        self.after(200, self._preload_model_async)

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="FaceSort",
            font=ctk.CTkFont(family="Segoe UI Semibold", size=26),
            text_color=ACCENT,
        ).pack(side="left", padx=(24, 8), pady=16)

        ctk.CTkLabel(
            header,
            text="本地人脸照片整理 · 数据不出本机",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).pack(side="left", pady=16)

        self.stats_label = ctk.CTkLabel(header, text="", font=ctk.CTkFont(size=12), text_color=MUTED)
        self.stats_label.pack(side="right", padx=24)

        # 底部固定导出栏（避免被左侧内容挤出视野）
        export_bar = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=64)
        export_bar.pack(side="bottom", fill="x")
        export_bar.pack_propagate(False)

        ctk.CTkLabel(
            export_bar,
            text="③ 导出（按 文件夹_人名 分目录）",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(side="left", padx=20)

        self.export_hint = ctk.CTkLabel(export_bar, text="查找后可导出", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.export_hint.pack(side="left", padx=8)

        self.open_out_btn = ctk.CTkButton(
            export_bar,
            text="打开导出目录",
            width=120,
            height=36,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_output_dir,
        )
        self.open_out_btn.pack(side="right", padx=(8, 20), pady=12)

        self.export_btn = ctk.CTkButton(
            export_bar,
            text="导出匹配结果",
            width=140,
            height=36,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#042F2E",
            font=ctk.CTkFont(weight="bold"),
            command=self._export,
        )
        self.export_btn.pack(side="right", padx=8, pady=12)

        body = ctk.CTkFrame(self, fg_color=BG)
        body.pack(fill="both", expand=True, padx=18, pady=12)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=3)
        body.grid_rowconfigure(0, weight=1)

        left_wrap = ctk.CTkFrame(body, fg_color=CARD, corner_radius=16)
        left_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left = ctk.CTkScrollableFrame(left_wrap, fg_color=CARD, corner_radius=16)
        left.pack(fill="both", expand=True, padx=4, pady=4)

        right = ctk.CTkFrame(body, fg_color=CARD, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent,
            text="① 选择相册并扫描",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=14, pady=(12, 4))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=6)
        self.folder_var = tk.StringVar()
        ctk.CTkEntry(row, textvariable=self.folder_var, placeholder_text="选择含照片的文件夹…", height=36).pack(
            side="left", fill="x", expand=True, padx=(0, 8)
        )
        ctk.CTkButton(
            row,
            text="浏览",
            width=72,
            height=36,
            fg_color="#334155",
            hover_color="#475569",
            command=self._pick_folder,
        ).pack(side="left")

        self.incremental_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            parent,
            text="增量扫描（跳过未改动的文件）",
            variable=self.incremental_var,
            text_color=MUTED,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
        ).pack(anchor="w", padx=14, pady=4)

        self.scan_btn = ctk.CTkButton(
            parent,
            text="开始扫描",
            height=40,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#042F2E",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_scan,
        )
        self.scan_btn.pack(fill="x", padx=14, pady=(8, 4))

        self.scan_progress = ctk.CTkProgressBar(parent, progress_color=ACCENT, height=10)
        self.scan_progress.pack(fill="x", padx=14, pady=6)
        self.scan_progress.set(0)
        self.scan_status = ctk.CTkLabel(parent, text="等待扫描", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.scan_status.pack(anchor="w", padx=14)

        ctk.CTkLabel(
            parent,
            text="② 参考脸查找",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=14, pady=(18, 4))

        name_row = ctk.CTkFrame(parent, fg_color="transparent")
        name_row.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(name_row, text="人名", text_color=MUTED, width=48).pack(side="left")
        self.name_var = tk.StringVar(value="PersonA")
        ctk.CTkEntry(name_row, textvariable=self.name_var, height=34).pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(
            parent,
            text="标注格式：照片所在文件夹_人名（例 groups_PersonA）",
            text_color=MUTED,
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", padx=14, pady=(0, 6))

        ref_row = ctk.CTkFrame(parent, fg_color="transparent")
        ref_row.pack(fill="x", padx=14, pady=8)
        self.ref_preview = ctk.CTkLabel(
            ref_row,
            text="未选参考照",
            width=120,
            height=120,
            fg_color="#0B1220",
            corner_radius=10,
            text_color=MUTED,
        )
        self.ref_preview.pack(side="left", padx=(0, 10))
        ref_btns = ctk.CTkFrame(ref_row, fg_color="transparent")
        ref_btns.pack(side="left", fill="both", expand=True)
        ctk.CTkButton(
            ref_btns,
            text="选择参考照片",
            height=36,
            fg_color="#334155",
            hover_color="#475569",
            command=self._pick_reference,
        ).pack(fill="x", pady=(8, 6))
        ctk.CTkLabel(
            ref_btns,
            text="建议清晰正脸；\n合照检索阈值可降至 0.35。",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
            justify="left",
        ).pack(anchor="w")

        thr_row = ctk.CTkFrame(parent, fg_color="transparent")
        thr_row.pack(fill="x", padx=14, pady=6)
        ctk.CTkLabel(thr_row, text="相似度", text_color=MUTED, width=48).pack(side="left")
        self.threshold_var = tk.DoubleVar(value=float(self.cfg.get("match_threshold", 0.45)))
        self.threshold_label = ctk.CTkLabel(
            thr_row, text=f"{self.threshold_var.get():.2f}", text_color=TEXT, width=40
        )
        self.threshold_label.pack(side="right")
        ctk.CTkSlider(
            thr_row,
            from_=0.30,
            to=0.85,
            number_of_steps=55,
            variable=self.threshold_var,
            progress_color=ACCENT,
            button_color=ACCENT,
            command=self._on_threshold,
        ).pack(side="left", fill="x", expand=True, padx=8)

        self.find_btn = ctk.CTkButton(
            parent,
            text="开始查找",
            height=40,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#042F2E",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_find,
        )
        self.find_btn.pack(fill="x", padx=14, pady=(8, 4))

        self.find_progress = ctk.CTkProgressBar(parent, progress_color=ACCENT, height=10)
        self.find_progress.pack(fill="x", padx=14, pady=6)
        self.find_progress.set(0)
        self.find_status = ctk.CTkLabel(parent, text="等待查找", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.find_status.pack(anchor="w", padx=14, pady=(0, 14))

    def _build_right(self, parent: ctk.CTkFrame) -> None:
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(14, 6))
        ctk.CTkLabel(
            top,
            text="匹配结果预览",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).pack(side="left")
        self.result_count = ctk.CTkLabel(top, text="0 张", text_color=ACCENT)
        self.result_count.pack(side="right")

        self.gallery = ctk.CTkScrollableFrame(parent, fg_color="#0B1220", corner_radius=12)
        self.gallery.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.gallery_empty = ctk.CTkLabel(
            self.gallery,
            text="查找后在此显示缩略图\n（会自动滚回顶部）",
            text_color=MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.gallery_empty.pack(expand=True, pady=80)

    def _on_threshold(self, value: float) -> None:
        self.threshold_label.configure(text=f"{float(value):.2f}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.scan_btn.configure(state=state)
        self.find_btn.configure(state=state)
        self.export_btn.configure(state=state)

    def _gpu_status_text(self) -> str:
        if self.detector is None:
            return "模型加载中…"
        providers = ",".join(self.detector.active_providers)
        if self.detector.using_gpu and "CUDAExecutionProvider" in self.detector.active_providers:
            return f"GPU加速中 · {self.detector.device_label}"
        return f"CPU运行 · providers=[{providers}]"

    def _refresh_stats(self) -> None:
        stats = self.indexer.stats()
        self.stats_label.configure(
            text=(
                f"{self._gpu_status_text()}  ·  "
                f"索引 {stats['photos']} 张 / 人脸 {stats['faces']}  ·  标注 {stats['tags']}"
            )
        )

    def _preload_model_async(self) -> None:
        def work() -> None:
            try:
                self._get_detector()

                def done() -> None:
                    self._refresh_stats()
                    gpu = "已启用 CUDA 加速" if self.detector and self.detector.using_gpu else "当前为 CPU"
                    self.scan_status.configure(text=f"模型已就绪（{gpu}），可以开始扫描")

                self.after(0, done)
            except Exception as exc:  # noqa: BLE001
                logger.exception("模型加载失败")
                self.after(0, lambda: messagebox.showerror("模型加载失败", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _get_detector(self) -> FaceDetector:
        if self.detector is None:
            self.detector = FaceDetector(
                model_name=self.cfg["model_name"],
                use_gpu=self.cfg["use_gpu"],
                det_size=int(self.cfg["det_size"]),
                min_face_size=int(self.cfg["min_face_size"]),
                det_thresh=float(self.cfg.get("det_thresh", 0.35)),
            )
        return self.detector

    def _pick_folder(self) -> None:
        path = filedialog.askdirectory(title="选择照片文件夹")
        if path:
            self.folder_var.set(path)

    def _pick_reference(self) -> None:
        path = filedialog.askopenfilename(
            title="选择参考照片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.webp *.bmp"), ("全部", "*.*")],
        )
        if not path:
            return
        self.reference_path = path
        img = Image.open(path).convert("RGB")
        img.thumbnail((120, 120))
        self._ref_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.ref_preview.configure(image=self._ref_image, text="")

    def _scroll_gallery_top(self) -> None:
        """预览区滚回顶部。"""
        try:
            canvas = getattr(self.gallery, "_parent_canvas", None)
            if canvas is not None:
                canvas.yview_moveto(0)
            else:
                self.gallery._parent_canvas.yview_moveto(0)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    def _clear_gallery(self) -> None:
        for child in self.gallery.winfo_children():
            child.destroy()
        self._thumb_refs.clear()

    def _render_gallery(self, matches: list[dict[str, Any]]) -> None:
        self._clear_gallery()
        self.result_count.configure(text=f"{len(matches)} 张")
        self._scroll_gallery_top()

        if not matches:
            ctk.CTkLabel(
                self.gallery,
                text="没有匹配结果\n可降低相似度阈值后重试",
                text_color=MUTED,
                font=ctk.CTkFont(size=13),
            ).pack(expand=True, pady=80)
            self.export_hint.configure(text="暂无结果可导出")
            self.after(50, self._scroll_gallery_top)
            return

        cols = 3
        for i, item in enumerate(matches[:60]):
            path = item["path"]
            tag = item.get("person_name", "")
            frame = ctk.CTkFrame(self.gallery, fg_color=CARD, corner_radius=10)
            frame.grid(row=i // cols, column=i % cols, padx=8, pady=8, sticky="nsew")
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((200, 200))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self._thumb_refs.append(ctk_img)
                ctk.CTkLabel(frame, image=ctk_img, text="").pack(padx=6, pady=(6, 2))
            except OSError:
                ctk.CTkLabel(frame, text="无法预览", text_color=MUTED).pack(padx=6, pady=20)
            ctk.CTkLabel(
                frame,
                text=f"{Path(path).name}\n{tag}\n相似度 {item['similarity']:.3f}",
                text_color=MUTED,
                font=ctk.CTkFont(size=11),
            ).pack(padx=6, pady=(0, 8))

        tags = sorted({m.get("person_name", "") for m in matches if m.get("person_name")})
        self.export_hint.configure(text=f"将导出到：{' / '.join(tags[:3])}" + ("…" if len(tags) > 3 else ""))
        # 渲染后再回正一次，避免滚动条停在旧位置
        self.after(50, self._scroll_gallery_top)
        self.after(200, self._scroll_gallery_top)

    def _start_scan(self) -> None:
        if self._busy:
            return
        folder = self.folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showwarning("提示", "请先选择有效的照片文件夹")
            return

        self._set_busy(True)
        self.scan_progress.set(0)
        self.scan_status.configure(text="准备扫描…")

        def work() -> None:
            try:
                detector = self._get_detector()

                def on_progress(cur: int, total: int, msg: str) -> None:
                    ratio = cur / total if total else 0

                    def update(r: float = ratio, c: int = cur, t: int = total, m: str = msg) -> None:
                        self.scan_progress.set(r)
                        self.scan_status.configure(text=f"{c}/{t}  {m}")

                    self.after(0, update)

                result = scan_directory(
                    folder,
                    self.indexer,
                    detector,
                    self.cfg["image_extensions"],
                    incremental=self.incremental_var.get(),
                    progress_callback=on_progress,
                )

                def done() -> None:
                    self.scan_progress.set(1)
                    msg = (
                        f"完成：共 {result['total_files']} 张｜新扫描 {result['scanned']}｜"
                        f"跳过 {result['skipped']}｜新人脸 {result['faces']}｜"
                        f"保留人脸 {result['retained_faces']}"
                    )
                    if result["errors"]:
                        msg += f"｜失败 {result['errors']}"
                    self.scan_status.configure(text=msg)
                    self._refresh_stats()
                    self._set_busy(False)

                self.after(0, done)
            except Exception as exc:  # noqa: BLE001
                logger.exception("扫描失败")
                self.after(0, lambda: (messagebox.showerror("扫描失败", str(exc)), self._set_busy(False)))

        threading.Thread(target=work, daemon=True).start()

    def _start_find(self) -> None:
        if self._busy:
            return
        name = self.name_var.get().strip() or "PersonA"
        if not self.reference_path:
            messagebox.showwarning("提示", "请先选择参考照片")
            return

        self._set_busy(True)
        self.find_progress.set(0)
        self.find_status.configure(text="提取参考脸…")
        self._clear_gallery()
        self.result_count.configure(text="查找中…")
        self._scroll_gallery_top()
        threshold = float(self.threshold_var.get())

        def work() -> None:
            try:
                detector = self._get_detector()
                img = cv2.imread(self.reference_path)
                if img is None:
                    raise RuntimeError("无法读取参考照片")
                faces = detector.detect_from_array(img)
                reference = detector.reference_embedding(faces)
                if reference is None:
                    raise RuntimeError("参考照未检测到人脸，请换一张更清晰的正脸")

                def on_progress(cur: int, total: int, msg: str) -> None:
                    ratio = cur / total if total else 0

                    def update(r: float = ratio, m: str = msg) -> None:
                        self.find_progress.set(r)
                        self.find_status.configure(text=m)

                    self.after(0, update)

                raw = find_matches(
                    self.indexer,
                    reference,
                    threshold,
                    name,
                    progress_callback=on_progress,
                )
                matches = apply_folder_person_tags(raw, name)
                self.matches = matches
                if matches:
                    self.indexer.add_tags(matches)

                def done() -> None:
                    self.find_progress.set(1)
                    if matches:
                        sample_tags = sorted({m["person_name"] for m in matches})[:3]
                        self.find_status.configure(
                            text=(
                                f"找到 {len(matches)} 张（阈值 {threshold:.2f}），"
                                f"标注示例：{', '.join(sample_tags)}"
                            )
                        )
                    else:
                        self.find_status.configure(
                            text=f"未找到匹配，可降低阈值后重试（当前 {threshold:.2f}）"
                        )
                    self._render_gallery(matches)
                    self._refresh_stats()
                    self._set_busy(False)

                self.after(0, done)
            except Exception as exc:  # noqa: BLE001
                logger.exception("查找失败")
                self.after(0, lambda: (messagebox.showerror("查找失败", str(exc)), self._set_busy(False)))

        threading.Thread(target=work, daemon=True).start()

    def _export(self) -> None:
        name = self.name_var.get().strip() or "PersonA"
        matches = self.matches
        if not matches:
            # 回退：按人名模糊查库（旧标注可能不含文件夹前缀）
            tagged = self.indexer.get_tagged_photos(name)
            if not tagged:
                # 尝试取所有含 _人名 后缀的标签
                messagebox.showwarning("提示", "没有可导出的结果，请先查找")
                return
            matches = apply_folder_person_tags(
                [{"path": t["path"], "similarity": t["similarity"], "photo_id": 0} for t in tagged],
                name,
            )

        dest = export_path(self.cfg)
        result = export_photos(matches, dest, name, mode="copy")
        messagebox.showinfo(
            "导出完成",
            f"已导出 {result['exported']} 张\n"
            f"子文件夹数：{result.get('folders', 1)}\n"
            f"位置：{result['dest']}\n\n"
            "目录名格式：文件夹_人名",
        )
        self.export_hint.configure(text=f"已导出 {result['exported']} 张 → {result['dest']}")

    def _open_output_dir(self) -> None:
        dest = export_path(self.cfg)
        dest.mkdir(parents=True, exist_ok=True)
        import os

        os.startfile(dest)  # noqa: S606 — Windows 本地打开文件夹


def launch() -> None:
    """启动桌面小程序。"""
    app = FaceSortApp()
    app.mainloop()
