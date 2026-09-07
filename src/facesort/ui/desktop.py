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
    """单窗口工作流：扫描 → 查找 → 预览 → 导出。左侧无滚动条，一屏展示全部控件。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("FaceSort 本地人脸整理")
        self.geometry("1280x820")
        self.minsize(1180, 760)
        self.configure(fg_color=BG)

        self.cfg = load_config()
        self.indexer = Indexer(db_path(self.cfg))
        self.detector: FaceDetector | None = None
        self.reference_path: str | None = None
        self.matches: list[dict[str, Any]] = []
        self._thumb_refs: list[ctk.CTkImage] = []
        self._ref_image: ctk.CTkImage | None = None
        self._busy = False
        self._default_threshold = float(self.cfg.get("match_threshold", 0.45))

        self._build_ui()
        self._refresh_stats()
        self.after(200, self._preload_model_async)

    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="FaceSort",
            font=ctk.CTkFont(family="Segoe UI Semibold", size=22),
            text_color=ACCENT,
        ).pack(side="left", padx=(18, 8), pady=12)

        ctk.CTkLabel(
            header,
            text="本地整理 · 数据不出本机",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        ).pack(side="left", pady=12)

        self.reset_btn = ctk.CTkButton(
            header,
            text="重置",
            width=72,
            height=32,
            fg_color="#7F1D1D",
            hover_color="#991B1B",
            command=self._reset_all,
        )
        self.reset_btn.pack(side="right", padx=(8, 16), pady=12)

        self.stats_label = ctk.CTkLabel(header, text="", font=ctk.CTkFont(size=11), text_color=MUTED)
        self.stats_label.pack(side="right", padx=8)

        export_bar = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=54)
        export_bar.pack(side="bottom", fill="x")
        export_bar.pack_propagate(False)

        ctk.CTkLabel(
            export_bar,
            text="③ 导出（文件夹_人名）",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(side="left", padx=16)

        self.export_hint = ctk.CTkLabel(
            export_bar, text="查找后可导出", text_color=MUTED, font=ctk.CTkFont(size=11)
        )
        self.export_hint.pack(side="left", padx=6)

        self.open_out_btn = ctk.CTkButton(
            export_bar,
            text="打开导出目录",
            width=110,
            height=32,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_output_dir,
        )
        self.open_out_btn.pack(side="right", padx=(6, 16), pady=10)

        self.export_btn = ctk.CTkButton(
            export_bar,
            text="导出匹配结果",
            width=120,
            height=32,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#042F2E",
            font=ctk.CTkFont(weight="bold"),
            command=self._export,
        )
        self.export_btn.pack(side="right", padx=6, pady=10)

        body = ctk.CTkFrame(self, fg_color=BG)
        body.pack(fill="both", expand=True, padx=14, pady=10)
        body.grid_columnconfigure(0, weight=0, minsize=420)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # 左侧固定面板：不用 ScrollableFrame，避免滚轮
        left = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14, width=420)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_propagate(False)

        right = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent: ctk.CTkFrame) -> None:
        pad_x = 12

        ctk.CTkLabel(
            parent,
            text="① 扫描相册",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=pad_x, pady=(10, 2))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=pad_x, pady=2)
        self.folder_var = tk.StringVar()
        ctk.CTkEntry(
            row,
            textvariable=self.folder_var,
            placeholder_text="选择含照片的文件夹…",
            height=30,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(
            row,
            text="浏览",
            width=64,
            height=30,
            fg_color="#334155",
            hover_color="#475569",
            command=self._pick_folder,
        ).pack(side="left")

        opt = ctk.CTkFrame(parent, fg_color="transparent")
        opt.pack(fill="x", padx=pad_x, pady=2)
        self.incremental_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opt,
            text="增量扫描",
            variable=self.incremental_var,
            text_color=MUTED,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            width=100,
        ).pack(side="left")

        self.scan_btn = ctk.CTkButton(
            parent,
            text="开始扫描",
            height=34,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#042F2E",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._start_scan,
        )
        self.scan_btn.pack(fill="x", padx=pad_x, pady=(4, 2))

        self.scan_progress = ctk.CTkProgressBar(parent, progress_color=ACCENT, height=8)
        self.scan_progress.pack(fill="x", padx=pad_x, pady=2)
        self.scan_progress.set(0)
        self.scan_status = ctk.CTkLabel(
            parent, text="等待扫描", text_color=MUTED, font=ctk.CTkFont(size=11), wraplength=380, justify="left"
        )
        self.scan_status.pack(anchor="w", padx=pad_x)

        # 分隔线
        ctk.CTkFrame(parent, fg_color="#334155", height=1).pack(fill="x", padx=pad_x, pady=8)

        ctk.CTkLabel(
            parent,
            text="② 参考脸查找",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=pad_x, pady=(0, 2))

        name_row = ctk.CTkFrame(parent, fg_color="transparent")
        name_row.pack(fill="x", padx=pad_x, pady=2)
        ctk.CTkLabel(name_row, text="人名", text_color=MUTED, width=40).pack(side="left")
        self.name_var = tk.StringVar(value="PersonA")
        ctk.CTkEntry(name_row, textvariable=self.name_var, height=30).pack(
            side="left", fill="x", expand=True
        )

        ctk.CTkLabel(
            parent,
            text="标注：文件夹_人名（如 USA_Obama）",
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", padx=pad_x)

        ref_row = ctk.CTkFrame(parent, fg_color="transparent")
        ref_row.pack(fill="x", padx=pad_x, pady=6)
        self.ref_preview = ctk.CTkLabel(
            ref_row,
            text="参考照",
            width=72,
            height=72,
            fg_color="#0B1220",
            corner_radius=8,
            text_color=MUTED,
            font=ctk.CTkFont(size=11),
        )
        self.ref_preview.pack(side="left", padx=(0, 8))
        ref_btns = ctk.CTkFrame(ref_row, fg_color="transparent")
        ref_btns.pack(side="left", fill="both", expand=True)
        ctk.CTkButton(
            ref_btns,
            text="选择参考照片",
            height=30,
            fg_color="#334155",
            hover_color="#475569",
            command=self._pick_reference,
        ).pack(fill="x", pady=(4, 4))
        ctk.CTkLabel(
            ref_btns,
            text="清晰正脸；合照可降至 0.35",
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
            anchor="w",
        ).pack(fill="x")

        thr_row = ctk.CTkFrame(parent, fg_color="transparent")
        thr_row.pack(fill="x", padx=pad_x, pady=2)
        ctk.CTkLabel(thr_row, text="相似度", text_color=MUTED, width=44).pack(side="left")
        self.threshold_var = tk.DoubleVar(value=self._default_threshold)
        self.threshold_label = ctk.CTkLabel(
            thr_row, text=f"{self._default_threshold:.2f}", text_color=TEXT, width=36
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
            height=16,
            command=self._on_threshold,
        ).pack(side="left", fill="x", expand=True, padx=6)

        self.find_btn = ctk.CTkButton(
            parent,
            text="开始查找",
            height=34,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color="#042F2E",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._start_find,
        )
        self.find_btn.pack(fill="x", padx=pad_x, pady=(6, 2))

        self.find_progress = ctk.CTkProgressBar(parent, progress_color=ACCENT, height=8)
        self.find_progress.pack(fill="x", padx=pad_x, pady=2)
        self.find_progress.set(0)
        self.find_status = ctk.CTkLabel(
            parent, text="等待查找", text_color=MUTED, font=ctk.CTkFont(size=11), wraplength=380, justify="left"
        )
        self.find_status.pack(anchor="w", padx=pad_x, pady=(0, 8))

    def _build_right(self, parent: ctk.CTkFrame) -> None:
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(
            top,
            text="匹配结果预览",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT,
        ).pack(side="left")
        self.result_count = ctk.CTkLabel(top, text="0 张", text_color=ACCENT)
        self.result_count.pack(side="right")

        self.gallery = ctk.CTkScrollableFrame(parent, fg_color="#0B1220", corner_radius=10)
        self.gallery.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._show_gallery_placeholder("查找后在此显示缩略图")

    def _show_gallery_placeholder(self, text: str) -> None:
        self._clear_gallery()
        ctk.CTkLabel(
            self.gallery,
            text=text,
            text_color=MUTED,
            font=ctk.CTkFont(size=13),
        ).pack(expand=True, pady=60)
        self.result_count.configure(text="0 张")
        self.after(30, self._scroll_gallery_top)

    def _on_threshold(self, value: float) -> None:
        self.threshold_label.configure(text=f"{float(value):.2f}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.scan_btn.configure(state=state)
        self.find_btn.configure(state=state)
        self.export_btn.configure(state=state)
        self.reset_btn.configure(state=state)

    def _gpu_status_text(self) -> str:
        if self.detector is None:
            return "模型加载中…"
        providers = ",".join(self.detector.active_providers)
        if self.detector.using_gpu and "CUDAExecutionProvider" in self.detector.active_providers:
            return f"GPU加速中 · {self.detector.device_label}"
        return f"CPU运行 · [{providers}]"

    def _refresh_stats(self) -> None:
        stats = self.indexer.stats()
        self.stats_label.configure(
            text=(
                f"{self._gpu_status_text()}  ·  "
                f"索引 {stats['photos']} / 人脸 {stats['faces']} / 标注 {stats['tags']}"
            )
        )

    def _preload_model_async(self) -> None:
        def work() -> None:
            try:
                self._get_detector()

                def done() -> None:
                    self._refresh_stats()
                    gpu = "CUDA 已启用" if self.detector and self.detector.using_gpu else "CPU"
                    self.scan_status.configure(text=f"模型就绪（{gpu}），可开始扫描")

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
        img.thumbnail((72, 72))
        self._ref_image = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.ref_preview.configure(image=self._ref_image, text="")

    def _scroll_gallery_top(self) -> None:
        try:
            canvas = getattr(self.gallery, "_parent_canvas", None)
            if canvas is not None:
                canvas.yview_moveto(0)
        except Exception:  # noqa: BLE001
            pass

    def _clear_gallery(self) -> None:
        for child in self.gallery.winfo_children():
            child.destroy()
        self._thumb_refs.clear()

    def _reset_all(self) -> None:
        """重置界面与匹配结果；可选清空本地索引库。"""
        if self._busy:
            messagebox.showinfo("提示", "请等待当前任务完成后再重置")
            return

        clear_db = messagebox.askyesnocancel(
            "重置",
            "将清空：参考照、匹配结果、进度与预览。\n\n"
            "是否同时清空本地索引库？\n"
            "（是=清空索引；否=只重置界面；取消=不操作）",
        )
        if clear_db is None:
            return

        self.matches = []
        self.reference_path = None
        self._ref_image = None
        self.ref_preview.configure(image=None, text="参考照")
        self.name_var.set("PersonA")
        self.threshold_var.set(self._default_threshold)
        self.threshold_label.configure(text=f"{self._default_threshold:.2f}")
        self.incremental_var.set(True)

        self.scan_progress.set(0)
        self.find_progress.set(0)
        self.scan_status.configure(text="已重置，可重新扫描")
        self.find_status.configure(text="等待查找")
        self.export_hint.configure(text="查找后可导出")
        self._show_gallery_placeholder("已重置，查找后在此显示结果")

        if clear_db:
            db = db_path(self.cfg)
            try:
                if db.exists():
                    db.unlink()
            except OSError as exc:
                messagebox.showerror("重置失败", f"无法删除索引库：{exc}")
                return
            self.indexer = Indexer(db)
            self.scan_status.configure(text="已重置并清空索引，请重新扫描")

        self._refresh_stats()
        messagebox.showinfo("完成", "界面已重置" + ("，索引库已清空" if clear_db else ""))

    def _render_gallery(self, matches: list[dict[str, Any]]) -> None:
        self._clear_gallery()
        self.result_count.configure(text=f"{len(matches)} 张")
        self._scroll_gallery_top()

        if not matches:
            self._show_gallery_placeholder("没有匹配结果\n可降低相似度阈值后重试")
            self.export_hint.configure(text="暂无结果可导出")
            return

        cols = 3
        for i, item in enumerate(matches[:60]):
            path = item["path"]
            tag = item.get("person_name", "")
            frame = ctk.CTkFrame(self.gallery, fg_color=CARD, corner_radius=8)
            frame.grid(row=i // cols, column=i % cols, padx=6, pady=6, sticky="nsew")
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((180, 180))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self._thumb_refs.append(ctk_img)
                ctk.CTkLabel(frame, image=ctk_img, text="").pack(padx=4, pady=(4, 2))
            except OSError:
                ctk.CTkLabel(frame, text="无法预览", text_color=MUTED).pack(padx=4, pady=16)
            ctk.CTkLabel(
                frame,
                text=f"{Path(path).name}\n{tag}\n{item['similarity']:.3f}",
                text_color=MUTED,
                font=ctk.CTkFont(size=10),
            ).pack(padx=4, pady=(0, 6))

        tags = sorted({m.get("person_name", "") for m in matches if m.get("person_name")})
        self.export_hint.configure(text=f"将导出：{' / '.join(tags[:3])}" + ("…" if len(tags) > 3 else ""))
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
                        f"完成：共{result['total_files']}｜新扫{result['scanned']}｜"
                        f"跳过{result['skipped']}｜新人脸{result['faces']}｜"
                        f"保留{result['retained_faces']}"
                    )
                    if result["errors"]:
                        msg += f"｜失败{result['errors']}"
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
        self._show_gallery_placeholder("查找中…")
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
                            text=f"找到 {len(matches)} 张｜标注：{', '.join(sample_tags)}"
                        )
                    else:
                        self.find_status.configure(
                            text=f"未找到匹配，可降低阈值（当前 {threshold:.2f}）"
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
            tagged = self.indexer.get_tagged_photos(name)
            if not tagged:
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
        self.export_hint.configure(text=f"已导出 {result['exported']} 张")

    def _open_output_dir(self) -> None:
        dest = export_path(self.cfg)
        dest.mkdir(parents=True, exist_ok=True)
        import os

        os.startfile(dest)  # noqa: S606


def launch() -> None:
    """启动桌面小程序。"""
    app = FaceSortApp()
    app.mainloop()
