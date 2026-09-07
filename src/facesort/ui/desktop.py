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
from PIL import Image, ImageTk

from facesort.config import db_path, export_path, load_config
from facesort.detector import FaceDetector
from facesort.export import export_photos
from facesort.indexer import Indexer
from facesort.matcher import find_matches
from facesort.pipeline import scan_directory

logger = logging.getLogger(__name__)

# 视觉主题：深色 + 青绿强调（避免默认紫）
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

ACCENT = "#2DD4BF"
ACCENT_HOVER = "#14B8A6"
CARD = "#1A1F2E"
BG = "#0F1219"
MUTED = "#94A3B8"
TEXT = "#F1F5F9"


class FaceSortApp(ctk.CTk):
    """单窗口三步工作流：选目录扫描 → 参考脸查找 → 导出。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("FaceSort 本地人脸整理")
        self.geometry("1180x760")
        self.minsize(960, 640)
        self.configure(fg_color=BG)

        self.cfg = load_config()
        self.indexer = Indexer(db_path(self.cfg))
        self.detector: FaceDetector | None = None
        self.reference_path: str | None = None
        self.matches: list[dict[str, Any]] = []
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._busy = False

        self._build_ui()
        self._refresh_stats()
        self.after(200, self._preload_model_async)

    # ---------- UI ----------
    def _build_ui(self) -> None:
        header = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="FaceSort",
            font=ctk.CTkFont(family="Segoe UI Semibold", size=26),
            text_color=ACCENT,
        ).pack(side="left", padx=(24, 8), pady=18)

        ctk.CTkLabel(
            header,
            text="本地人脸照片整理 · 数据不出本机",
            font=ctk.CTkFont(size=13),
            text_color=MUTED,
        ).pack(side="left", pady=18)

        self.stats_label = ctk.CTkLabel(
            header,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        )
        self.stats_label.pack(side="right", padx=24)

        body = ctk.CTkFrame(self, fg_color=BG)
        body.pack(fill="both", expand=True, padx=18, pady=14)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=3)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color=CARD, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = ctk.CTkFrame(body, fg_color=CARD, corner_radius=16)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left(left)
        self._build_right(right)

        foot = ctk.CTkFrame(self, fg_color=BG, height=36)
        foot.pack(fill="x", padx=18, pady=(0, 10))
        ctk.CTkLabel(
            foot,
            text="提示：合照小人脸可将阈值降到 0.40；遮挡严重可能漏检。测试素材勿提交 Git。",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
        ).pack(side="left")

    def _build_left(self, parent: ctk.CTkFrame) -> None:
        pad = {"padx": 18, "pady": (14, 4)}
        ctk.CTkLabel(
            parent,
            text="① 选择相册并扫描",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", **pad)

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=6)
        self.folder_var = tk.StringVar()
        self.folder_entry = ctk.CTkEntry(
            row,
            textvariable=self.folder_var,
            placeholder_text="选择含照片的文件夹…",
            height=36,
        )
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
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
        ).pack(anchor="w", padx=18, pady=4)

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
        self.scan_btn.pack(fill="x", padx=18, pady=(8, 4))

        self.scan_progress = ctk.CTkProgressBar(parent, progress_color=ACCENT, height=10)
        self.scan_progress.pack(fill="x", padx=18, pady=6)
        self.scan_progress.set(0)
        self.scan_status = ctk.CTkLabel(parent, text="等待扫描", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.scan_status.pack(anchor="w", padx=18)

        ctk.CTkLabel(
            parent,
            text="② 上传参考脸并查找",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=18, pady=(20, 4))

        name_row = ctk.CTkFrame(parent, fg_color="transparent")
        name_row.pack(fill="x", padx=18, pady=4)
        ctk.CTkLabel(name_row, text="标签名", text_color=MUTED, width=56).pack(side="left")
        self.name_var = tk.StringVar(value="PersonA")
        ctk.CTkEntry(name_row, textvariable=self.name_var, height=34).pack(
            side="left", fill="x", expand=True
        )

        ref_row = ctk.CTkFrame(parent, fg_color="transparent")
        ref_row.pack(fill="x", padx=18, pady=8)
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
            text="建议 1 张清晰正脸；\n合照查找可再补侧脸。",
            text_color=MUTED,
            font=ctk.CTkFont(size=12),
            justify="left",
        ).pack(anchor="w")

        thr_row = ctk.CTkFrame(parent, fg_color="transparent")
        thr_row.pack(fill="x", padx=18, pady=6)
        ctk.CTkLabel(thr_row, text="相似度", text_color=MUTED, width=56).pack(side="left")
        self.threshold_var = tk.DoubleVar(value=float(self.cfg.get("match_threshold", 0.45)))
        self.threshold_label = ctk.CTkLabel(thr_row, text=f"{self.threshold_var.get():.2f}", text_color=TEXT, width=40)
        self.threshold_label.pack(side="right")
        self.threshold_slider = ctk.CTkSlider(
            thr_row,
            from_=0.30,
            to=0.85,
            number_of_steps=55,
            variable=self.threshold_var,
            progress_color=ACCENT,
            button_color=ACCENT,
            command=self._on_threshold,
        )
        self.threshold_slider.pack(side="left", fill="x", expand=True, padx=8)

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
        self.find_btn.pack(fill="x", padx=18, pady=(8, 4))

        self.find_progress = ctk.CTkProgressBar(parent, progress_color=ACCENT, height=10)
        self.find_progress.pack(fill="x", padx=18, pady=6)
        self.find_progress.set(0)
        self.find_status = ctk.CTkLabel(parent, text="等待查找", text_color=MUTED, font=ctk.CTkFont(size=12))
        self.find_status.pack(anchor="w", padx=18, pady=(0, 8))

        ctk.CTkLabel(
            parent,
            text="③ 导出结果",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=18, pady=(12, 4))

        self.export_btn = ctk.CTkButton(
            parent,
            text="复制到 output / 标签名",
            height=40,
            fg_color="#334155",
            hover_color="#475569",
            command=self._export,
        )
        self.export_btn.pack(fill="x", padx=18, pady=(4, 16))

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

    # ---------- helpers ----------
    def _on_threshold(self, value: float) -> None:
        self.threshold_label.configure(text=f"{float(value):.2f}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.scan_btn.configure(state=state)
        self.find_btn.configure(state=state)
        self.export_btn.configure(state=state)

    def _refresh_stats(self) -> None:
        stats = self.indexer.stats()
        device = "GPU 就绪" if self.detector and self.detector.using_gpu else (
            self.detector.device_label if self.detector else "模型加载中…"
        )
        if self.detector:
            device = self.detector.device_label
        self.stats_label.configure(
            text=f"{device}  ·  索引 {stats['photos']} 张 / 人脸 {stats['faces']}  ·  标注 {stats['tags']}"
        )

    def _preload_model_async(self) -> None:
        def work() -> None:
            try:
                self._get_detector()
                self.after(0, lambda: (self._refresh_stats(), self.scan_status.configure(text="模型已就绪，可以开始扫描")))
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
        photo = ImageTk.PhotoImage(img)
        self.ref_preview.configure(image=photo, text="")
        self.ref_preview.image = photo  # type: ignore[attr-defined]

    def _clear_gallery(self) -> None:
        for child in self.gallery.winfo_children():
            child.destroy()
        self._thumb_refs.clear()

    def _render_gallery(self, matches: list[dict[str, Any]]) -> None:
        self._clear_gallery()
        self.result_count.configure(text=f"{len(matches)} 张")
        cols = 3
        for i, item in enumerate(matches[:60]):
            path = item["path"]
            frame = ctk.CTkFrame(self.gallery, fg_color=CARD, corner_radius=10)
            frame.grid(row=i // cols, column=i % cols, padx=8, pady=8, sticky="nsew")
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(img)
                self._thumb_refs.append(photo)
                ctk.CTkLabel(frame, image=photo, text="").pack(padx=6, pady=(6, 2))
            except OSError:
                ctk.CTkLabel(frame, text="无法预览", text_color=MUTED).pack(padx=6, pady=20)
            name = Path(path).name
            ctk.CTkLabel(
                frame,
                text=f"{name}\n相似度 {item['similarity']:.3f}",
                text_color=MUTED,
                font=ctk.CTkFont(size=11),
            ).pack(padx=6, pady=(0, 8))

    # ---------- actions ----------
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

                matches = find_matches(
                    self.indexer,
                    reference,
                    threshold,
                    name,
                    progress_callback=on_progress,
                )
                self.matches = matches
                if matches:
                    self.indexer.add_tags(matches)

                def done() -> None:
                    self.find_progress.set(1)
                    self.find_status.configure(
                        text=f"找到 {len(matches)} 张（阈值 {threshold:.2f}），已自动写入标注"
                        if matches
                        else f"未找到匹配，可降低阈值后重试（当前 {threshold:.2f}）"
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
        matches = self.matches or [{"path": t["path"]} for t in self.indexer.get_tagged_photos(name)]
        if not matches:
            messagebox.showwarning("提示", "没有可导出的结果，请先查找")
            return
        dest = export_path(self.cfg)
        result = export_photos(matches, dest, name, mode="copy")
        messagebox.showinfo("导出完成", f"已导出 {result['exported']} 张到：\n{result['dest']}")


def launch() -> None:
    """启动桌面小程序。"""
    app = FaceSortApp()
    app.mainloop()
