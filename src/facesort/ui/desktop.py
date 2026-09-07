"""FaceSort 桌面小程序界面（CustomTkinter，不依赖浏览器）。"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
from PIL import Image

from facesort.config import db_path, export_path, load_config
from facesort.detector import FaceDetector
from facesort.export import export_photos
from facesort.image_io import imread_bgr, short_filename
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
MAX_REFS = 5


class FaceSortApp(ctk.CTk):
    """扫描 → 多参考脸查找 → 预览 → 导出。"""

    def __init__(self) -> None:
        super().__init__()
        self.title("FaceSort 本地人脸整理")
        self.geometry("1280x840")
        self.minsize(1180, 780)
        self.configure(fg_color=BG)

        self.cfg = load_config()
        self.indexer = Indexer(db_path(self.cfg))
        self.detector: FaceDetector | None = None
        self.reference_paths: list[str] = []
        self.matches: list[dict[str, Any]] = []
        self._thumb_refs: list[ctk.CTkImage] = []
        self._ref_thumbs: list[ctk.CTkImage] = []
        self._busy = False
        self._default_threshold = float(self.cfg.get("match_threshold", 0.45))

        self._build_ui()
        self._hook_drag_drop()
        self._refresh_stats()
        self._set_global_progress(0, "就绪")
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

        # 全局进度条（任何阶段都可见）
        global_bar = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=40)
        global_bar.pack(fill="x")
        global_bar.pack_propagate(False)
        ctk.CTkLabel(global_bar, text="总进度", text_color=MUTED, width=50).pack(side="left", padx=(16, 6))
        self.global_progress = ctk.CTkProgressBar(global_bar, progress_color=ACCENT, height=10, width=220)
        self.global_progress.pack(side="left", padx=4)
        self.global_progress.set(0)
        self.global_status = ctk.CTkLabel(
            global_bar,
            text="就绪",
            text_color=MUTED,
            font=ctk.CTkFont(size=11),
            width=700,
            anchor="w",
        )
        self.global_status.pack(side="left", padx=10, fill="x", expand=True)

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
        body.pack(fill="both", expand=True, padx=14, pady=8)
        body.grid_columnconfigure(0, weight=0, minsize=430)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14, width=430)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_propagate(False)
        right = ctk.CTkFrame(body, fg_color=CARD, corner_radius=14)
        right.grid(row=0, column=1, sticky="nsew")

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent: ctk.CTkFrame) -> None:
        pad_x = 12
        ctk.CTkLabel(
            parent, text="① 扫描相册", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT
        ).pack(anchor="w", padx=pad_x, pady=(8, 2))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=pad_x, pady=2)
        self.folder_var = tk.StringVar()
        ctk.CTkEntry(
            row, textvariable=self.folder_var, placeholder_text="选择含照片的文件夹…", height=30
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
        ).pack(side="left")

        self.scan_btn = ctk.CTkButton(
            parent,
            text="开始扫描",
            height=32,
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
        # 固定高度/宽度，避免文件名长短导致抽动
        self.scan_status = ctk.CTkLabel(
            parent,
            text="等待扫描".ljust(42),
            text_color=MUTED,
            font=ctk.CTkFont(size=11, family="Consolas"),
            height=28,
            width=400,
            anchor="w",
        )
        self.scan_status.pack(anchor="w", padx=pad_x)
        self.scan_status.pack_propagate(False)

        ctk.CTkFrame(parent, fg_color="#334155", height=1).pack(fill="x", padx=pad_x, pady=6)

        ctk.CTkLabel(
            parent, text="② 参考脸查找（可多张）", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT
        ).pack(anchor="w", padx=pad_x, pady=(0, 2))

        name_row = ctk.CTkFrame(parent, fg_color="transparent")
        name_row.pack(fill="x", padx=pad_x, pady=2)
        ctk.CTkLabel(name_row, text="人名", text_color=MUTED, width=40).pack(side="left")
        self.name_var = tk.StringVar(value="PersonA")
        ctk.CTkEntry(name_row, textvariable=self.name_var, height=28).pack(
            side="left", fill="x", expand=True
        )
        ctk.CTkLabel(
            parent,
            text="标注：文件夹_人名｜多张参考脸可提高合照召回",
            text_color=MUTED,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", padx=pad_x)

        # 拖拽投放区
        self.drop_zone = ctk.CTkFrame(parent, fg_color="#0B1220", corner_radius=10, height=92)
        self.drop_zone.pack(fill="x", padx=pad_x, pady=6)
        self.drop_zone.pack_propagate(False)
        self.ref_strip = ctk.CTkFrame(self.drop_zone, fg_color="transparent")
        self.ref_strip.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right_btns = ctk.CTkFrame(self.drop_zone, fg_color="transparent", width=110)
        right_btns.pack(side="right", padx=6, pady=6)
        ctk.CTkButton(
            right_btns,
            text="添加参考脸",
            width=100,
            height=28,
            fg_color="#334155",
            hover_color="#475569",
            command=self._pick_references,
        ).pack(pady=(4, 4))
        ctk.CTkButton(
            right_btns,
            text="清空参考",
            width=100,
            height=26,
            fg_color="#475569",
            hover_color="#64748B",
            command=self._clear_references,
        ).pack()
        self.drop_hint = ctk.CTkLabel(
            self.ref_strip,
            text=f"拖拽图片到此处（最多 {MAX_REFS} 张）\n或点右侧添加",
            text_color=MUTED,
            font=ctk.CTkFont(size=11),
            justify="left",
        )
        self.drop_hint.pack(anchor="w", padx=4, pady=16)

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
            height=32,
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
            parent,
            text="等待查找".ljust(42),
            text_color=MUTED,
            font=ctk.CTkFont(size=11, family="Consolas"),
            height=28,
            width=400,
            anchor="w",
        )
        self.find_status.pack(anchor="w", padx=pad_x, pady=(0, 6))
        self.find_status.pack_propagate(False)

    def _build_right(self, parent: ctk.CTkFrame) -> None:
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(
            top, text="匹配结果预览", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT
        ).pack(side="left")
        self.result_count = ctk.CTkLabel(top, text="0 张", text_color=ACCENT)
        self.result_count.pack(side="right")
        self.gallery = ctk.CTkScrollableFrame(parent, fg_color="#0B1220", corner_radius=10)
        self.gallery.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self._show_gallery_placeholder("查找后在此显示缩略图")

    def _hook_drag_drop(self) -> None:
        try:
            import windnd

            def on_drop(files: list) -> None:
                paths: list[str] = []
                for item in files:
                    if isinstance(item, bytes):
                        try:
                            text = item.decode("utf-8")
                        except UnicodeDecodeError:
                            text = item.decode("gbk", errors="ignore")
                    else:
                        text = str(item)
                    paths.append(text)
                self.after(0, lambda: self._add_reference_paths(paths))

            windnd.hook_dropfiles(self.drop_zone, func=on_drop)
            # 也允许拖到整窗左侧
            windnd.hook_dropfiles(self, func=on_drop)
        except Exception as exc:  # noqa: BLE001
            logger.warning("拖拽不可用: %s", exc)

    def _set_global_progress(self, ratio: float, text: str) -> None:
        self.global_progress.set(max(0.0, min(1.0, ratio)))
        self.global_status.configure(text=short_filename(text, 80).rstrip())

    def _set_fixed_status(self, label: ctk.CTkLabel, text: str) -> None:
        label.configure(text=short_filename(text, 42))

    def _show_gallery_placeholder(self, text: str) -> None:
        self._clear_gallery()
        ctk.CTkLabel(self.gallery, text=text, text_color=MUTED, font=ctk.CTkFont(size=13)).pack(
            expand=True, pady=60
        )
        self.result_count.configure(text="0 张")
        self.after(30, self._scroll_gallery_top)

    def _on_threshold(self, value: float) -> None:
        self.threshold_label.configure(text=f"{float(value):.2f}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for btn in (self.scan_btn, self.find_btn, self.export_btn, self.reset_btn):
            btn.configure(state=state)

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
                self._set_global_progress(0.05, "正在加载人脸模型…")
                self._get_detector()

                def done() -> None:
                    self._refresh_stats()
                    gpu = "CUDA 已启用" if self.detector and self.detector.using_gpu else "CPU"
                    self._set_fixed_status(self.scan_status, f"模型就绪（{gpu}）")
                    self._set_global_progress(0, f"就绪（{gpu}）")

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

    def _pick_references(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择参考照片（可多选）",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.webp *.bmp"), ("全部", "*.*")],
        )
        if paths:
            self._add_reference_paths(list(paths))

    def _add_reference_paths(self, paths: list[str]) -> None:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        for p in paths:
            path = Path(p.strip().strip("{}"))
            if path.suffix.lower() not in exts:
                continue
            if not path.is_file():
                continue
            sp = str(path.resolve())
            if sp in self.reference_paths:
                continue
            if len(self.reference_paths) >= MAX_REFS:
                messagebox.showinfo("提示", f"最多 {MAX_REFS} 张参考脸")
                break
            self.reference_paths.append(sp)
        self._refresh_ref_strip()

    def _clear_references(self) -> None:
        self.reference_paths.clear()
        self._refresh_ref_strip()

    def _refresh_ref_strip(self) -> None:
        for child in self.ref_strip.winfo_children():
            child.destroy()
        self._ref_thumbs.clear()
        if not self.reference_paths:
            self.drop_hint = ctk.CTkLabel(
                self.ref_strip,
                text=f"拖拽图片到此处（最多 {MAX_REFS} 张）\n或点右侧添加",
                text_color=MUTED,
                font=ctk.CTkFont(size=11),
                justify="left",
            )
            self.drop_hint.pack(anchor="w", padx=4, pady=16)
            return
        for path in self.reference_paths:
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((64, 64))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                self._ref_thumbs.append(ctk_img)
                ctk.CTkLabel(self.ref_strip, image=ctk_img, text="").pack(side="left", padx=3)
            except OSError:
                ctk.CTkLabel(self.ref_strip, text="?", width=40, text_color=MUTED).pack(
                    side="left", padx=3
                )

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
        self.reference_paths.clear()
        self._refresh_ref_strip()
        self.name_var.set("PersonA")
        self.threshold_var.set(self._default_threshold)
        self.threshold_label.configure(text=f"{self._default_threshold:.2f}")
        self.incremental_var.set(True)
        self.scan_progress.set(0)
        self.find_progress.set(0)
        self._set_fixed_status(self.scan_status, "已重置，可重新扫描")
        self._set_fixed_status(self.find_status, "等待查找")
        self.export_hint.configure(text="查找后可导出")
        self._show_gallery_placeholder("已重置，查找后在此显示结果")
        self._set_global_progress(0, "已重置")

        if clear_db:
            db = db_path(self.cfg)
            try:
                if db.exists():
                    db.unlink()
            except OSError as exc:
                messagebox.showerror("重置失败", f"无法删除索引库：{exc}")
                return
            self.indexer = Indexer(db)
            self._set_fixed_status(self.scan_status, "已重置并清空索引")

        self._refresh_stats()
        messagebox.showinfo("完成", "界面已重置" + ("，索引库已清空" if clear_db else ""))

    def _render_gallery(self, matches: list[dict[str, Any]]) -> None:
        self._clear_gallery()
        self.result_count.configure(text=f"{len(matches)} 张")
        self._scroll_gallery_top()
        if not matches:
            self._show_gallery_placeholder("没有匹配结果\n可降低相似度或增加参考脸")
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
        self._set_fixed_status(self.scan_status, "准备扫描…")
        self._set_global_progress(0.01, "开始扫描相册…")

        def work() -> None:
            try:
                detector = self._get_detector()

                def on_progress(cur: int, total: int, msg: str) -> None:
                    ratio = cur / total if total else 0

                    def update(r: float = ratio, c: int = cur, t: int = total, m: str = msg) -> None:
                        self.scan_progress.set(r)
                        self._set_fixed_status(self.scan_status, f"{c}/{t} {m}")
                        self._set_global_progress(r * 0.95, f"扫描 {c}/{t} {m}")

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
                        f"完成 共{result['total_files']} 新扫{result['scanned']} "
                        f"跳过{result['skipped']} 人脸{result['faces']} "
                        f"未读{result.get('unread', 0)} 无人脸{result.get('no_face', 0)}"
                    )
                    self._set_fixed_status(self.scan_status, msg)
                    self._set_global_progress(1, msg)
                    if result["faces"] == 0 and result["scanned"] > 0:
                        messagebox.showwarning(
                            "未检测到人脸",
                            "已扫描文件但人脸数为 0。\n"
                            "可能原因：图片无人脸、截图/便签、或格式特殊。\n"
                            f"无法读取：{result.get('unread', 0)} 张；"
                            f"可读但无人脸：{result.get('no_face', 0)} 张。\n"
                            "请取消增量后重试，或换含清晰人脸的相册。",
                        )
                    self._refresh_stats()
                    self._set_busy(False)

                self.after(0, done)
            except Exception as exc:  # noqa: BLE001
                logger.exception("扫描失败")
                self.after(
                    0,
                    lambda: (
                        messagebox.showerror("扫描失败", str(exc)),
                        self._set_busy(False),
                        self._set_global_progress(0, "扫描失败"),
                    ),
                )

        threading.Thread(target=work, daemon=True).start()

    def _start_find(self) -> None:
        if self._busy:
            return
        name = self.name_var.get().strip() or "PersonA"
        if not self.reference_paths:
            messagebox.showwarning("提示", "请先添加至少 1 张参考脸（可拖拽）")
            return

        self._set_busy(True)
        self.find_progress.set(0)
        self._set_fixed_status(self.find_status, "提取参考脸…")
        self._set_global_progress(0.02, "提取参考脸…")
        self._show_gallery_placeholder("查找中…")
        threshold = float(self.threshold_var.get())
        ref_paths = list(self.reference_paths)

        def work() -> None:
            try:
                detector = self._get_detector()
                all_faces: list[dict[str, Any]] = []
                total_refs = len(ref_paths)
                for i, ref in enumerate(ref_paths, start=1):
                    ratio = i / max(total_refs, 1) * 0.25

                    def upd(
                        r: float = ratio,
                        idx: int = i,
                        n: int = total_refs,
                        fn: str = Path(ref).name,
                    ) -> None:
                        self.find_progress.set(r)
                        self._set_fixed_status(self.find_status, f"提取参考脸 {idx}/{n} {short_filename(fn, 18)}")
                        self._set_global_progress(r, f"提取参考脸 {idx}/{n}")

                    self.after(0, upd)
                    img = imread_bgr(ref)
                    if img is None:
                        continue
                    all_faces.extend(detector.detect_from_array(img))

                reference = detector.reference_embedding(all_faces)
                if reference is None:
                    raise RuntimeError("参考照未检测到人脸，请换更清晰的正脸（可多张）")

                def on_progress(cur: int, total: int, msg: str) -> None:
                    # 参考脸提取占 0~25%，比对占 25%~100%
                    ratio = 0.25 + (cur / total if total else 0) * 0.75

                    def update(r: float = ratio, m: str = msg, c: int = cur, t: int = total) -> None:
                        self.find_progress.set(r)
                        self._set_fixed_status(self.find_status, f"{c}/{t} {m}")
                        self._set_global_progress(r, f"比对 {c}/{t}")

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
                        self._set_fixed_status(
                            self.find_status,
                            f"找到{len(matches)}张 标注:{','.join(sample_tags)}",
                        )
                        self._set_global_progress(1, f"查找完成：{len(matches)} 张")
                    else:
                        self._set_fixed_status(
                            self.find_status,
                            f"未匹配，可降阈值或加参考脸({threshold:.2f})",
                        )
                        self._set_global_progress(1, "查找完成：0 张")
                    self._render_gallery(matches)
                    self._refresh_stats()
                    self._set_busy(False)

                self.after(0, done)
            except Exception as exc:  # noqa: BLE001
                logger.exception("查找失败")
                self.after(
                    0,
                    lambda: (
                        messagebox.showerror("查找失败", str(exc)),
                        self._set_busy(False),
                        self._set_global_progress(0, "查找失败"),
                    ),
                )

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
            f"已导出 {result['exported']} 张\n子文件夹数：{result.get('folders', 1)}\n位置：{result['dest']}",
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
