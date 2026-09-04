"""GPU 检测与 ONNX Runtime 执行提供者选择。"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def get_onnx_providers() -> list[str]:
    """返回 ONNX Runtime 当前可用的执行提供者列表。"""
    import onnxruntime as ort

    return list(ort.get_available_providers())


def get_nvidia_gpu_name() -> str | None:
    """通过 nvidia-smi 读取显卡型号，失败时返回 None。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return None


def resolve_execution_backend(use_gpu: bool) -> dict[str, Any]:
    """
    根据配置与硬件选择 ONNX 执行后端。
    返回 providers、ctx_id、device_label 等信息。
    """
    providers = get_onnx_providers()
    gpu_name = get_nvidia_gpu_name()

    if use_gpu and "CUDAExecutionProvider" in providers:
        label = f"GPU · {gpu_name}" if gpu_name else "GPU · CUDA"
        return {
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "ctx_id": 0,
            "device_label": label,
            "using_gpu": True,
            "onnx_providers": providers,
        }

    if use_gpu:
        logger.warning(
            "config.yaml 中 use_gpu=true，但未检测到 CUDAExecutionProvider。"
            "请执行: pip uninstall onnxruntime -y && pip install onnxruntime-gpu"
        )

    return {
        "providers": ["CPUExecutionProvider"],
        "ctx_id": -1,
        "device_label": "CPU",
        "using_gpu": False,
        "onnx_providers": providers,
    }


def gpu_status_message(use_gpu: bool) -> str:
    """供界面展示的 GPU 状态文本。"""
    backend = resolve_execution_backend(use_gpu)
    if backend["using_gpu"]:
        return f"**加速模式：{backend['device_label']}**"
    gpu_name = get_nvidia_gpu_name()
    if use_gpu and gpu_name:
        return (
            f"**加速模式：CPU**（检测到 {gpu_name}，但未启用 CUDA，"
            "请重装 onnxruntime-gpu）"
        )
    return "**加速模式：CPU**"
