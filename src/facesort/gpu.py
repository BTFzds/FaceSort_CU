"""GPU 检测与 ONNX Runtime 执行提供者选择。"""

from __future__ import annotations

import logging
import os
import site
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CUDA_PATH_READY = False
_CUDA_USABLE: bool | None = None


def ensure_cuda_dll_path() -> None:
    """把 pip 安装的 NVIDIA CUDA 运行库目录加入 PATH（Windows 必需）。"""
    global _CUDA_PATH_READY
    if _CUDA_PATH_READY:
        return
    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass
    user = site.getusersitepackages()
    if user:
        roots.append(Path(user))

    added: list[str] = []
    for root in roots:
        for bin_dir in root.glob("nvidia/*/bin"):
            path_str = str(bin_dir)
            if path_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
                added.append(path_str)
    if added:
        logger.info("已注入 CUDA DLL 路径: %s", "; ".join(added))
    _CUDA_PATH_READY = True


def get_onnx_providers() -> list[str]:
    """返回 ONNX Runtime 当前可用的执行提供者列表。"""
    ensure_cuda_dll_path()
    import onnxruntime as ort

    return list(ort.get_available_providers())


def cuda_provider_usable() -> bool:
    """真正尝试创建 CUDA session，避免仅列表可用但 DLL 缺失。"""
    global _CUDA_USABLE
    if _CUDA_USABLE is not None:
        return _CUDA_USABLE
    ensure_cuda_dll_path()
    if "CUDAExecutionProvider" not in get_onnx_providers():
        _CUDA_USABLE = False
        return False
    try:
        from onnx import TensorProto, helper
        import onnxruntime as ort

        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph([node], "t", [x], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _CUDA_USABLE = "CUDAExecutionProvider" in session.get_providers()
        return _CUDA_USABLE
    except Exception as exc:  # noqa: BLE001
        logger.warning("CUDA 不可用，将回退 CPU: %s", exc)
        _CUDA_USABLE = False
        return False


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
    ensure_cuda_dll_path()
    providers = get_onnx_providers()
    gpu_name = get_nvidia_gpu_name()

    if use_gpu and "CUDAExecutionProvider" in providers and cuda_provider_usable():
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
            "config.yaml 中 use_gpu=true，但 CUDA 未能实际加载。"
            "请确认已安装 onnxruntime-gpu==1.20.2 与 nvidia-cublas-cu12 等运行库。"
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
            f"**加速模式：CPU**（检测到 {gpu_name}，但 CUDA 运行库未就绪，"
            "已自动回退；功能不受影响）"
        )
    return "**加速模式：CPU**"
