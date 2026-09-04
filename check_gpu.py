"""启动前检查 GPU / CUDA 是否可用。"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from facesort.gpu import get_nvidia_gpu_name, get_onnx_providers, resolve_execution_backend  # noqa: E402


def main() -> None:
    gpu_name = get_nvidia_gpu_name()
    providers = get_onnx_providers()
    backend = resolve_execution_backend(use_gpu=True)

    print("=" * 50)
    print("  FaceSort GPU 检测")
    print("=" * 50)
    print(f"显卡:       {gpu_name or '未检测到 NVIDIA 显卡'}")
    print(f"ONNX 提供者: {', '.join(providers)}")
    print(f"将使用:     {backend['device_label']}")
    print("=" * 50)

    if backend["using_gpu"]:
        print("GPU 加速已就绪，可以运行 run.bat")
    else:
        print("当前为 CPU 模式。若你有 NVIDIA 显卡，请执行：")
        print("  pip uninstall onnxruntime onnxruntime-gpu -y")
        print("  pip install onnxruntime-gpu>=1.18.0")


if __name__ == "__main__":
    main()
