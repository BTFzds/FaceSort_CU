"""FaceSort 入口：启动本地桌面小程序。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facesort.gpu import ensure_cuda_dll_path  # noqa: E402

ensure_cuda_dll_path()

from facesort.ui.desktop import launch  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    launch()


if __name__ == "__main__":
    main()
