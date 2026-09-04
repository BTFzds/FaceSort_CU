"""FaceSort 入口：启动本地 Gradio 界面。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 将 src 加入模块搜索路径，方便直接 python main.py 运行
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from facesort.ui.app import launch  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    launch()


if __name__ == "__main__":
    main()
