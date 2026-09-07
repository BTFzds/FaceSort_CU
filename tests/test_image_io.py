"""image_io 单元测试。"""

from facesort.image_io import short_filename


def test_short_filename_fixed_width() -> None:
    a = short_filename("a.jpg", 24)
    b = short_filename("非常长的中文文件名用来测试界面抖动问题.jpg", 24)
    assert len(a) == 24
    assert len(b) == 24
