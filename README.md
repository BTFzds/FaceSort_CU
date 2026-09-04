# FaceSort

**本地人脸照片整理工具** — 从大量文件夹中精准找出某个人的全部照片，打标签并分类导出。

> 100% 本地运行，照片不会上传到任何服务器。

## 功能

- 递归扫描本地文件夹，自动检测人脸并建立索引
- 上传参考照片，一键找出该人物的所有相关照片
- 保存人物标签，复制或硬链接到分类文件夹
- 增量扫描，重复运行时跳过未变化的文件

## 快速开始（Windows）

1. 安装 [Python 3.10+](https://www.python.org/downloads/)
2. 双击运行 **`run.bat`**
3. 浏览器打开 http://127.0.0.1:7860

## 快速开始（Mac / Linux）

```bash
chmod +x run.sh
./run.sh
```

## 手动安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 使用步骤

| 步骤 | 操作 |
|------|------|
| ① 扫描照片 | 输入照片根目录路径，点击「开始扫描」 |
| ② 查找人物 | 输入姓名，上传 1~3 张清晰正脸参考照，调整阈值后查找 |
| ③ 标注导出 | 保存标签，选择复制或硬链接导出到 `output/姓名/` |

### 相似度阈值建议

| 阈值 | 效果 |
|------|------|
| 0.45 ~ 0.55 | 召回率高，可能有误报 |
| 0.55 ~ 0.65 | **推荐**，平衡精度与召回 |
| 0.65+ | 精度高，可能漏掉侧脸/模糊照 |

## GPU 加速（可选）

有 NVIDIA 显卡时，编辑 `requirements.txt`：

```
# 注释掉 onnxruntime，改用：
onnxruntime-gpu>=1.17.0
```

并在 `config.yaml` 中确认 `use_gpu: true`。

## 隐私与安全

- 所有推理在本地完成，不调用任何云端 API
- Web 界面仅绑定 `127.0.0.1`，不对外网开放
- 源照片默认只读；仅索引库（`data/`）和导出目录（`output/`）会写入
- 首次运行会从 InsightFace 官方源下载模型到 `~/.insightface/`

## 项目结构

```
FaceSort_CU/
├── main.py              # 入口
├── run.bat / run.sh     # 一键启动
├── config.yaml          # 配置
├── requirements.txt
├── src/facesort/
│   ├── scanner.py       # 目录扫描
│   ├── detector.py      # InsightFace 检测
│   ├── matcher.py       # 相似度匹配
│   ├── indexer.py       # SQLite 索引
│   ├── pipeline.py      # 扫描编排
│   ├── export.py        # 导出
│   └── ui/app.py        # Gradio 界面
└── data/                # 本地索引（自动生成）
```

## 常见问题

**Q: 首次运行很慢？**  
A: 需要下载约 300MB 的 AI 模型，仅需一次。

**Q: 找不到某人？**  
A: 降低阈值，或上传多张不同角度的参考脸。

**Q: 误报太多？**  
A: 提高阈值，或换更清晰的参考正脸。

## License

MIT
