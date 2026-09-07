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
3. 将弹出 **桌面窗口**（不再打开浏览器）

## 使用步骤

| 步骤 | 操作 |
|------|------|
| ① 扫描 | 点「浏览」选相册文件夹 →「开始扫描」（有进度条） |
| ② 查找 | 填标签名、选参考脸 → 调阈值 →「开始查找」（右侧预览） |
| ③ 导出 | 「复制到 output / 标签名」 |

查找成功后会自动写入标注。合照小人脸建议阈值 **0.40~0.50**，并确保已全量扫描（可取消「增量」强制重扫）。

## GPU 加速（NVIDIA 显卡）

默认已启用 **GPU 加速**（`onnxruntime-gpu`），适用于 RTX 3060 / 4060 等 NVIDIA 显卡。

1. 确保已安装 [NVIDIA 显卡驱动](https://www.nvidia.cn/drivers/)
2. 双击 `run.bat` 即可，界面顶部会显示 `加速模式：GPU · 你的显卡型号`
3. `config.yaml` 中 `use_gpu: true`（默认已开启）

**无 NVIDIA 显卡时**，改用 CPU 版：

```bash
pip uninstall onnxruntime-gpu onnxruntime -y
pip install -r requirements-cpu.txt
```

并在 `config.yaml` 中设置 `use_gpu: false`。

## 隐私与安全

- 所有推理在本地完成，不调用任何云端 API
- Web 界面仅绑定 `127.0.0.1`，不对外网开放
- 源照片默认只读；仅索引库（`data/`）和导出目录（`output/`）会写入
- 首次运行会从 InsightFace 官方源下载模型到 `~/.insightface/`
- **请勿**将真实工作照片、内部文档、客户资料提交到 Git；`data/`、`output/`、`.venv/` 已在 `.gitignore` 中排除
- 对外分享仓库时，标签与导出目录请使用通用占位姓名（如 `PersonA`），避免写入可识别的真实身份信息

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
