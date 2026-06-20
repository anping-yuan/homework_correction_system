# 作业批改系统

基于视觉大模型（Qwen-VL）和文本大模型（DeepSeek）的智能作业批改系统，支持 Web 交互式批改、错题本管理、多页编辑、三路投票判题。

**版本**: 2.2.0

## 功能特性

- 🔍 **三种批改模式**: Vision 视觉批改 / OCR+LLM 文本批改 / Vision+DeepSeek 双阶段三路投票
- 🌐 **Web 交互界面**: 拖拽上传、实时预览、多页编辑选区、环形成绩图
- 📝 **错题本**: 手动保存+去重、学科/难度/复习状态筛选、知识点云、批量操作
- 🖼️ **图像增强**: 光照归一化、双边滤波去噪、CLAHE 对比度增强、USM 锐化
- ⚡ **高并发**: 4 worker 进程 + API 信号量 + MySQL 连接池
- 📊 **FERMAT 评测**: 1800 张手写数学作业基准测试，支持 3 种模式断点续跑
- 🔄 **重做批改**: 支持纠错提示词，卡片内进度覆盖层

## 项目结构

```
homework_correction_system/
├── src/
│   ├── main.py                      # 入口 (api/file/camera/batch 四种模式)
│   ├── api/
│   │   ├── server.py                # FastAPI 服务 (24 端点, 2330行)
│   │   └── static/index.html        # 前端单文件 (2751行)
│   ├── modules/
│   │   ├── llm_grader.py            # 批改引擎 (20方法, 1477行)
│   │   ├── image_processor.py       # 图像处理 (20+方法, 双模式预处理管线)
│   │   ├── annotation_renderer.py   # 批注渲染 (含智能画布扩展)
│   │   ├── db_manager.py            # MySQL 数据管理 (DBUtils连接池, 652行)
│   │   ├── error_notebook.py        # 错题本模块 (DB 委托, 173行)
│   │   ├── aliyun_ocr.py           # 阿里云 OCR + 教育切题 API
│   │   └── camera_capture.py       # 摄像头采集 (with 上下文管理器)
│   └── utils/helpers.py             # 日志、JSON 读写、路径工具
├── config/
│   └── config.json                  # 完整配置 (aliyun/llm/dashscope/mysql/annotation/image_processing)
├── eval/
│   ├── fermat_benchmark.py          # FERMAT 评测脚本 (3模式并发+断点续跑)
│   └── fermat_view.py               # 评测可视化报告
├── data/
│   ├── input/                       # 输入图像
│   ├── output/                      # 批注结果 + 错题图片
│   └── upload_store/                # 上传会话缓存
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

编辑 `config/config.json`，填入 API 密钥：
- `dashscope.api_key` — 通义千问 VL API Key
- `llm.api_key` — DeepSeek API Key
- `aliyun.access_key_id` / `access_key_secret` — 阿里云 OCR
- `mysql` — MySQL 连接信息

### 运行

```bash
# Web API 服务（推荐，4 worker）
python src/main.py --mode api

# 单文件批改
python src/main.py --mode file --input data/input/homework.jpg

# 摄像头批改
python src/main.py --mode camera
```

启动后访问 `http://localhost:8000` 进入交互界面。

## 三种批改模式

| 模式 | 流程 | 准确率(FERMAT) | 耗时 |
|------|------|:------:|:----:|
| **Vision**（默认） | 整页图片 → Qwen-VL 看图识别+批改 | 69.0% | 18.1s |
| **OCR+LLM** | 逐题裁剪 → 阿里云OCR → DeepSeek 文本批改 | 56.0% | 25.6s |
| **Vision+DeepSeek** | Stage1: Qwen-VL 提取文字 → Stage2: DeepSeek 三路投票判题 | 待评测 | — |

### Vision+DeepSeek 三路投票机制
三种不同判题视角并发投票：
1. **数学老师** — 先解题再对比
2. **严格考官** — 假定错误，找证据翻案
3. **数学家教** — 关注理解而非最终答案

多数投票决定 `is_correct`，降低单一模型偏差。

## 核心流程

```
上传图片 → 自动切题 → 预览/手动调整选区
                         ↓
              选择批改模式 (Vision / OCR+LLM / Vision+DeepSeek)
                         ↓
              AI 批改打分 + 图片哈希缓存
                         ↓
              结果卡片（可折叠、重做、添加到错题本）
                         ↓
              错题本 → 筛选/搜索/批量操作/统计
```

## API 端点速览

| 类别 | 端点 | 数量 |
|------|------|:--:|
| 页面 | `/`, `/enhance-view` | 2 |
| 健康检查 | `/health` | 1 |
| 批改流水线 | `/upload`, `/enhance-image`, `/grade-selected`, `/correct`, `/result/{id}`, `/image/{name}` | 6 |
| 重做 | `/api/redo-grading` | 1 |
| 错题本 | CRUD + 批量删除 + 全部标记已复习 + 清空 | 12 |
| 统计 | `/api/stats`, `/api/subjects` | 2 |

## 运行测试

```bash
pytest tests/
```

## FERMAT 基准评测

```bash
# 跑 100 条 Vision 模式评测
python eval/fermat_benchmark.py --mode vision --count 100

# 查看可视化报告
python eval/fermat_view.py
```
