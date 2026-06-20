# 作业批改系统 — 项目详解手册

> 本文档对当前系统（v2.2.0）进行完整讲解，涵盖所有模块、API、前端功能和评测体系。

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 目录结构](#2-目录结构)
- [3. 核心数据流](#3-核心数据流)
- [4. 主程序入口 —— `main.py`](#4-主程序入口--srcmainpy)
- [5. 批改引擎 —— `llm_grader.py`](#5-批改引擎--llm_graderpy)
- [6. 图像预处理 —— `image_processor.py`](#6-图像预处理--image_processorpy)
- [7. 阿里云OCR —— `aliyun_ocr.py`](#7-阿里云ocr--aliyun_ocrpy)
- [8. 批注渲染 —— `annotation_renderer.py`](#8-批注渲染--annotation_rendererpy)
- [9. 数据库管理 —— `db_manager.py`](#9-数据库管理--db_managerpy)
- [10. 错题本 —— `error_notebook.py`](#10-错题本--error_notebookpy)
- [11. API 服务 —— `server.py`](#11-api-服务--srcapiserverpy)
- [12. 前端界面 —— `index.html`](#12-前端界面--indexhtml)
- [13. FERMAT 评测体系](#13-fermat-评测体系)
- [14. 配置文件](#14-配置文件)
- [15. 模块完成度一览](#15-模块完成度一览)

---

## 1. 项目概述

### 这个系统做什么？

模拟老师批改作业的完整流程，将其自动化：

```
拍照/上传图片 → 图像增强 → AI 识别+批改+打分 → 在原图上画批注 → 结果展示 + 错题本
```

核心差异化：
- **不再强制 OCR+LLM 两步走**，视觉模型直接看图批改是默认模式
- **三种批改模式可选**，适应不同作业类型
- **Web 全功能界面**，拖拽上传、多页编辑、实时结果
- **MySQL 持久化**，批改记录和错题本统一存储

### 适用场景

- 老师批量批改纸质作业（拍照 → 自动批改 → 错题本）
- 在线教育平台 API 集成
- 手写数学作业基准评测（FERMAT）

---

## 2. 目录结构

```
homework_correction_system/
├── src/
│   ├── main.py                      # 入口 (api/file/camera/batch 四种模式)
│   ├── api/
│   │   ├── server.py                # FastAPI 服务 (24 端点, 2330行)
│   │   └── static/index.html        # 前端单文件 (2751行, 4个Tab页)
│   ├── modules/
│   │   ├── llm_grader.py            # 批改引擎 (20方法, 1477行)
│   │   ├── image_processor.py       # 图像处理 (20+方法)
│   │   ├── annotation_renderer.py   # 批注渲染
│   │   ├── db_manager.py            # MySQL (DBUtils连接池, 652行)
│   │   ├── error_notebook.py        # 错题本 (173行)
│   │   ├── aliyun_ocr.py           # 阿里云OCR
│   │   └── camera_capture.py       # 摄像头采集
│   └── utils/helpers.py             # 日志、JSON读写、路径工具
├── config/
│   └── config.json                  # 全部配置
├── eval/
│   ├── fermat_benchmark.py          # 评测脚本
│   └── fermat_view.py               # 可视化报告
├── data/
│   ├── input/                       # 输入图像
│   ├── output/                      # 输出结果
│   └── upload_store/                # 上传会话缓存
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

---

## 3. 核心数据流

```
┌──────────────────────────────────────────────────────────────────┐
│ ① 上传 (POST /upload)                                            │
│   图片 → 阿里云教育切题 API → page_images + page_questions          │
│   → 存入 upload_store (过期自动清理, >2h)                         │
└────────────┬─────────────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌──────────┐   ┌──────────────┐
│ 快速批改  │   │ 编辑选区      │
│ (一键)   │   │ (多页翻页调整) │
└────┬─────┘   └──────┬───────┘
     │                │
     └───────┬────────┘
             ▼
┌──────────────────────────────────────────────────────────────────┐
│ ② 批改 (POST /grade-selected)                                    │
│   根据 grading_mode 调用不同方法:                                  │
│   · vision → grade_full_page()     Qwen-VL 看图批改               │
│   · ocr_llm → 逐题OCR + grade()    DeepSeek 文本批改              │
│   · vision_deepseek → grade_vision_deepseek()  三路投票            │
│   图片哈希缓存: 相同图+相同模式跳过重复批改                          │
└────────────┬─────────────────────────────────────────────────────┘
             ▼
┌──────────────────────────────────────────────────────────────────┐
│ ③ 结果展示                                                        │
│   环形成绩图 + 结果卡片 (可折叠/重做/加入错题本)                     │
│   可选: 图像增强预览 (原始 vs 增强并排)                             │
└────────────┬─────────────────────────────────────────────────────┘
             ▼ (手动勾选保存)
┌──────────────────────────────────────────────────────────────────┐
│ ④ 错题本 (MySQL grading_records, in_notebook=1)                   │
│   筛选 (学科/复习/难度/关键词) → 分页列表 → 批量操作                │
│   统计页: 概览卡片 + SVG环形图 + 知识点云 (可点击跳转错题本)         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. 主程序入口 —— `src/main.py`

### 四种运行模式

| 模式 | 启动命令 | 适用场景 |
|------|----------|----------|
| **api** | `python src/main.py --mode api` | 启动 Web 服务 (4 workers) |
| **file** | `python src/main.py --mode file --input 图片.jpg` | 处理已有图片 |
| **camera** | `python src/main.py --mode camera` | 摄像头拍照批改 |
| **batch** | `python src/main.py --mode batch --input 目录/` | 批量处理图片 |

### API 模式关键代码

```python
# uvicorn 多 worker 启动
uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, workers=4)
```

---

## 5. 批改引擎 —— `llm_grader.py`

**角色**: AI 老师，同时调用视觉模型和文本模型完成批改。**1477行，20方法，已全部实现。**

### 模型配置

| 模型 | 厂商 | 用途 | 默认值 |
|------|------|------|--------|
| Qwen-VL Max | 通义千问 DashScope | 视觉批改、看图识字 | `qwen-vl-max` |
| DeepSeek V4 Pro | DeepSeek | 文本批改、逻辑推理 | `deepseek-v4-pro` |

### 三种批改模式的方法入口

| 模式 | 方法 | 流程 |
|------|------|------|
| Vision | `grade_full_page(image_path)` | 整页图 → Qwen-VL → JSON 结果 |
| OCR+LLM | `grade(ocr_text)` | OCR 文本 → DeepSeek → JSON 结果 |
| Vision+DeepSeek | `grade_vision_deepseek(image_path)` | Qwen-VL 提取文字 → DeepSeek 三路投票 |

### 核心批改方法详解

#### `grade_full_page(image_path)` — 整页视觉批改
```python
# 输入: 作业图片路径
# 输出: {page_text, questions: [{question_no, is_correct, score, max_score,
#         comment, subject, topic, difficulty, sub_questions: [...]}]}
# temperature=0.1, max_tokens=4096, timeout=120s
```
要求 VL 模型找出所有题目，以嵌套 sub_questions 返回每道题的独立批改结果。

#### `grade_vision_deepseek(image_path)` — 双阶段+三路投票
```
Stage 1: Qwen-VL 提取题目和答案文字（不做判断，只做 OCR）
Stage 2: DeepSeek 三路并发判题:
  ┌─ Voter 1 "数学老师" — 先解题再对比答案，每题5分
  ├─ Voter 2 "严格考官" — 假定错误，找到明确证据才能判对
  └─ Voter 3 "数学家教" — 关注概念理解而非最终答案
→ 多数投票决定 is_correct
→ 投票全部失败时单次回退调用
→ _enforce_consistency() 修正矛盾
```

### 重做方法（4个）

| 方法 | 用途 |
|------|------|
| `redo_question(image_path, prev, hint, sub_label)` | **统一入口** — Vision 69% 优于 OCR+LLM 56%，始终走 Vision |
| `_redo_vision(image_path, prev, hint, sub_label)` | 看图 + 纠错指令重新批改 |
| `_redo_text(prev, hint, sub_label)` | OCR+LLM 文本模式重做 |
| `_redo_vision_deepseek(image_path, prev, hint, sub_label)` | Vision+DeepSeek 双阶段重做 |

### 辅助方法

| 方法 | 用途 |
|------|------|
| `_prepare_image_for_api(image_path)` | 图片预处理 → 压缩 → base64 data URL |
| `_call_with_retry(fn)` [static] | API 重试包装器: Semaphore(5) 限流 + 指数退避(2s/4s) |
| `_enforce_consistency(result)` [static] | 修正矛盾: 父题分数从子题重算、判错但满分归零 |
| `_extract_json(text)` | **7策略 JSON 提取引擎**: 直接解析→去markdown→搜索花括号→修复截断→引号修正→尾逗号→组合修复 |
| `_build_redo_prompt(prev, hint, sub_label)` | 构建纠错指令注入 system prompt |
| `batch_grade(questions)` | 批量批改 |
| `calculate_total_score(results)` | 算总分 |
| `generate_feedback(result)` | LLM 生成鼓励评语 |

### 学科自动检测

`auto_detect_subject(question_text, model_subject)` — 模块级函数：
- 6 大学科关键词库（数学/物理/化学/生物/语文/英语）
- 每个学科独立打分 + 交叉验证
- **强制修正**: 模型误判"语文"但含≥3个数学特征 → 自动修正为"数学"
- 所有批改方法的结果自动调用此函数验证

---

## 6. 图像预处理 —— `image_processor.py`

**角色**: 修图师。**20+方法，已全部实现。**

### 基础处理

| 方法 | 功能 |
|------|------|
| `to_grayscale(image)` | 灰度化 |
| `binarize(image, method)` | 二值化 (otsu/adaptive/固定阈值) |
| `denoise(image, strength)` | 去噪 (灰度用 fastNlMeansDenoising, 彩色用 DenoisingColored) |
| `deskew(image)` | 纠偏 |
| `resize(image, width, height)` | 缩放 |
| `sharpen(image, strength)` | 锐化 |
| `enhance_contrast(image)` | CLAHE 对比度增强 |

### 进阶处理

| 方法 | 功能 |
|------|------|
| `bilateral_denoise(image)` | 双边滤波 — 保边去噪，保护文字笔迹边缘 |
| `normalize_illumination(image)` | 光照归一化 — 图像除以自身的模糊版本，消除光照不均 |
| `unsharp_mask(image)` | USM 锐化 — 比简单 kernel 锐化更自然 |
| `msrcr(image)` | MSRCR 多尺度 Retinex — 精细光照归一化 |
| `sauvola_binarize(image)` | Sauvola 自适应二值化 — 手写文字专用 |
| `morphological_clean(image)` | 形态学开运算 — 修复断笔 + 去背景噪点 |
| `high_pass_enhance(image)` | 高反差保留 — 比 USM 更自然的锐化方式 |
| `linear_contrast_stretch(image)` | 线性对比度拉伸 |

### 双模式预处理管线

| 管线 | 流程 | 用途 |
|------|------|------|
| `enhance_for_vlm(image)` | 光照归一化 → 双边滤波 → CLAHE → USM 锐化 | 送视觉模型前预处理 |
| `enhance_document(image)` | 同上流程 | 给人看的文档增强 |

### 几何变换

| 方法 | 功能 |
|------|------|
| `find_paper_contour(image)` | 检测试卷纸张四角 |
| `warp_perspective(image)` | 透视变换校正 |
| `auto_correct_and_crop(image)` | 自动检测纸张并校正裁剪 |
| `split_regions(image, positions)` | 按坐标切出题目区域 |
| `stitch_vertical(image_paths)` [static] | 多图纵向拼接（中文路径安全） |

---

## 7. 阿里云OCR —— `aliyun_ocr.py`

**角色**: 文字识别。用于 OCR+LLM 模式。**已全部实现。**

| 方法 | 功能 |
|------|------|
| `recognize_text(image_path)` | 通用文字识别（印刷体+手写体） |
| `recognize_edu_paper_cut(image_path)` | 教育试卷切题 — 自动检测题目区域 |
| `get_question_regions(image_path)` | 获取题目区域坐标列表 |
| `parse_result(ocr_result)` | 原始结果转纯文本 |

---

## 8. 批注渲染 —— `annotation_renderer.py`

**角色**: 批改标记员。在图片上画批注。**已全部实现。**

| 方法 | 功能 |
|------|------|
| `draw_correct_mark(image, position, radius)` | 绿色对勾 ✓ |
| `draw_wrong_mark(image, position, radius)` | 红色叉号 ✗ |
| `draw_text_annotation(image, text, position, color)` | 文字批注（中文, 半透明背景, 自动换行） |
| `draw_bounding_box(image, box, color, thickness)` | 边界框 |
| `render_score(image, score, total, position)` | 分数显示（按比例绿/橙/红） |
| `render_all(image, corrections)` | 渲染全部批注 |
| `render_all_with_regions(image, corrections, regions)` | 按区域渲染 — **自动扩展画布**以容纳评语，不遮挡原图 |

---

## 9. 数据库管理 —— `db_manager.py`

**角色**: MySQL 数据管理。**652行，已全部实现。**

### 连接管理

- **DBUtils.PooledDB**: max=30, min=5, max=10
- **惰性初始化**: 首次查询时才创建连接池，MySQL 不可用时服务仍可启动
- **自动重连**: `ping(reconnect=True)`

### 两张表

| 表 | 用途 |
|------|------|
| `grading_records` | 批改结果 + 错题本合一（`in_notebook` 字段区分） |
| `benchmark_runs` | FERMAT 评测记录 |

### 核心方法

**写入**: `insert_record()`, `insert_batch()`
**查询**: `get_by_id()`, `get_by_upload()`, `get_by_task()`, `get_notebook_entries()`（分页+筛选）
**更新**: `update_grading_result()`, `update_topic()`, `update_subject()`, `rename_entry()`
**错题本**: `mark_as_notebook()`, `mark_reviewed()`, `mark_unreviewed()`, `mark_all_reviewed()`
**删除**: `delete_record()`, `batch_delete()`, `clear_all_notebook()`
**去重**: `is_duplicate(question_no, subject, question_text)`
**缓存**: `get_by_image_hash(image_hash, grading_mode)` — 图片哈希缓存命中
**统计**: `get_stats()` (按学科/知识点TOP20/难度/近7天/30天), `get_subjects()`
**评测**: `save_benchmark_run()`, `get_benchmark_history()`

---

## 10. 错题本 —— `error_notebook.py`

**角色**: 错题本业务封装。**173行，完全委托 DBManager，保留旧接口兼容。**

由于 MySQL 迁移，所有数据操作委托给 `DBManager`，模块自身只做字段名映射：
- `add_entry()` → `DBManager.mark_as_notebook_by_task()`
- `get_all()` → `DBManager.get_notebook_entries()`
- 其余方法直接委托对应 `DBManager` 方法

---

## 11. API 服务 —— `src/api/server.py`

**角色**: FastAPI Web 服务。**2330行，24个端点，已全部实现。**

### 完整端点清单

#### 页面
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 交互式前端页面 |
| GET | `/enhance-view` | 图像增强对比预览页 |

#### 系统
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 (version, db_connected, concurrency) |

#### 批改流水线
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload` | 上传图片 + 阿里云教育切题 API，返回 page_images + page_questions |
| POST | `/enhance-image` | 图像增强管线，返回原始 vs 增强 base64 |
| POST | `/grade-selected` | 核心批改接口，支持 3 种 grading_mode + 图片哈希缓存 |
| POST | `/correct` | 一键批改 (上传+OCR+批改+渲染)，返回 task_id |
| GET | `/result/{task_id}` | 查询批改结果 (先查 DB，再回退内存) |
| GET | `/image/{image_name}` | 提供批注图片文件 |

#### 重做
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/redo-grading` | 重做批改，支持 hint + sub_label |

#### 错题本 (12个端点)
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/error-notebook` | 分页查询 (subject/reviewed/keyword/difficulty 筛选) |
| GET | `/api/error-notebook/{id}` | 查询单条 |
| DELETE | `/api/error-notebook/{id}` | 删除单条 |
| POST | `/api/error-notebook/batch-delete` | 批量删除 |
| POST | `/api/error-notebook/save-batch` | 手动保存 (去重) |
| DELETE | `/api/error-notebook/clear-all/all` | 清空所有（不可逆） |
| PATCH | `/api/error-notebook/{id}/review` | 标记已复习 |
| PATCH | `/api/error-notebook/{id}/unreview` | 标记未复习 |
| PATCH | `/api/error-notebook/mark-all-reviewed` | 全部标记已复习 |
| PATCH | `/api/error-notebook/{id}/rename` | 重命名 |
| PATCH | `/api/error-notebook/{id}/topic` | 编辑知识点 |
| PATCH | `/api/error-notebook/{id}/subject` | 编辑学科 |

#### 统计
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/stats` | 统计数据 (总数/复习率/学科/知识点TOP20/难度/7天/30天) |
| GET | `/api/subjects` | 所有学科列表 |

---

## 12. 前端界面 —— `index.html`

**角色**: 完整的单文件 Web 应用。**2751行，Vanilla JS，零构建工具。**

### 四个 Tab 页面

| Tab | 功能 |
|-----|------|
| **批改页** (#page-correct) | 拖拽上传、快速批改/编辑选区、结果展示 |
| **编辑页** (#page-editor) | 全屏多页编辑器，拖动调整选区，8向缩放手柄 |
| **统计页** (#page-stats) | 概览卡片、SVG 环形图、知识点云 |
| **错题本** (#page-notebook) | 筛选、分页、批量操作、内联编辑 |

### 交互功能清单

- 拖拽文件上传 + 多文件排序
- 浏览器内摄像头拍照 (1920x1080)
- 三种批改模式切换 (vision / ocr_llm / vision_deepseek)
- 图像增强预览 (原始 vs 增强并排，逐页切换)
- 多页编辑选区 (左右箭头翻页、圆点跳页、8向缩放手柄)
- 5 阶段批改进度条动画
- SVG 环形成绩图 (≥60%绿/30-60%黄/<30%红)
- 可折叠结果卡片 (彩色左边框、小题嵌套、重做按钮、错题本复选框)
- 重做弹窗 (输入纠错提示 → 卡片内进度覆盖层 → 刷新结果)
- 缓存标识 (后端返回缓存结果时显示 badge)
- 取消批改 (AbortController)
- 错题本筛选 (学科/复习/难度下拉 + 关键词搜索)
- 批量选择删除 + 全部标记已复习
- 知识点内联编辑 (点击标签 → prompt() → PATCH)
- 知识点云可点击 (跳转错题本搜索)
- 分页导航

---

## 13. FERMAT 评测体系

### 评测脚本 `eval/fermat_benchmark.py`

```bash
# 跑 100 条 Vision 模式
python eval/fermat_benchmark.py --mode vision --count 100

# 跑全部 3 种模式
python eval/fermat_benchmark.py --mode all --count 100 --concurrency 5
```

**特性**: 并发评测 (ThreadPoolExecutor)、断点续跑 (checkpoint JSON)、结果写入 MySQL `benchmark_runs` 表。

### 可视化 `eval/fermat_view.py`

生成 HTML 报告: 准确率圆环、混淆矩阵 (精确率/召回率/F1)、三种模式对比柱状图、错误汇总表。

### 评测结果 (100条样本)

| 指标 | Vision | OCR+LLM |
|------|:------:|:-------:|
| 准确率 | 69.0% | 56.0% |
| 精确率 | 87.7% | 79.0% |
| 召回率 | 71.3% | 61.3% |
| F1 | 78.6 | 69.1 |
| 平均耗时 | 18.1s | 25.6s |

V2 评测 (含 Vision+DeepSeek) 已跑，数据在 `eval/fermat_v2_checkpoint.json`。

### 数据集

- 位置: `C:\Users\anpingyuan\Desktop\ceshiphoto\FERMAT_dataset\data\`
- 8 个 parquet 文件, 3.6GB, ~1800 张手写数学作业
- 字段: image.bytes, orig_q(LaTeX), orig_a, pert_a, pert_reasoning, has_error, grade(c06-c12), domain_code

---

## 14. 配置文件

`config/config.json` 完整结构:

```json
{
    "aliyun": {
        "access_key_id": "...",         // 阿里云 AccessKey（OCR用）
        "access_key_secret": "...",
        "region": "cn-hangzhou"
    },
    "llm": {
        "api_key": "...",               // DeepSeek API Key
        "model": "deepseek-v4-pro",
        "api_base_url": "https://api.deepseek.com",
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 60
    },
    "dashscope": {
        "api_key": "...",               // 通义千问 API Key
        "model": "qwen-vl-max",
        "timeout": 60
    },
    "mysql": {
        "host": "localhost",
        "port": 3308,
        "user": "root",
        "password": "...",
        "database": "homework"
    },
    "annotation": {
        "font_size": 24,
        "correct_color": [0, 255, 0],
        "wrong_color": [255, 0, 0],
        "comment_color": [0, 0, 255]
    },
    "image_processing": {
        "denoise_strength": 10,
        "binarize_method": "otsu",
        "resize_width": 1200,
        "preprocess_for_vl": true
    },
    "error_notebook": {
        "image_dir": "data/output",
        "storage_path": "data/error_notebook.json"
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "version": "2.2.0",
        "max_upload_size_mb": 50
    }
}
```

---

## 15. 模块完成度一览

| 模块 | 状态 | 代码量 | 说明 |
|------|:----:|:------:|------|
| `llm_grader.py` | ✅ **完成** | 1477行 | 3种批改模式 + 3种重做 + 三路投票 + 7策略JSON提取 |
| `image_processor.py` | ✅ **完成** | — | 20+方法, 双模式预处理管线, MSRCR/Sauvola/高反差保留 |
| `annotation_renderer.py` | ✅ **完成** | — | 全部标记+文字+分数渲染, 智能画布扩展 |
| `aliyun_ocr.py` | ✅ **完成** | — | 通用识别 + 教育切题 + 区域检测 |
| `db_manager.py` | ✅ **完成** | 652行 | 连接池 + 全部CRUD + 统计 + 缓存 + 评测 |
| `error_notebook.py` | ✅ **完成** | 173行 | 委托DBManager, 接口兼容 |
| `camera_capture.py` | ✅ **完成** | — | with上下文管理器 |
| `server.py` | ✅ **完成** | 2330行 | 24端点 + 图片哈希缓存 + 优雅降级 |
| `index.html` | ✅ **完成** | 2751行 | 4Tab + 拖拽上传 + 多页编辑 + 重做 + 统计 + 错题本 |
| `main.py` | ✅ **完成** | — | 4模式 + 多worker |
| `fermat_benchmark.py` | ✅ **完成** | 577行 | 3模式并发 + 断点续跑 |
| `fermat_view.py` | ✅ **完成** | 448行 | HTML报告 + HTTP服务 |
| `helpers.py` | ✅ **完成** | — | 日志/JSON/路径工具 |

**总代码量**: ~7383行 Python + ~2751行 JS/HTML/CSS

> 系统已从初版骨架全部实现完毕，当前版本 2.2.0，可直接部署使用。
