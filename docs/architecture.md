# 架构设计文档

## 系统概述

作业批改系统采用 **FastAPI Web 服务 + 模块化批改引擎** 架构。前端为单文件 SPA（Vanilla JS），后端通过 REST API 暴露所有功能，MySQL 存储批改记录和错题本。

## 整体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                         浏览器 (index.html, 2751行)                    │
│  4个Tab: 批改页 | 编辑页 | 统计页 | 错题本                               │
│  拖拽上传 | 模式选择 | 选区编辑 | 进度动画 | 环形图 | 重做               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP REST API (24 端点)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FastAPI 服务层 (server.py, 2330行)                  │
│                                                                      │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────────┐   │
│  │ 上传/切题  │ │ 批改调度   │ │ 错题本CRUD │ │ 统计/健康检查      │   │
│  │ /upload   │ │ /grade-*  │ │ /api/err* │ │ /stats /health    │   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └───────────────────┘   │
│        │              │             │                                 │
└────────┼──────────────┼─────────────┼─────────────────────────────────┘
         │              │             │
         ▼              ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        核心模块层                                  │
│                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ ImageProcessor │  │   LLMGrader    │  │ AnnotationRenderer│  │
│  │ (图像预处理)    │  │   (批改引擎)    │  │   (批注渲染)      │  │
│  │                │  │                │  │                  │  │
│  │ · 双模式管线   │  │ · 3种批改模式  │  │ · 对/错标记     │  │
│  │ · MSRCR/Sauvola│  │ · 三路投票     │  │ · 中文批注       │  │
│  │ · 高反差保留   │  │ · 7策略JSON提取│  │ · 智能画布扩展   │  │
│  │ · 线性拉伸     │  │ · 重做+纠错    │  │ · 分数渲染       │  │
│  └───────┬────────┘  └───────┬────────┘  └──────────────────┘  │
│          │                   │                                   │
│  ┌───────┴────────┐  ┌───────┴────────┐                          │
│  │   AliyunOCR    │  │  ErrorNotebook │                          │
│  │   (文字识别)    │  │  (错题本)      │                          │
│  │                │  │                │                          │
│  │ · 通用文字识别 │  │ · 委托DBManager│                          │
│  │ · 教育试卷切题 │  │ · 兼容旧接口   │                          │
│  └────────────────┘  └────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据层                                     │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │   DBManager          │  │   upload_store       │             │
│  │   (DBUtils PooledDB) │  │   (内存会话缓存)      │             │
│  │                      │  │                      │             │
│  │ · grading_records    │  │ · 上传→批改间临时数据 │             │
│  │ · benchmark_runs     │  │ · 过期自动清理(>2h)  │             │
│  └──────────┬───────────┘  └──────────────────────┘             │
│             │                                                     │
│             ▼                                                     │
│  ┌──────────────────────┐                                        │
│  │   MySQL 8.0          │                                        │
│  │   localhost:3308     │                                        │
│  │   database: homework │                                        │
│  └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 三种批改模式的架构差异

```
Vision 模式（单阶段）:
  图片 → [预处理管线] → Qwen-VL 看图 → JSON 结果 → 一致性修正

OCR+LLM 模式（两阶段）:
  图片 → 逐题裁剪 → 阿里云OCR → 文本拼接 → DeepSeek 批改 → JSON 结果

Vision+DeepSeek 模式（双阶段+三路投票）:
  图片 → [预处理管线] → Qwen-VL 提取文字
                       ↓
                 DeepSeek 三路并发投票
                 ┌─ 数学老师视角 ──┐
                 ├─ 严格考官视角 ──┼─→ 多数投票 → 一致性修正
                 └─ 数学家教视角 ──┘
```

## 并发架构

```
uvicorn (4 workers)
  └── 每个 worker:
        ├── LLMGrader (单例)
        │     └── _api_semaphore = Semaphore(5)  ← 限制 API 并发
        └── DBManager
              └── PooledDB (max=30, min=5)
```

- **4 worker 进程 × 5 信号量 = 最多 20 并发 API 调用**
- **30 个 DB 连接池**，连接复用 + 自动重连检测

## 数据库表结构

### grading_records（批改记录 + 错题本合一）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT AUTO_INCREMENT | 主键 |
| upload_id | VARCHAR(64) | 上传会话 ID |
| task_id | VARCHAR(64) | 批改任务 ID |
| question_no | VARCHAR(32) | 题号 |
| page | INT | 页码（多页作业） |
| question_text | TEXT | 题目文本 |
| student_answer | TEXT | 学生答案 |
| is_correct | TINYINT(1) | 是否正确 |
| score | FLOAT | 得分 |
| max_score | FLOAT | 满分 |
| comment | TEXT | 评语 |
| subject | VARCHAR(32) | 学科 |
| topic | VARCHAR(128) | 知识点 |
| difficulty | VARCHAR(16) | 难度 (easy/medium/hard) |
| sub_questions | JSON | 子题嵌套结果 |
| original_image | VARCHAR(512) | 原图路径 |
| annotated_image | VARCHAR(512) | 批注图路径 |
| crop_image | VARCHAR(512) | 裁切图路径 |
| region_x/y/width/height | INT | 选区坐标 |
| grading_mode | VARCHAR(32) | 批改模式 |
| image_hash | VARCHAR(64) | 图片哈希（缓存） |
| in_notebook | TINYINT(1) | 是否在错题本中 |
| is_reviewed | TINYINT(1) | 是否已复习 |
| reviewed_at | DATETIME | 复习时间 |

### benchmark_runs（评测记录）

| 字段 | 说明 |
|------|------|
| id, sample_count, mode, accuracy, total_valid | 基本评测指标 |
| false_positives, false_negatives | 误判/漏判 |
| recall_rate, precision_rate, avg_time_sec | 精确率/召回率/耗时 |
| concurrency, note, run_time | 并发数/备注/运行时间 |

## 图像预处理管线

```
原始图片
  → normalize_illumination()    # 光照归一化（去阴影）
  → bilateral_denoise()         # 双边滤波（保边去噪）
  → CLAHE                       # 对比度增强
  → unsharp_mask()              # USM 锐化
  → 尺寸检查 + 质量压缩          # >3.5MB 或 >2048px 时压缩
  → base64 data URL             # 发送给 VL 模型
```

可选替代方法（按需调用）:
- `msrcr()` — MSRCR 多尺度 Retinex（更精细的光照归一化）
- `sauvola_binarize()` — Sauvola 自适应二值化（手写文字专用）
- `high_pass_enhance()` — 高反差保留（比 USM 更自然的锐化）
- `morphological_clean()` — 形态学开运算（修复断笔）

## 关键设计决策

| 决策 | 原因 |
|------|------|
| 整页发 Vision 模型 | 保留上下文，不裁剪分别批改 |
| 保留 OCR+LLM 模式 | 适合印刷清晰的作业，老师要求 |
| Vision+DeepSeek 双阶段 | 视觉模型读图 + 文本模型推理，各取所长 |
| 三路投票 | 多数投票降低单一模型偏差 |
| 图片哈希缓存 | 相同图片+相同模式跳过重复批改 |
| 错题本手动保存+去重 | 避免自动添加导致重复 |
| 图片存路径不存 BLOB | DB 轻量，文件系统更适合图片 |
| DB 惰性初始化 | MySQL 不可用时服务仍可启动 |
| 连接池 + 信号量 | 控制并发，避免 API 限流 |
| 单文件前端 | 减少部署复杂度，无需构建工具 |
