# 作业批改系统 — 项目详解手册

> 本文档对项目框架进行了全面、通俗的讲解，适合第一次接触该项目的开发者阅读。

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 目录结构](#2-目录结构)
- [3. 核心数据流](#3-核心数据流)
- [4. 主程序入口 —— `src/main.py`](#4-主程序入口--srcmainpy)
- [5. 五大核心模块](#5-五大核心模块)
  - [5.1 摄像头采集 —— `camera_capture.py`](#51-摄像头采集--camera_capturepy)
  - [5.2 图像预处理 —— `image_processor.py`](#52-图像预处理--image_processorpy)
  - [5.3 阿里云OCR —— `aliyun_ocr.py`](#53-阿里云ocr--aliyun_ocrpy)
  - [5.4 大模型批改 —— `llm_grader.py`](#54-大模型批改--llm_graderpy)
  - [5.5 批注渲染 —— `annotation_renderer.py`](#55-批注渲染--annotation_rendererpy)
- [6. API服务 —— `src/api/server.py`](#6-api服务--srcapiserverpy)
- [7. 工具函数 —— `src/utils/helpers.py`](#7-工具函数--srcutilshelperspy)
- [8. 配置文件 —— `config/config.json`](#8-配置文件--configconfigjson)
- [9. 模块完成度一览](#9-模块完成度一览)

---

## 1. 项目概述

### 这个系统做什么？

模拟老师批改作业的完整流程，将其自动化：

```
拍照/上传图片 → 图片清晰化 → 识别文字 → AI 批改打分 → 在原图上画批注
```

每一站都由一个独立模块负责，最终输出一张带批注的图片和一份 JSON 批改结果。

### 适用场景

- 老师批量批改纸质作业
- 在线教育平台自动批改
- 考试答题卡自动阅卷

---

## 2. 目录结构

```
homework_correction_system/
├── src/                          # 源代码
│   ├── modules/                  # 功能模块（5个核心模块）
│   │   ├── camera_capture.py     # 摄像头采集
│   │   ├── image_processor.py    # 图像预处理
│   │   ├── aliyun_ocr.py         # 阿里云OCR识别
│   │   ├── llm_grader.py         # 大模型批改
│   │   └── annotation_renderer.py # 批注渲染
│   ├── api/                      # HTTP API接口
│   │   └── server.py             # FastAPI 服务
│   ├── utils/                    # 工具函数
│   │   └── helpers.py            # 日志、JSON读写等
│   └── main.py                   # 主程序入口（总指挥）
├── config/
│   └── config.json               # 系统配置（密钥、参数等）
├── data/                         # 数据目录
│   ├── input/                    # 输入图像存放处
│   ├── processed/                # 预处理后的图像
│   └── output/                   # 输出结果（带批注的图 + JSON）
├── tests/                        # 单元测试
├── docs/                         # 文档
├── requirements.txt              # Python 依赖包
└── README.md                     # 项目说明
```

---

## 3. 核心数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│                         作业批改完整数据流                              │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │ ① 图像输入    │  摄像头拍照 或 本地文件上传                          │
│  │   CameraCapture │                                                  │
│  └──────┬───────┘                                                    │
│         │ 原始图像 (numpy数组)                                        │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │ ② 图像预处理  │  去噪 → 纠偏 → (可选)灰度化/二值化/缩放             │
│  │   ImageProcessor │                                                 │
│  └──────┬───────┘                                                    │
│         │ 干净端正的图像                                              │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │ ③ 文字识别   │  阿里云OCR：印刷体/手写体/公式识别                   │
│  │   AliyunOCR    │  → 输出：题目文本 + 学生答案文本                    │
│  └──────┬───────┘                                                    │
│         │ 结构化文字数据                                             │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │ ④ AI 批改    │  大模型判断对错、打分、生成评语                      │
│  │   LLMGrader    │  → 输出：{对/错, 得分, 评语}                      │
│  └──────┬───────┘                                                    │
│         │ 批改结果列表                                               │
│         ▼                                                            │
│  ┌──────────────┐                                                    │
│  │ ⑤ 批注渲染   │  在图上画对勾/叉号/框/分数/评语                     │
│  │   AnnotationRenderer │                                             │
│  └──────┬───────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────────────────────────┐                                │
│  │ ⑥ 最终输出                       │                                │
│  │  📄 带批注的图片  +  📋 JSON结果  │                                │
│  └──────────────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. 主程序入口 —— `src/main.py`

### 作用

`main.py` 是整个系统的**总指挥**。它不亲自做批改，而是负责：
1. 读取命令行参数（你想怎么运行？）
2. 加载配置文件（密钥、参数从哪读？）
3. 根据运行模式调用对应的模块

### 三种运行模式

| 模式 | 启动命令 | 适用场景 |
|------|----------|----------|
| **camera** | `python src/main.py --mode camera` | 用电脑摄像头拍作业 |
| **file** | `python src/main.py --mode file --input 图片.jpg` | 处理已有图片 |
| **api** | `python src/main.py --mode api` | 启动 Web 服务供远程调用 |

### 关键代码解读

```python
# 第20-28行：定义命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="作业批改系统")
    parser.add_argument("--config", default="config/config.json")  # 配置文件路径
    parser.add_argument("--mode", choices=["camera", "file", "api"], default="api")
    parser.add_argument("--input", default=None)   # file模式下的输入图片路径
    parser.add_argument("--output", default="data/output")  # 输出目录
    return parser.parse_args()
```

```python
# 第68-79行：核心调度逻辑
def main():
    args = parse_args()              # ① 解析命令
    config = load_config(args.config) # ② 读配置
    if args.mode == "camera":        # ③ 按模式分发
        run_camera_mode(config, args.output)
    elif args.mode == "file":
        run_file_mode(config, args.input, args.output)
    elif args.mode == "api":
        run_api_mode(config)
```

```python
# 第31-42行：摄像头模式的完整流程
def run_camera_mode(config, output_dir):
    logger = setup_logging()                              # 初始化日志
    processor = ImageProcessor()                          # 创建预处理器
    with CameraCapture(camera_id=0) as camera:            # 打开摄像头
        frame = camera.capture_frame()                    # 拍一张
        if frame is not None:
            processed = processor.process(frame)          # 去噪+纠偏
            save_path = os.path.join(output_dir, "captured.jpg")
            camera.save_frame(processed, save_path)       # 保存
```

---

## 5. 五大核心模块

### 5.1 摄像头采集 —— `camera_capture.py`

**角色**：照相师，负责拍照。

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `open()` | 打开摄像头 | `True`(成功) / `False`(失败) |
| `capture_frame()` | 拍一张照片 | numpy数组（图像数据）或 `None` |
| `save_frame(frame, path)` | 把照片存到硬盘 | `True` / `False` |
| `close()` | 关闭摄像头 | 无 |

**特色设计**：支持 `with` 语句，自动开关摄像头。

```python
with CameraCapture() as camera:   # 自动 open()
    frame = camera.capture_frame()
# 离开 with 块 → 自动 close()，不会忘记关摄像头
```

---

### 5.2 图像预处理 —— `image_processor.py`

**角色**：修图师，把照片修清晰端正。

| 方法 | 功能 | 通俗理解 |
|------|------|----------|
| `to_grayscale(image)` | 转灰度图 | 彩色→黑白，去掉颜色干扰 |
| `binarize(image, method)` | 二值化 | 变成纯黑纯白，文字更突出 |
| `denoise(image, strength)` | 去噪 | 去掉照片颗粒感 |
| `deskew(image)` | 纠偏 | 拍歪了自动旋转校正 |
| `resize(image, width, height)` | 调尺寸 | 缩放到指定大小 |
| `process(image)` | **一键处理** | 自动执行 `去噪 → 纠偏` |

**二值化的两种策略**：

| method | 说明 | 适用场景 |
|--------|------|----------|
| `"otsu"` | 自动算最优阈值 | 光照均匀的图片 |
| `"adaptive"` | 分区域自适应阈值 | 有阴影、光照不均的图片 |

---

### 5.3 阿里云OCR —— `aliyun_ocr.py`

**角色**：认字的人，把图片上的文字提取为文本。

> ⚠️ **状态：骨架已就位，待填充实现**。方法签名已定义好，内部逻辑需填入阿里云SDK调用代码。

| 方法 | 功能 |
|------|------|
| `recognize_text(image_path)` | 识别印刷体文字 |
| `recognize_handwriting(image_path)` | 识别手写文字 |
| `recognize_formula(image_path)` | 识别数学公式 |
| `get_text_regions(ocr_result)` | 从结果中提取文字位置信息 |
| `parse_result(ocr_result)` | 将原始结果转成纯文本字符串 |

---

### 5.4 大模型批改 —— `llm_grader.py`

**角色**：AI老师，判断对错、打分、写评语。

> ⚠️ **状态：部分完成**。`batch_grade()` 和 `calculate_total_score()` 已实现，`grade()` 核心方法待填充。

| 方法 | 功能 | 完成度 |
|------|------|--------|
| `load_prompt(template)` | 加载自定义批改提示词 | ✅ |
| `build_evaluation_context(...)` | 把题目+答案+参考答案拼成一段请求文本 | ✅ |
| `grade(question, answer, reference)` | 对一道题批改 | ❌ 待实现 |
| `batch_grade([题目列表])` | 批量批改（循环调用 grade） | ✅ |
| `calculate_total_score([批改结果])` | 算总分 → `(得分, 满分)` | ✅ |
| `generate_feedback(结果)` | 根据批改结果生成评语 | ❌ 待实现 |

**`build_evaluation_context` 的输出示例**：

```
题目：计算 3 + 5 = ?
学生答案：3 + 5 = 9
参考答案：3 + 5 = 8
```

这段文本会被发送给大模型，让它判断对错。

**`batch_grade` 的输入输出**：

```python
输入:
[
    {"question": "1+1=?", "student_answer": "2", "reference_answer": "2", "max_score": 5},
    {"question": "3×4=?", "student_answer": "10", "reference_answer": "12", "max_score": 5},
]

输出:
[
    {"question": "1+1=?", "is_correct": True,  "score": 5, "comment": "正确"},
    {"question": "3×4=?", "is_correct": False, "score": 0, "comment": "3×4=12"},
]
```

---

### 5.5 批注渲染 —— `annotation_renderer.py`

**角色**：批改标记员，像老师用红笔在卷子上批改一样，把结果画到图片上。

**颜色约定**：

| 颜色 | RGB值 | 含义 |
|------|-------|------|
| 绿色 | `(0, 255, 0)` | 正确标记 |
| 红色 | `(255, 0, 0)` | 错误标记 |
| 蓝色 | `(0, 0, 255)` | 文字评语 |

| 方法 | 功能 | 完成度 |
|------|------|--------|
| `draw_correct_mark(image, position)` | 画绿色对勾 | ❌ 待实现 |
| `draw_wrong_mark(image, position)` | 画红色叉号 | ❌ 待实现 |
| `draw_text_annotation(image, text, position)` | 写文字批注 | ❌ 待实现 |
| `draw_bounding_box(image, box, color)` | 画矩形框 | ✅ 已实现 |
| `render_score(image, score, total, position)` | 显示分数 | ❌ 待实现 |
| `render_all(image, corrections)` | **一键渲染所有批注** | ✅ 已实现 |

**`render_all` 的实现逻辑**（已写好，无需改动）：

```python
def render_all(self, image, corrections):
    result = image.copy()
    for correction in corrections:             # 遍历每道题
        position = correction["position"]       # 题目位置
        is_correct = correction["is_correct"]   # 对/错
        comment = correction["comment"]         # 评语
        if is_correct:
            result = self.draw_correct_mark(result, position)   # 画 ✓
        else:
            result = self.draw_wrong_mark(result, position)     # 画 ✗
        if comment:
            result = self.draw_text_annotation(result, comment, position)
    return result
```

---

## 6. API服务 —— `src/api/server.py`

**作用**：基于 FastAPI 框架，把批改功能包装成 HTTP 接口，方便其他系统调用。

| 接口 | 方法 | 作用 |
|------|------|------|
| `/health` | GET | 健康检查（服务是否正常运行） |
| `/correct` | POST | 上传作业图片，提交批改任务 |
| `/result/{task_id}` | GET | 查询某次批改的结果 |

> ⚠️ `/correct` 和 `/result` 的内部逻辑尚未实现。

---

## 7. 工具函数 —— `src/utils/helpers.py`

**作用**：提供可复用的通用函数，避免各模块重复写相同的代码。

| 函数 | 功能 |
|------|------|
| `setup_logging(level, file)` | 配置日志（同时输出到控制台和文件） |
| `load_config(path)` | 加载 JSON 配置文件 |
| `ensure_dir(path)` | 确保目录存在，不存在就创建 |
| `save_json(data, path)` | 保存数据为 JSON 文件 |
| `load_json(path)` | 从 JSON 文件读取数据 |
| `get_supported_formats()` | 返回支持的图片格式列表 |

---

## 8. 配置文件 —— `config/config.json`

**作用**：集中管理所有外部配置，避免把密钥和参数写死在代码里。

```json
{
    "camera": {
        "camera_id": 0,           // 摄像头编号（0=笔记本自带）
        "resolution": { "width": 1920, "height": 1080 }
    },
    "image_processing": {
        "denoise_strength": 10,   // 去噪强度
        "binarize_method": "otsu",// 二值化策略
        "resize_width": 1200      // 缩放目标宽度
    },
    "aliyun": {
        "access_key_id": "你的阿里云密钥ID",      // ← 需要替换
        "access_key_secret": "你的阿里云密钥Secret",// ← 需要替换
        "region": "cn-hangzhou"
    },
    "llm": {
        "api_key": "你的大模型API密钥",            // ← 需要替换
        "model": "gpt-4",
        "api_base_url": "https://api.openai.com/v1",
        "temperature": 0.3,        // 随机性（0=确定, 1=创意）
        "max_tokens": 2048         // 最大返回长度
    },
    "api": {
        "host": "0.0.0.0",        // 监听所有网卡
        "port": 8000              // 端口号
    },
    "output": {
        "save_annotated_image": true, // 是否保存带批注的图片
        "save_json_result": true      // 是否保存JSON结果
    }
}
```

**使用时需要替换的配置项**：

| 配置项 | 说明 |
|--------|------|
| `aliyun.access_key_id` | 阿里云 AccessKey ID |
| `aliyun.access_key_secret` | 阿里云 AccessKey Secret |
| `llm.api_key` | 大模型 API 密钥（OpenAI / 通义千问等） |
| `llm.model` | 模型名称（如 `gpt-4`、`qwen-plus` 等） |

---

## 9. 模块完成度一览

| 模块 | 状态 | 需要做的工作 |
|------|------|-------------|
| `camera_capture.py` | ✅ **完全可用** | 无需改动 |
| `image_processor.py` | ✅ **完全可用** | 无需改动 |
| `helpers.py` | ✅ **完全可用** | 无需改动 |
| `main.py` | ✅ **完全可用** | 框架完整，可按需扩展 |
| `config.json` | 🔶 **需填密钥** | 替换 `your-xxx` 为真实密钥 |
| `llm_grader.py` | 🔶 **核心待实现** | 实现 `grade()` 方法（调用大模型API） |
| `aliyun_ocr.py` | 🔶 **核心待实现** | 实现 `recognize_text()` 等方法（调用阿里云SDK） |
| `annotation_renderer.py` | 🔶 **底层待实现** | 实现 `draw_correct_mark()`、`draw_wrong_mark()`、`draw_text_annotation()` 的绘图逻辑 |
| `api/server.py` | 🔶 **接口待实现** | 实现 `/correct` 和 `/result` 的业务逻辑 |

---

> **建议开发顺序**：先填密钥配通 `aliyun_ocr.py` → 再实现 `llm_grader.py` → 最后完善 `annotation_renderer.py` 的绘图 → 组装到 `api/server.py`。