# 作业批改系统

## 项目简介

基于计算机视觉和大语言模型的智能作业批改系统，支持摄像头实时采集作业图像、OCR文字识别、LLM智能批改和批注可视化。

## 项目结构

```
homework_correction_system/
├── src/                          # 源代码
│   ├── modules/                  # 功能模块
│   │   ├── camera_capture.py     # 摄像头采集模块
│   │   ├── image_processor.py    # 图像预处理模块
│   │   ├── aliyun_ocr.py         # 阿里云OCR模块
│   │   ├── llm_grader.py         # 大模型批改模块
│   │   └── annotation_renderer.py # 批注渲染模块
│   ├── api/                      # API接口
│   ├── utils/                    # 工具函数
│   └── main.py                   # 主程序
├── config/
│   └── config.json               # 系统配置
├── data/                         # 数据目录
│   ├── input/                    # 输入图像
│   ├── processed/                # 处理后图像
│   └── output/                   # 输出结果
├── tests/                        # 测试代码
├── docs/                         # 文档
├── requirements.txt              # 依赖包
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

编辑 `config/config.json`，填入阿里云OCR和大模型API的密钥。

### 运行

```bash
# API服务模式
python src/main.py --mode api

# 文件输入模式
python src/main.py --mode file --input data/input/homework.jpg

# 摄像头采集模式
python src/main.py --mode camera
```

## 运行测试

```bash
pytest tests/
```

## 核心流程

1. **图像采集** - 通过摄像头或文件输入获取作业图像
2. **图像预处理** - 去噪、纠偏、二值化等处理
3. **OCR识别** - 调用阿里云OCR识别文字内容
4. **智能批改** - 调用大模型进行答案批改和评分
5. **批注渲染** - 在原图上渲染批改标记和评语