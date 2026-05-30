"""
主程序入口
负责协调各模块完成作业批改的完整流程。
"""
import json
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.camera_capture import CameraCapture
from src.modules.image_processor import ImageProcessor
from src.modules.aliyun_ocr import AliyunOCR
from src.modules.llm_grader import LLMGrader
from src.modules.annotation_renderer import AnnotationRenderer
from src.utils.helpers import load_config, setup_logging


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="作业批改系统")
    parser.add_argument("--config", type=str, default="config/config.json", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["camera", "file", "api"], default="api",
                        help="运行模式：camera-摄像头采集, file-文件输入, api-API服务")
    parser.add_argument("--input", type=str, default=None, help="输入图像路径（file模式）")
    parser.add_argument("--output", type=str, default="data/output", help="输出目录")
    return parser.parse_args()


def run_camera_mode(config: dict, output_dir: str):
    """摄像头采集模式"""
    logger = setup_logging()
    logger.info("启动摄像头采集模式...")
    processor = ImageProcessor()
    with CameraCapture(camera_id=config.get("camera_id", 0)) as camera:
        frame = camera.capture_frame()
        if frame is not None:
            processed = processor.process(frame)
            save_path = os.path.join(output_dir, "captured.jpg")
            camera.save_frame(processed, save_path)
            logger.info(f"图像已保存至: {save_path}")


def run_file_mode(config: dict, input_path: str, output_dir: str):
    """文件输入模式: ocr识别 -> 大模型批改 -> 输出结果"""
    logger = setup_logging()
    logger.info(f"处理文件: {input_path}")

    # 这里就用默认的模型, 后续可以改成自定义
    # ocr提取图片中的信息
    ocr = AliyunOCR()
    # 大模型也是用默认的
    # 批改
    grader = LLMGrader()
    # 渲染
    renderer = AnnotationRenderer()

    logger.info("步骤1:ocr识别")
    # 原始字典文件
    ocr_result = ocr.recognize_text(input_path)
    # ocr结果的纯文本
    text = ocr.parse_result(ocr_result)

    logger.info("步骤2:大模型批改")
    question = config.get("question", "请批改以下作业")
    reference_answer = config.get("reference_answer", "")

    grade_result = grader.grade(question = question, student_answer = text, reference_answer =reference_answer)

    logger.info("步骤3:保存结果")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "result.json")
    with open(json_path, "w", encoding = "utf-8") as f:
        json.dump(grade_result, f, ensure_ascii = False, indent = 2)

    logger.info(f"结果已保存至: {json_path}")
    logger.info(f"是否正确:{grade_result.get("is_correct")}")
    logger.info(f"得分:{grade_result.get('score')}/{grade_result.get('max_score')}")
    logger.info(f"评语:{grade_result.get('comment')}")

def run_api_mode(config: dict):
    """API服务模式"""
    import uvicorn
    from src.api.server import app
    uvicorn.run(app, host=config.get("api", {}).get("host", "0.0.0.0"),
                port=config.get("api", {}).get("port", 8000))


def main():
    args = parse_args()
    if not os.path.isabs(args.config):
        project_root = Path(__file__).resolve().parent.parent
        args.config = str(project_root / args.config)
    config = load_config(args.config)

    if args.mode == "camera":
        run_camera_mode(config, args.output)
    elif args.mode == "file":
        run_file_mode(config, args.input, args.output)
    elif args.mode == "api":
        run_api_mode(config)
    else:
        raise ValueError(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main()