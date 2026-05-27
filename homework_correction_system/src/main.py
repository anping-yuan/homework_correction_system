"""
主程序入口
负责协调各模块完成作业批改的完整流程。
"""

import sys
import os
import argparse

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
    """文件输入模式"""
    logger = setup_logging()
    logger.info(f"处理文件: {input_path}")

    ocr = AliyunOCR(
        access_key_id=config["aliyun"]["access_key_id"],
        access_key_secret=config["aliyun"]["access_key_secret"],
    )
    grader = LLMGrader(api_key=config["llm"]["api_key"])
    renderer = AnnotationRenderer()

    logger.info("处理完成")


def run_api_mode(config: dict):
    """API服务模式"""
    import uvicorn
    from src.api.server import app
    uvicorn.run(app, host=config.get("api", {}).get("host", "0.0.0.0"),
                port=config.get("api", {}).get("port", 8000))


def main():
    args = parse_args()
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