"""
主程序入口
负责协调各模块完成作业批改的完整流程。
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.camera_capture import CameraCapture
from src.modules.image_processor import ImageProcessor
from src.modules.aliyun_ocr import AliyunOCR
from src.modules.llm_grader import LLMGrader
from src.modules.annotation_renderer import AnnotationRenderer
from src.utils.helpers import load_config, setup_logging, ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="作业批改系统")
    parser.add_argument("--config", type=str, default="config/config.json", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["camera", "file", "api"], default="api",
                        help="运行模式：camera-摄像头采集, file-文件输入, api-API服务")
    parser.add_argument("--input", type=str, default=None, help="输入图像路径（file模式）")
    parser.add_argument("--subject", type=str, default="general", help="学科类型")
    parser.add_argument("--output", type=str, default="data/output", help="输出目录")
    return parser.parse_args()


def run_camera_mode(config: dict, output_dir: str):
    logger = setup_logging()
    logger.info("启动摄像头采集模式...")
    cam_cfg = config.get("camera", {})
    img_cfg = config.get("image_processing", {})
    processor = ImageProcessor(config=img_cfg)
    camera_id = cam_cfg.get("camera_id", 0)
    resolution = cam_cfg.get("resolution", {})
    width = resolution.get("width")
    height = resolution.get("height")
    with CameraCapture(camera_id=camera_id, width=width, height=height) as camera:
        frame = camera.capture_frame()
        if frame is not None:
            processed = processor.process_with_config(frame)
            save_path = os.path.join(output_dir, "captured.jpg")
            camera.save_frame(processed, save_path)
            logger.info(f"图像已保存至: {save_path}")
        else:
            logger.error("未能捕获到图像帧")


def run_file_mode(config: dict, input_path: str, output_dir: str, subject: str = "general"):
    logger = setup_logging()
    logger.info(f"处理文件: {input_path}")
    ensure_dir(output_dir)

    img_cfg = config.get("image_processing", {})
    concurrency_cfg = config.get("concurrency", {})
    output_cfg = config.get("output", {})

    ocr_max_workers = concurrency_cfg.get("ocr_batch_size", 5)
    llm_max_workers = concurrency_cfg.get("llm_batch_size", 3)

    processor = ImageProcessor(config=img_cfg)

    ocr = AliyunOCR(
        access_key_id=config["aliyun"]["access_key_id"],
        access_key_secret=config["aliyun"]["access_key_secret"],
        endpoint=config["aliyun"].get("paper_cut_endpoint", "ocr-api.cn-hangzhou.aliyuncs.com"),
        connect_timeout=config["aliyun"].get("connect_timeout", 10),
        read_timeout=config["aliyun"].get("read_timeout", 30),
    )

    grader = LLMGrader(
        api_key=config["llm"]["api_key"],
        model=config["llm"].get("model", "gpt-4"),
        api_base_url=config["llm"].get("api_base_url", "https://api.openai.com/v1"),
        temperature=config["llm"].get("temperature", 0.3),
        max_tokens=config["llm"].get("max_tokens", 2048),
        request_timeout=config["llm"].get("request_timeout", 60),
    )

    renderer = AnnotationRenderer()

    logger.info("=== 阶段1: 图像预处理 ===")
    import cv2
    image = cv2.imread(input_path)
    if image is None:
        logger.error(f"无法读取图像: {input_path}")
        return
    logger.info(f"原始图像尺寸: {image.shape}")

    corrected = processor.auto_correct_and_crop(image)
    if corrected is not None:
        image = corrected
        logger.info("已执行透视校正")
    image = processor.process_with_config(image)
    preprocessed_path = os.path.join(output_dir, "preprocessed.jpg")
    cv2.imwrite(preprocessed_path, image)
    logger.info(f"预处理完成, 保存至: {preprocessed_path}")

    logger.info("=== 阶段2: 试卷切题与OCR识别 ===")
    regions = ocr.process_paper_pipeline(
        preprocessed_path, subject=subject, max_workers=ocr_max_workers
    )
    for region in regions:
        logger.info(
            f"  题目{region.get('index', '?')}: "
            f"题目={region.get('question_text', '')[:50]}... "
            f"答案={region.get('answer_text', '')[:30]}..."
        )

    logger.info("=== 阶段3: 大模型批改 ===")
    grading_result = grader.grade_pipeline(regions, max_workers=llm_max_workers)
    logger.info(f"总分: {grading_result['earned_score']}/{grading_result['total_score']}")
    logger.info(f"评语: {grading_result['feedback'][:100]}...")

    logger.info("=== 阶段4: 批注渲染 ===")
    corrections = []
    for i, gr in enumerate(grading_result["grading_results"]):
        region = regions[i] if i < len(regions) else {}
        position = region.get("position", [])
        if position and len(position) >= 2:
            pos = (int(position[0]), int(position[1]))
        else:
            pos = (50, 80 + i * 100)
        corrections.append({
            "position": pos,
            "is_correct": gr.get("is_correct", False),
            "comment": gr.get("comment", ""),
            "score": gr.get("score", 0),
            "max_score": gr.get("max_score", 10),
        })

    annotated_image = renderer.render_all(image, corrections)
    annotated_path = os.path.join(output_dir, "annotated.jpg")
    cv2.imwrite(annotated_path, annotated_image)
    logger.info(f"批注图像保存至: {annotated_path}")

    logger.info("=== 阶段5: 保存结果 ===")
    result_data = {
        "input_file": input_path,
        "subject": subject,
        "earned_score": grading_result["earned_score"],
        "total_score": grading_result["total_score"],
        "feedback": grading_result["feedback"],
        "question_count": len(regions),
        "grading_results": grading_result["grading_results"],
        "regions": [
            {
                "index": r.get("index"),
                "question_text": r.get("question_text", ""),
                "answer_text": r.get("answer_text", ""),
                "position": r.get("position", []),
            }
            for r in regions
        ],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result_path = os.path.join(output_dir, "result.json")
    save_json(result_data, result_path)
    logger.info(f"结果JSON保存至: {result_path}")

    logger.info("=== 批改完成 ===")
    logger.info(
        f"共 {len(regions)} 道题, "
        f"得分: {grading_result['earned_score']}/{grading_result['total_score']}"
    )
    return result_data


def run_api_mode(config: dict):
    import uvicorn
    from src.api.server import app, init_app_config
    init_app_config(config)
    uvicorn.run(app, host=config.get("api", {}).get("host", "0.0.0.0"),
                port=config.get("api", {}).get("port", 8000))


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.mode == "camera":
        run_camera_mode(config, args.output)
    elif args.mode == "file":
        run_file_mode(config, args.input, args.output, subject=args.subject)
    elif args.mode == "api":
        run_api_mode(config)
    else:
        raise ValueError(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main()