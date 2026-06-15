"""
主程序入口
负责协调各模块完成作业批改的完整流程。
支持四种运行模式：camera / file / api / batch。
file/batch 模式使用试卷切题 API 实现题目级批改。
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.camera_capture import CameraCapture
from src.modules.image_processor import ImageProcessor
from src.modules.aliyun_ocr import AliyunOCR
from src.modules.llm_grader import LLMGrader
from src.modules.annotation_renderer import AnnotationRenderer
from src.utils.helpers import load_config, setup_logging, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="作业批改系统")
    parser.add_argument("--config", type=str, default="config/config.json", help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["camera", "file", "api", "batch"], default="file",
                        help="运行模式：camera / file / batch / api")
    parser.add_argument("--input", type=str, default=None, help="输入图像路径（file模式）或目录路径（batch模式）")
    parser.add_argument("--output", type=str, default="data/output", help="输出目录")
    parser.add_argument("--max-workers", type=int, default=4, help="最大并发数（batch模式）")
    return parser.parse_args()


def process_single_image(
    image_path: str,
    output_dir: str,
    config: dict,
) -> Dict:
    """
    切题版流水线:
    预处理 → 试卷切题 → 每题独立 OCR + 批改 + 渲染 → 总分汇总 → 输出
    """
    logger = setup_logging()
    # 图像预处理对象
    processor = ImageProcessor(config=config.get("image_processing", {}))
    # 提取文字对象
    ocr = AliyunOCR()
    # ai批改
    grader = LLMGrader()
    # 将批改结果渲染为可视化结果
    renderer = AnnotationRenderer()

    # 第一步根据路径找图片
    image = cv2.imread(image_path)
    if image is None:
        return {"file": image_path, "error": "无法读取图片"}

    processed = processor.process_with_config(image, config.get("image_processing", {}))

    logger.info("切题: 调用 RecognizeEduPaperCut ...")
    paper_cut_result = ocr.recognize_edu_paper_cut(image_path)
    questions = ocr.get_question_regions(paper_cut_result)
    logger.info(f"切出 {len(questions)} 道题")

    if not questions:
        logger.warning("未切出题目，回退到整页识别模式")
        paper_cut_result = None
        ocr_result = ocr.recognize_text(image_path)
        text = ocr.parse_result(ocr_result)
        regions = ocr.get_text_regions(ocr_result)
        grading = config.get("grading", {})
        grade_result = grader.grade(
            ocr_text=text,
            reference_answer=grading.get("reference_answer", ""),
        )
        questions_results = [{
            "question_no": 1,
            "question_text": text,
            "is_correct": grade_result.get("is_correct"),
            "score": grade_result.get("score"),
            "max_score": grade_result.get("max_score"),
            "comment": grade_result.get("comment"),
        }]
    else:
        questions_results = []
        total_grading = config.get("grading", {})

        for q in questions:
            q_no = q["question_no"]
            logger.info(f"批改第 {q_no} 题 ...")

            x, y, w, h = q["x"], q["y"], q["width"], q["height"]
            x1 = max(0, x - 10)
            y1 = max(0, y - 10)
            x2 = min(processed.shape[1], x + w + 10)
            y2 = min(processed.shape[0], y + h + 10)
            cropped = processed[y1:y2, x1:x2]
            crop_path = os.path.join(output_dir, f"question_{q_no}_crop.jpg")
            cv2.imwrite(crop_path, cropped)

            q_text = q.get("text", "")
            grade_result = grader.grade(
                ocr_text=q_text,
                reference_answer=total_grading.get("reference_answer", ""),
            )

            questions_results.append({
                "question_no": q_no,
                "question_text": q_text,
                "is_correct": grade_result.get("is_correct"),
                "score": grade_result.get("score"),
                "max_score": grade_result.get("max_score"),
                "comment": grade_result.get("comment"),
                "position": (x, y),
                "region": (x1, y1, x2 - x1, y2 - y1),
                "crop_path": crop_path,
            })

    corrections = []
    for qr in questions_results:
        pos = qr.get("position", (50, 50))
        is_correct = qr.get("is_correct", False)
        comment = qr.get("comment", "")
        score = qr.get("score", 0)
        max_score = qr.get("max_score", 0)
        corrections.append({
            "position": (pos[0], pos[1]),
            "is_correct": is_correct,
            "comment": f"第{qr['question_no']}题: {comment}",
            "score": score,
            "max_score": max_score,
        })
    # 批改结果渲染可视化
    annotated = renderer.render_all(processed, corrections)

    base_name = Path(image_path).stem
    ensure_dir(output_dir)

    annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    processed_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
    cv2.imwrite(processed_path, processed)

    total_score = sum(qr.get("score", 0) for qr in questions_results)
    total_max = sum(qr.get("max_score", 0) for qr in questions_results)
    correct_count = sum(1 for qr in questions_results if qr.get("is_correct"))

    all_correct = all(qr.get("is_correct") for qr in questions_results)
    if all_correct and len(questions_results) > 1:
        overall_correct = True
    elif not questions_results:
        overall_correct = False
    else:
        overall_correct = all_correct

    result_data = {
        "file": image_path,
        "questions_count": len(questions_results),
        "correct_count": correct_count,
        "total_score": total_score,
        "total_max": total_max,
        "is_correct": overall_correct,
        "questions": questions_results,
        "annotated_image": annotated_path,
        "processed_image": processed_path,
    }
    json_path = os.path.join(output_dir, f"{base_name}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


def run_file_mode(config: dict, input_path: str, output_dir: str):
    """文件输入模式：切题 → 每题独立批改 → 渲染 → 输出"""
    logger = setup_logging()
    logger.info(f"处理文件: {input_path}")

    t_start = time.time()
    # 调用上面自定义的函数来进行整个流程处理
    result = process_single_image(input_path, output_dir, config)
    t_elapsed = time.time() - t_start

    if "error" in result:
        logger.error(f"处理失败: {result['error']}")
        return

    logger.info(f"切出题目: {result['questions_count']} 道")
    logger.info(f"正确题数: {result['correct_count']}/{result['questions_count']}")
    logger.info(f"总分: {result['total_score']}/{result['total_max']}")
    for q in result["questions"]:
        status = "正确" if q["is_correct"] else "错误"
        logger.info(f"  第{q['question_no']}题: {status} ({q['score']}/{q['max_score']}) — {q['comment'][:40]}")
    logger.info(f"带批注图片: {result['annotated_image']}")
    logger.info(f"预处理图片: {result['processed_image']}")
    logger.info(f"总耗时: {t_elapsed:.2f} 秒")


def run_batch_mode(config: dict, input_dir: str, output_dir: str, max_workers: int):
    """高并发批量处理模式"""
    logger = setup_logging()
    logger.info(f"批量处理目录: {input_dir}，最大并发数: {max_workers}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_paths:
        logger.error("未找到支持的图片文件")
        return

    logger.info(f"共找到 {len(image_paths)} 张图片，开始并发处理...")
    t_start = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, p, output_dir, config): p
            for p in image_paths
        }
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"完成: {Path(image_path).name} → "
                            f"{result.get('correct_count', 0)}/{result.get('questions_count', 0)} 正确")
            except Exception as e:
                logger.error(f"处理失败: {image_path}, 错误: {e}")

    t_elapsed = time.time() - t_start

    summary = {
        "total_images": len(image_paths),
        "success": len([r for r in results if "error" not in r]),
        "total_questions": sum(r.get("questions_count", 0) for r in results),
        "total_correct": sum(r.get("correct_count", 0) for r in results),
        "time_seconds": round(t_elapsed, 2),
        "avg_time_per_image": round(t_elapsed / len(image_paths), 2) if image_paths else 0,
        "results": results,
    }
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"批量处理完成！图片: {summary['total_images']}, "
                f"题目: {summary['total_questions']}, "
                f"正确: {summary['total_correct']}, "
                f"总耗时: {t_elapsed:.2f}s, "
                f"平均: {summary['avg_time_per_image']}s/张")
    logger.info(f"汇总报告: {summary_path}")


def run_camera_mode(config: dict, output_dir: str):
    """摄像头采集模式"""
    logger = setup_logging()
    logger.info("启动摄像头采集模式...")
    processor = ImageProcessor(config=config.get("image_processing", {}))
    camera_cfg = config.get("camera", {})
    ensure_dir(output_dir)
    with CameraCapture(camera_id=camera_cfg.get("camera_id", 0)) as camera:
        for i in range(3):
            frame = camera.capture_frame()
            if frame is not None:
                processed = processor.process_with_config(
                    frame, config.get("image_processing", {})
                )
                save_path = os.path.join(output_dir, f"captured_{i+1}.jpg")
                camera.save_frame(processed, save_path)
                logger.info(f"图像已保存至: {save_path}")


def run_api_mode(config: dict):
    """API服务模式"""
    import uvicorn
    from src.api.server import app
    api_cfg = config.get("api", {})
    uvicorn.run(
        app,
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
    )


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
    elif args.mode == "batch":
        run_batch_mode(config, args.input, args.output, args.max_workers)
    elif args.mode == "api":
        run_api_mode(config)
    else:
        raise ValueError(f"不支持的模式: {args.mode}")


if __name__ == "__main__":
    main()
