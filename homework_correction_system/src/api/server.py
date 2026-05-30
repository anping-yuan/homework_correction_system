import os
import sys
import json
import uuid
import time
import logging
import threading
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.modules.image_processor import ImageProcessor
from src.modules.aliyun_ocr import AliyunOCR
from src.modules.llm_grader import LLMGrader
from src.modules.annotation_renderer import AnnotationRenderer
from src.utils.helpers import load_config, ensure_dir

logger = logging.getLogger("homework_correction.api")

_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_config: Dict = {}
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()

_tasks: Dict[str, Dict] = {}
_tasks_lock = threading.Lock()

_data_dir = os.path.join(_script_dir, "data")
ensure_dir(_data_dir)


def init_app_config(config: Dict):
    global _config, _executor
    _config = config
    max_workers = config.get("concurrency", {}).get("max_workers", 8)
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hw_correction")
    logger.info(f"API配置已初始化, max_workers={max_workers}")


def _get_config() -> Dict:
    if not _config:
        default_path = os.path.join(_script_dir, "config", "config.json")
        if os.path.exists(default_path):
            loaded = load_config(default_path)
            init_app_config(loaded)
    return _config


def _get_executor() -> ThreadPoolExecutor:
    with _executor_lock:
        if _executor is None:
            _get_config()
            max_workers = _config.get("concurrency", {}).get("max_workers", 8)
            return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="hw_correction")
        return _executor


app = FastAPI(title="Homework Correction System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_ocr() -> AliyunOCR:
    cfg = _get_config()
    ali_cfg = cfg.get("aliyun", {})
    return AliyunOCR(
        access_key_id=ali_cfg.get("access_key_id", ""),
        access_key_secret=ali_cfg.get("access_key_secret", ""),
        endpoint=ali_cfg.get("paper_cut_endpoint", "ocr-api.cn-hangzhou.aliyuncs.com"),
        connect_timeout=ali_cfg.get("connect_timeout", 10),
        read_timeout=ali_cfg.get("read_timeout", 30),
    )


def _get_grader() -> LLMGrader:
    cfg = _get_config()
    llm_cfg = cfg.get("llm", {})
    return LLMGrader(
        api_key=llm_cfg.get("api_key", ""),
        model=llm_cfg.get("model", "gpt-4"),
        api_base_url=llm_cfg.get("api_base_url", "https://api.openai.com/v1"),
        temperature=llm_cfg.get("temperature", 0.3),
        max_tokens=llm_cfg.get("max_tokens", 2048),
        request_timeout=llm_cfg.get("request_timeout", 60),
    )


def _execute_correction(task_id: str, image_path: str, subject: str):
    try:
        with _tasks_lock:
            _tasks[task_id]["status"] = "processing"
            _tasks[task_id]["progress"] = "图像预处理"

        img_cfg = _get_config().get("image_processing", {})
        concurrency_cfg = _get_config().get("concurrency", {})
        ocr_workers = concurrency_cfg.get("ocr_batch_size", 5)
        llm_workers = concurrency_cfg.get("llm_batch_size", 3)

        processor = ImageProcessor(config=img_cfg)
        ocr = _get_ocr()
        grader = _get_grader()
        renderer = AnnotationRenderer()

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        corrected = processor.auto_correct_and_crop(image)
        if corrected is not None:
            image = corrected
        image = processor.process_with_config(image)

        output_dir = os.path.join(_data_dir, "output", task_id)
        ensure_dir(output_dir)
        preprocessed_path = os.path.join(output_dir, "preprocessed.jpg")
        cv2.imwrite(preprocessed_path, image)

        with _tasks_lock:
            _tasks[task_id]["progress"] = "OCR识别"

        regions = ocr.process_paper_pipeline(
            preprocessed_path, subject=subject, max_workers=ocr_workers
        )

        with _tasks_lock:
            _tasks[task_id]["progress"] = "大模型批改"

        grading_result = grader.grade_pipeline(regions, max_workers=llm_workers)

        with _tasks_lock:
            _tasks[task_id]["progress"] = "批注渲染"

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

        result_data = {
            "task_id": task_id,
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
            "annotated_image_path": annotated_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with _tasks_lock:
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = "完成"
            _tasks[task_id]["result"] = result_data
            _tasks[task_id]["completed_at"] = time.time()

    except Exception as e:
        logger.exception(f"任务 {task_id} 处理失败")
        with _tasks_lock:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["progress"] = "失败"
            _tasks[task_id]["error"] = str(e)
            _tasks[task_id]["completed_at"] = time.time()


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "homework_correction_system"}


@app.post("/correct")
async def correct_homework(
    file: UploadFile = File(...),
    subject: Optional[str] = Form("general"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "仅支持上传图像文件"},
        )
    content = await file.read()
    if len(content) == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "上传文件为空"},
        )

    task_id = str(uuid.uuid4())
    upload_dir = os.path.join(_data_dir, "uploads")
    ensure_dir(upload_dir)
    ext = os.path.splitext(file.filename or "homework.jpg")[1] or ".jpg"
    image_path = os.path.join(upload_dir, f"{task_id}{ext}")
    with open(image_path, "wb") as f:
        f.write(content)

    with _tasks_lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": "排队中",
            "subject": subject,
            "created_at": time.time(),
            "filename": file.filename,
        }

    _get_executor().submit(_execute_correction, task_id, image_path, subject)

    return JSONResponse({
        "task_id": task_id,
        "status": "pending",
        "message": "任务已提交，请通过 /result/{task_id} 查询结果",
    })


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return JSONResponse(
            status_code=404,
            content={"error": "任务不存在"},
        )
    if task["status"] == "pending" or task["status"] == "processing":
        return JSONResponse({
            "task_id": task_id,
            "status": task["status"],
            "progress": task.get("progress", ""),
            "message": "任务处理中，请稍后查询",
        })
    if task["status"] == "failed":
        return JSONResponse({
            "task_id": task_id,
            "status": "failed",
            "error": task.get("error", "未知错误"),
        })
    return JSONResponse({
        "task_id": task_id,
        "status": "completed",
        "result": task.get("result", {}),
    })


@app.get("/tasks")
async def list_tasks(limit: int = 20):
    with _tasks_lock:
        all_tasks = list(_tasks.values())
    all_tasks.sort(key=lambda t: t.get("created_at", 0), reverse=True)
    return JSONResponse({
        "total": len(all_tasks),
        "tasks": [
            {
                "task_id": t["task_id"],
                "status": t["status"],
                "progress": t.get("progress", ""),
                "subject": t.get("subject", ""),
                "created_at": t.get("created_at", 0),
            }
            for t in all_tasks[:limit]
        ],
    })


@app.delete("/result/{task_id}")
async def delete_task(task_id: str):
    with _tasks_lock:
        if task_id in _tasks:
            del _tasks[task_id]
            return JSONResponse({"message": "任务已删除"})
    return JSONResponse(
        status_code=404,
        content={"error": "任务不存在"},
    )