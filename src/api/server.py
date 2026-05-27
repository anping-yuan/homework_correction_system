"""
API接口模块
提供HTTP RESTful API，用于接收作业图像、返回批改结果。
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI(title="Homework Correction System API", version="1.0.0")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "homework_correction_system"}


@app.post("/correct")
async def correct_homework(file: UploadFile = File(...), subject: Optional[str] = None):
    """上传作业图像并获取批改结果"""
    pass


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """根据任务ID获取批改结果"""
    pass