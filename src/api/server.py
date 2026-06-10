"""
API 接口模块
提供 HTTP RESTful 接口，用于上传作业图片并获取批改结果。
支持交互式批改：上传图片 → 自动切题预览 → 用户调整选区 → 按选区批改 → 展示结果

启动方式：python src/main.py --mode api
访问文档：http://localhost:8000/docs
访问交互式页面：http://localhost:8000/
"""

import json
import os
import uuid
import tempfile
import logging
import shutil
from typing import Optional, List
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, Path as FastAPIPath
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger("api")

# 错题本实例（全局单例）
from src.modules.error_notebook import ErrorNotebook
error_notebook = ErrorNotebook()

# 加载配置，获取错题本图片目录
def _load_app_config():
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.json"
    if config_path.exists():
        import json as _json
        with open(config_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    return {}
_app_config = _load_app_config()
error_nb_image_dir = _app_config.get("error_notebook", {}).get("image_dir", "data/error_notebook_images")
ERROR_NB_IMAGE_DIR = Path(error_nb_image_dir)
ERROR_NB_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="作业批改系统 API",
    description="""
## 功能介绍

上传中小学作业图片，系统自动完成：

1. **OCR 文字识别** — 提取图片中的题目和答案
2. **大模型批改** — 调用 AI 判断对错并给出解题思路
3. **批注渲染** — 在原图上标记对勾/叉号/分数

## 使用步骤

### 方式一：交互式批改（推荐）
1. 打开 http://localhost:8000/ 进入网页
2. 上传图片，系统自动切题并展示题目区域
3. 可删除/添加/调整题目区域
4. 点击"开始批改"获取结果

### 方式二：API 调用
1. 调用 `POST /correct` 上传作业图片（一键批改）
2. 或调用 `POST /upload` 上传图片获取切题区域
3. 调用 `POST /grade-selected` 按选定区域批改
4. 调用 `GET /result/{task_id}` 查询批改结果
5. 调用 `GET /image/{image_name}` 查看带批注的图片
""",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件目录（用于前端页面）
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 挂载错题本图片目录（用户可配置）
app.mount("/error_images", StaticFiles(directory=str(ERROR_NB_IMAGE_DIR)), name="error_images")

task_store: dict = {}
# 临时存储上传的图片和切题结果（生产环境应使用Redis等）
upload_store: dict = {}


# ========== Pydantic 请求/响应模型 ==========

class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="服务状态")
    service: str = Field(default="作业批改系统", description="服务名称")
    version: str = Field(default="2.0.0", description="系统版本号")


class GradingResult(BaseModel):
    is_correct: bool = Field(description="答案是否正确")
    score: int = Field(description="所得分数")
    max_score: int = Field(description="题目满分")
    comment: str = Field(description="评语")
    ocr_text: str = Field(description="OCR 识别出的文字内容")
    annotated_image_url: str = Field(description="带批注图片的访问链接")


class CorrectResponse(BaseModel):
    task_id: str = Field(description="批改任务唯一标识")
    message: str = Field(description="处理结果说明")
    result: GradingResult = Field(description="批改结果详情")


class ResultQueryResponse(BaseModel):
    task_id: str = Field(description="批改任务 ID")
    result: GradingResult = Field(description="批改结果详情")


class ErrorResponse(BaseModel):
    error: str = Field(description="错误描述信息")


class FailedResponse(BaseModel):
    task_id: str = Field(description="任务 ID")
    message: str = Field(description="失败说明")
    error: str = Field(description="具体错误原因")


class QuestionRegion(BaseModel):
    question_no: int = Field(description="题号")
    x: int = Field(description="区域左上角 x 坐标")
    y: int = Field(description="区域左上角 y 坐标")
    width: int = Field(description="区域宽度")
    height: int = Field(description="区域高度")
    text: str = Field(default="", description="区域识别出的文字")


class UploadResponse(BaseModel):
    upload_id: str = Field(description="上传任务唯一标识")
    image_url: str = Field(description="原图访问地址（多图拼接后）")
    questions: List[QuestionRegion] = Field(description="自动切出的题目区域列表")
    message: str = Field(description="处理说明")
    stitched_from: int = Field(default=1, description="由几张图片拼接而成，1 表示未拼接")


class GradeSelectedRequest(BaseModel):
    upload_id: str = Field(description="上传时返回的 upload_id")
    questions: List[QuestionRegion] = Field(description="用户确认后的题目区域列表")
    subject: Optional[str] = Field(default=None, description="学科类型")


class SubQuestionResult(BaseModel):
    label: str = Field(description="小题编号")
    is_correct: bool = Field(description="是否正确")
    score: int = Field(description="得分")
    max_score: int = Field(description="满分")
    comment: str = Field(description="评语")


class QuestionResult(BaseModel):
    question_no: int = Field(description="题号")
    is_correct: bool = Field(description="是否正确")
    score: int = Field(description="得分")
    max_score: int = Field(description="满分")
    comment: str = Field(description="评语")
    ocr_text: str = Field(description="识别文字")
    student_answer_found: bool = Field(default=True, description="是否找到学生作答")
    sub_questions: List[SubQuestionResult] = Field(default=[], description="小题拆分结果")
    image_url: str = Field(default="", description="原图地址")


class GradeSelectedResponse(BaseModel):
    task_id: str = Field(description="批改任务 ID")
    message: str = Field(description="处理结果")
    total_score: int = Field(description="总分")
    total_max: int = Field(description="满分")
    correct_count: int = Field(description="正确题数")
    questions_count: int = Field(description="总题数")
    questions: List[QuestionResult] = Field(description="每题批改结果")
    annotated_image_url: str = Field(description="带批注图片地址")


# ========== 错题本 & 统计 相关模型 ==========

class ErrorEntryData(BaseModel):
    id: str = Field(description="错题ID")
    created_at: str = Field(description="创建时间")
    question_no: int = Field(description="题号")
    question_text: str = Field(description="题目文字")
    student_answer: str = Field(description="学生答案")
    subject: str = Field(description="学科")
    topic: str = Field(description="知识点")
    difficulty: str = Field(description="难度")
    score: float = Field(description="得分")
    max_score: float = Field(description="满分")
    is_correct: bool = Field(description="是否正确")
    comment: str = Field(description="评语/解析")
    explanation: str = Field(description="详细解析")
    image_url: str = Field(description="原图链接")
    is_reviewed: bool = Field(description="是否已复习")


class ErrorListResponse(BaseModel):
    entries: List[ErrorEntryData] = Field(description="错题列表")
    total: int = Field(description="总数量")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页数量")
    total_pages: int = Field(description="总页数")


class StatsResponse(BaseModel):
    total: int = Field(description="总错题数")
    reviewed: int = Field(description="已复习数")
    unreviewed: int = Field(description="未复习数")
    review_rate: float = Field(description="复习率(%)")
    by_subject: dict = Field(description="按学科分布")
    by_topic: dict = Field(description="按知识点分布(TOP20)")
    by_difficulty: dict = Field(description="按难度分布")
    recent_week: int = Field(description="最近7天新增")
    recent_month: int = Field(description="最近30天新增")


class RenameRequest(BaseModel):
    question_text: str = Field(description="新的题目名称")


class BatchDeleteRequest(BaseModel):
    ids: List[str] = Field(description="要删除的错题ID列表")


class MessageResponse(BaseModel):
    message: str = Field(description="操作结果消息")


# ========== 工具函数 ==========

def get_api_ocr():
    from src.modules.aliyun_ocr import AliyunOCR
    return AliyunOCR()


def get_api_grader():
    from src.modules.llm_grader import LLMGrader
    return LLMGrader()


def get_api_renderer():
    from src.modules.annotation_renderer import AnnotationRenderer
    return AnnotationRenderer()


def get_api_processor():
    from src.modules.image_processor import ImageProcessor
    return ImageProcessor()


async def run_correction_pipeline(image_path: str) -> dict:
    """原有的一键批改流水线"""
    ocr = get_api_ocr()
    grader = get_api_grader()
    renderer = get_api_renderer()

    import cv2
    image = cv2.imread(image_path)

    ocr_result = ocr.recognize_text(image_path)
    text = ocr.parse_result(ocr_result)
    regions = ocr.get_text_regions(ocr_result)

    grade_result = grader.grade(
        ocr_text=text,
        reference_answer="",
    )

    corrections = []
    for i, region in enumerate(regions):
        if not region.get("text"):
            continue
        cx = region["x"] + region["width"] // 2
        cy = region["y"] + region["height"] // 2
        corrections.append({
            "position": (cx, cy),
            "is_correct": grade_result.get("is_correct", False),
            "comment": grade_result.get("comment", "") if i == 0 else "",
            "score": grade_result.get("score", 0),
            "max_score": grade_result.get("max_score", 0),
        })

    annotated = renderer.render_all(image, corrections)
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = str(output_dir / "api_result.jpg")
    cv2.imwrite(annotated_path, annotated)

    return {
        "ocr_text": text,
        "is_correct": grade_result.get("is_correct"),
        "score": grade_result.get("score"),
        "max_score": grade_result.get("max_score"),
        "comment": grade_result.get("comment"),
        "annotated_image_url": f"/image/{os.path.basename(annotated_path)}",
    }


# ========== API 接口 ==========

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    """返回交互式前端页面"""
    html_path = static_dir / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>作业批改系统</h1><p>前端页面未找到，请检查 static/index.html</p>"


@app.get(
    "/health",
    summary="健康检查",
    description="检查服务是否正常运行，不消耗任何 API 额度。可用于监控系统可用性。",
    tags=["系统"],
    response_model=HealthResponse,
)
async def health_check():
    return {"status": "ok", "service": "作业批改系统", "version": "2.0.0"}


@app.post(
    "/upload",
    summary="上传图片并自动切题（支持多图）",
    description="""
上传一张或多张作业图片。多图场景下，**每张图独立切题**，然后将所有题目
汇总编号后展示在拼接预览图上。

**两种场景**：
- **一张图多道题**：常规场景，切题 API 自动识别每道题的区域
- **一道题跨多张图**：每张图独立切题，题目跨页的由用户在预览中手动调整选区合并

**返回内容**：
- `upload_id`：用于后续 `/grade-selected` 接口
- `image_url`：拼接预览图地址（单张则为原图）
- `questions`：所有题目区域列表（已跨页重新编号并修正坐标）
- `stitched_from`：由几张图片拼接而成
""",
    tags=["交互式批改"],
    response_model=UploadResponse,
    responses={
        200: {"description": "切题成功，返回区域列表"},
        500: {"description": "切题失败", "model": ErrorResponse},
    },
)
async def upload_and_cut(
    files: List[UploadFile] = File(
        ...,
        description="作业图片文件（可多选），支持 JPG / PNG / BMP / WebP",
    ),
):
    if not files:
        return JSONResponse(
            status_code=400,
            content={"error": "请至少上传一张图片"},
        )

    upload_id = str(uuid.uuid4())
    tmp_paths = []
    page_count = 0

    try:
        # —— 第 1 步：保存所有上传图片 ——
        for i, f in enumerate(files):
            ext = Path(f.filename).suffix or ".jpg"
            tmp_path = os.path.join(tempfile.gettempdir(), f"{upload_id}_p{i}{ext}")
            content = await f.read()
            with open(tmp_path, "wb") as fh:
                fh.write(content)
            tmp_paths.append(tmp_path)
            page_count += 1

        # —— 第 2 步：复制原图到静态目录 ——
        static_img_dir = static_dir / "uploads"
        static_img_dir.mkdir(exist_ok=True)

        for i, tp in enumerate(tmp_paths):
            saved_name = f"{upload_id}_page{i+1}.jpg"
            saved_path = static_img_dir / saved_name
            shutil.copy2(tp, str(saved_path))

        # —— 第 3 步：每张图独立调用切题 API，收集所有题目区域 ——
        ocr = get_api_ocr()
        all_questions = []      # 聚合后的问题列表（含修正后的坐标）
        page_heights = []       # 每页缩放后的高度，用于计算 Y 偏移
        page_original_imgs = [] # 每页缩放后的图像（用于拼接）

        from src.modules.image_processor import ImageProcessor

        for page_idx, tp in enumerate(tmp_paths):
            # 3a. 预处理：缩放统一宽度
            img = cv2.imread(tp)
            if img is None:
                # 回退：np.fromfile 处理中文路径
                img_array = np.fromfile(tp, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                logger.error(f"无法读取第 {page_idx+1} 页图片: {tp}")
                continue

            h, w = img.shape[:2]
            target_w = 1200
            if w != target_w:
                ratio = target_w / w
                img = cv2.resize(img, (target_w, int(h * ratio)))
                h, w = img.shape[:2]

            page_heights.append(h)
            page_original_imgs.append(img)

            # 3b. 对当前页独立调用切题 API
            page_tmp = os.path.join(tempfile.gettempdir(), f"{upload_id}_cut_p{page_idx}.jpg")
            cv2.imwrite(page_tmp, img)

            try:
                cut_result = ocr.recognize_edu_paper_cut(page_tmp)
                questions = ocr.get_question_regions(cut_result)
            except Exception as cut_err:
                logger.warning(f"第 {page_idx+1} 页切题失败: {cut_err}，返回空列表")
                questions = []
            finally:
                if os.path.exists(page_tmp):
                    os.remove(page_tmp)

            # 3c. 处理当前页切出的题目区域
            for q in questions:
                all_questions.append({
                    "question_no": 0,  # 稍后重新编号
                    "page": page_idx + 1,
                    "x": q.get("x", 0),
                    "y": q.get("y", 0),
                    "width": q.get("width", 0),
                    "height": q.get("height", 0),
                    "text": q.get("text", ""),
                    "y_offset": 0,  # 原始 Y（相对当前页），稍后修正
                })

        # —— 第 4 步：重新编号 + 修正坐标（Y 偏移为当前页在拼接图中的起始位置） ——
        y_cumulative = 0
        separator_h = 20   # 页间分隔线高度

        for page_idx in range(len(page_original_imgs)):
            # 更新当前页所有题目的编号和 Y 偏移
            page_questions = [q for q in all_questions if q["page"] == page_idx + 1]
            for q in page_questions:
                q["y_offset"] = y_cumulative
                q["y"] = q["y"] + y_cumulative  # 修正为拼接图中的绝对 Y
            y_cumulative += page_heights[page_idx]
            if page_idx < len(page_original_imgs) - 1:
                y_cumulative += separator_h  # 分隔线占用的高度

        # 全局重新编号
        for i, q in enumerate(all_questions):
            q["question_no"] = i + 1

        # —— 切题回退：如果所有页面都没切出题目，每页创建一整图区域 ——
        if not all_questions:
            logger.warning("切题 API 未识别到题目（可能是截图而非扫描件），自动将每页整图作为题目区域")
            y_cumulative = 0
            for page_idx, img in enumerate(page_original_imgs):
                h, w = img.shape[:2]
                all_questions.append({
                    "question_no": page_idx + 1,
                    "page": page_idx + 1,
                    "x": 0,
                    "y": y_cumulative,
                    "width": w,
                    "height": h,
                    "text": f"第{page_idx+1}页 整页内容",
                    "y_offset": y_cumulative,
                })
                y_cumulative += h
                if page_idx < len(page_original_imgs) - 1:
                    y_cumulative += separator_h

        # —— 第 5 步：拼接图片（仅用于预览） ——
        if page_count == 1:
            stitched = page_original_imgs[0]
        else:
            parts = []
            for i, img in enumerate(page_original_imgs):
                parts.append(img)
                if i < page_count - 1:
                    # 分隔线
                    sep = np.ones((separator_h, target_w, 3), dtype=np.uint8) * 255
                    cv2.line(sep, (50, separator_h // 2), (target_w - 50, separator_h // 2),
                             (200, 200, 200), 1)
                    # 在分隔线上标注页码
                    cv2.putText(sep, f"--- 第 {i+2} 页 ---", (target_w // 2 - 60, separator_h // 2 + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                    parts.append(sep)
            stitched = np.vstack(parts)

        stitched_name = f"{upload_id}_stitched.jpg"
        stitched_path = static_img_dir / stitched_name
        cv2.imwrite(str(stitched_path), stitched)

        # —— 第 6 步：构建响应 ——
        question_models = [
            QuestionRegion(
                question_no=q["question_no"],
                x=q["x"],
                y=q["y"],
                width=q["width"],
                height=q["height"],
                text=q.get("text", ""),
            )
            for q in all_questions
        ]

        # 存储上传信息（grade-selected 使用拼接图 + 修正后的坐标）
        upload_store[upload_id] = {
            "image_path": str(stitched_path),
            "questions": all_questions,
            "original_pages": page_count,
        }

        page_hint = f"（{page_count} 张图片，每张独立切题后汇总）" if page_count > 1 else ""
        q_count = len(all_questions)
        if q_count == 0:
            page_hint += " 未识别到题目区域，请在预览图中手动框选"

        return {
            "upload_id": upload_id,
            "image_url": f"/static/uploads/{stitched_name}",
            "questions": question_models,
            "message": f"切题完成，共识别 {q_count} 道题目{page_hint}",
            "stitched_from": page_count,
        }
    except Exception as e:
        logger.exception("切题失败")
        return JSONResponse(
            status_code=500,
            content={"error": f"切题失败: {str(e)}"},
        )
    finally:
        for tp in tmp_paths:
            if os.path.exists(tp):
                os.remove(tp)


@app.post(
    "/grade-selected",
    summary="按用户选定区域批改",
    description="""
接收用户确认后的题目区域，按每个区域裁剪图片、OCR识别、AI批改，最后汇总结果。

**请求参数**：
- `upload_id`：`/upload` 接口返回的标识
- `questions`：用户调整后的题目区域列表

**返回内容**：
- 每道题的批改结果（对错、得分、评语）
- 总分汇总
- 带批注的图片地址
""",
    tags=["交互式批改"],
    response_model=GradeSelectedResponse,
    responses={
        200: {"description": "批改成功"},
        404: {"description": "upload_id 不存在", "model": ErrorResponse},
        500: {"description": "批改失败", "model": FailedResponse},
    },
)
async def grade_selected_regions(body: GradeSelectedRequest):
    upload_id = body.upload_id
    if upload_id not in upload_store:
        return JSONResponse(
            status_code=404,
            content={"error": f"上传任务 {upload_id} 不存在或已过期"},
        )

    upload_info = upload_store[upload_id]
    image_path = upload_info["image_path"]
    questions = body.questions

    task_id = str(uuid.uuid4())

    try:
        grader = get_api_grader()
        renderer = get_api_renderer()

        image = cv2.imread(image_path)
        if image is None:
            # 回退：处理中文路径
            img_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # —— 回退策略：如果用户没有框选任何题目，整张图作为一道题 ——
        if not questions:
            img_h, img_w = image.shape[:2]
            logger.info("未检测到题目选区，将整张图片作为一道题目进行 OCR 识别")
            questions = [
                QuestionRegion(
                    question_no=1,
                    x=0, y=0,
                    width=img_w, height=img_h,
                    text="",
                )
            ]

        # —— 整页批改：将完整图片发给视觉模型一次批改所有题目 ——
        # 不再按区裁剪，让 Qwen-VL 直接看整页，避免上下文丢失
        full_page_path = os.path.join(tempfile.gettempdir(), f"{upload_id}_fullpage.jpg")
        cv2.imwrite(full_page_path, image)

        questions_results = []
        corrections = []

        try:
            grade_result = grader.grade_full_page(
                image_path=full_page_path,
                reference_answer="",
            )
            q_text = grade_result.get("ocr_text", "")
        except Exception as grade_err:
            logger.exception(f"整页批改失败: {grade_err}")
            q_text = "[批改失败]"
            grade_result = {
                "is_correct": False, "score": 0, "max_score": 0,
                "comment": f"批改出错: {str(grade_err)[:100]}",
                "subject": "未分类", "topic": "未分类", "difficulty": "中等",
                "student_answer_found": False, "sub_questions": [],
                "ocr_text": "",
            }

        # 保留整图用于错题本（复制到 output 目录）
        saved_image_name = f"{upload_id}_original.jpg"
        saved_image_path = Path("data/output") / saved_image_name
        Path("data/output").mkdir(parents=True, exist_ok=True)
        shutil.copy2(full_page_path, str(saved_image_path))

        # 作为一道大题，子题嵌套在内
        sub_questions = grade_result.get("sub_questions", [])
        questions_results.append({
            "question_no": 1,
            "is_correct": grade_result.get("is_correct", False),
            "score": grade_result.get("score", 0),
            "max_score": grade_result.get("max_score", 0),
            "comment": grade_result.get("comment", ""),
            "ocr_text": q_text or "",
            "subject": grade_result.get("subject", "未分类"),
            "topic": grade_result.get("topic", "未分类"),
            "difficulty": grade_result.get("difficulty", "中等"),
            "student_answer_found": grade_result.get("student_answer_found", True),
            "sub_questions": sub_questions,
            "image_url": f"/image/{saved_image_name}",
        })

        # 每个小题在批注图上有独立标记（从上到下均匀分布）
        img_h, img_w = image.shape[:2]
        for i, sq in enumerate(sub_questions):
            y_pos = int(img_h * (i + 1) / (len(sub_questions) + 1)) if sub_questions else img_h // 2
            corrections.append({
                "position": (img_w // 2, y_pos),
                "is_correct": sq.get("is_correct", False),
                "comment": f"{sq.get('label','')}: {sq.get('comment','')}",
                "score": sq.get("score", 0),
                "max_score": sq.get("max_score", 0),
            })
        if not sub_questions:
            corrections.append({
                "position": (img_w // 2, img_h // 2),
                "is_correct": grade_result.get("is_correct", False),
                "comment": grade_result.get("comment", ""),
                "score": grade_result.get("score", 0),
                "max_score": grade_result.get("max_score", 0),
            })

        # 清理全图临时文件
        try:
            if os.path.exists(full_page_path):
                os.remove(full_page_path)
        except Exception:
            pass

        # 渲染批注（每个小题在图片上标记区域）
        region_dicts = []
        for i, sq in enumerate(sub_questions):
            y_pos = int(img_h * (i + 1) / (len(sub_questions) + 1)) if sub_questions else img_h // 2
            region_dicts.append({
                "x": 10, "y": y_pos - 15,
                "width": img_w - 20, "height": 30,
            })
        if not region_dicts:
            region_dicts.append({"x": 10, "y": 10, "width": img_w - 20, "height": img_h - 20})
        annotated = renderer.render_all_with_regions(image, corrections, region_dicts)
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = str(output_dir / f"{upload_id}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)

        total_score = sum(q["score"] for q in questions_results)
        total_max = sum(q["max_score"] for q in questions_results)
        # 统计小题：汇总所有 sub_questions 的对错
        all_subs = [s for q in questions_results for s in q.get("sub_questions", [])]
        correct_count = sum(1 for s in all_subs if s.get("is_correct")) if all_subs else sum(1 for q in questions_results if q["is_correct"])
        total_question_count = len(all_subs) if all_subs else len(questions_results)

        result_data = {
            "task_id": task_id,
            "message": "批改完成",
            "total_score": total_score,
            "total_max": total_max,
            "correct_count": correct_count,
            "questions_count": total_question_count,
            "questions": questions_results,
            "annotated_image_url": f"/image/{os.path.basename(annotated_path)}",
        }
        task_store[task_id] = result_data
        return result_data
    except Exception as e:
        logger.exception("按选区批改失败")
        return JSONResponse(
            status_code=500,
            content={
                "task_id": task_id,
                "message": "批改失败",
                "error": str(e),
            },
        )


@app.post(
    "/correct",
    summary="上传作业图片并一键批改",
    description="""
上传一张作业图片，系统自动完成 **OCR 识别 + AI 批改 + 批注渲染**（一键完成，不经过交互式选区）。

**支持格式**：JPG、PNG、BMP、WebP  
**建议大小**：小于 2MB，图片清晰无过度倾斜  
**处理耗时**：约 10~15 秒（含 OCR + AI 批改）

**使用流程**：提交图片 → 获取 task_id → 调用 /result/{task_id} 查看结果
""",
    tags=["批改"],
    response_model=CorrectResponse,
    responses={
        200: {"description": "批改成功"},
        500: {"description": "批改失败", "model": FailedResponse},
    },
)
async def correct_homework(
    file: UploadFile = File(
        ...,
        description="作业图片文件，支持 JPG / PNG / BMP",
    ),
    subject: Optional[str] = Query(
        None,
        description="学科类型（可选，如：数学、语文、英语）",
    ),
):
    task_id = str(uuid.uuid4())
    tmp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{file.filename}")

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = await run_correction_pipeline(tmp_path)
        task_store[task_id] = result

        return JSONResponse(content={
            "task_id": task_id,
            "message": "批改完成",
            "result": result,
        })
    except Exception as e:
        logger.exception("批改失败")
        return JSONResponse(
            status_code=500,
            content={
                "task_id": task_id,
                "message": "批改失败",
                "error": str(e),
            },
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get(
    "/result/{task_id}",
    summary="查询批改结果",
    description="根据提交批改任务时返回的 `task_id`，查询该次批改的完整结果。",
    tags=["批改"],
    response_model=ResultQueryResponse,
    responses={
        200: {"description": "查询成功"},
        404: {"description": "任务不存在", "model": ErrorResponse},
    },
)
async def get_result(
    task_id: str = FastAPIPath(..., description="批改任务唯一标识"),
):
    if task_id in task_store:
        return {"task_id": task_id, "result": task_store[task_id]}
    return JSONResponse(
        status_code=404,
        content={"error": f"任务 {task_id} 不存在或已过期"},
    )


@app.get(
    "/image/{image_name}",
    summary="查看批注图片",
    description="获取带批注标注的作业图片。",
    tags=["批改"],
    responses={
        200: {"description": "返回 JPEG 图片"},
        404: {"description": "图片不存在", "model": ErrorResponse},
    },
)
async def get_annotated_image(
    image_name: str = FastAPIPath(..., description="图片文件名"),
):
    image_path = os.path.join("data/output", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/jpeg")
    return JSONResponse(
        status_code=404,
        content={"error": f"图片 {image_name} 不存在"},
    )


# ========== 错题本 API ==========

@app.get(
    "/api/error-notebook",
    summary="获取错题本列表",
    description="分页获取错题本中的所有错题，支持按学科、复习状态、关键词筛选。",
    tags=["错题本"],
    response_model=ErrorListResponse,
)
async def list_errors(
    subject: Optional[str] = Query(None, description="按学科筛选"),
    reviewed: Optional[bool] = Query(None, description="按复习状态筛选"),
    keyword: Optional[str] = Query(None, description="按关键词搜索（题目/知识点/评语）"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    result = error_notebook.get_all(
        subject=subject,
        reviewed=reviewed,
        keyword=keyword,
        page=page,
        page_size=page_size,
    )
    return result


@app.get(
    "/api/error-notebook/{entry_id}",
    summary="获取单条错题详情",
    description="根据 ID 获取某道错题的完整信息（含解析）。",
    tags=["错题本"],
    responses={
        200: {"description": "错题详情"},
        404: {"description": "错题不存在", "model": ErrorResponse},
    },
)
async def get_error_detail(
    entry_id: str = FastAPIPath(..., description="错题ID"),
):
    entry = error_notebook.get_by_id(entry_id)
    if entry is None:
        return JSONResponse(status_code=404, content={"error": "错题不存在"})
    return entry


@app.delete(
    "/api/error-notebook/{entry_id}",
    summary="删除单条错题",
    description="从错题本中删除指定的一条错题。",
    tags=["错题本"],
    responses={
        200: {"description": "删除成功", "model": MessageResponse},
        404: {"description": "错题不存在", "model": ErrorResponse},
    },
)
async def delete_error(
    entry_id: str = FastAPIPath(..., description="错题ID"),
):
    ok = error_notebook.delete_entry(entry_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "错题不存在"})
    return {"message": "删除成功"}


@app.post(
    "/api/error-notebook/batch-delete",
    summary="批量删除错题",
    description="一次性从错题本中删除多条错题。",
    tags=["错题本"],
    response_model=MessageResponse,
)
async def batch_delete_errors(body: BatchDeleteRequest):
    count = error_notebook.batch_delete(body.ids)
    return {"message": f"已删除 {count} 条错题"}


@app.patch(
    "/api/error-notebook/{entry_id}/review",
    summary="标记为已复习",
    description='将指定错题标记为「已复习」状态。',
    tags=["错题本"],
    responses={
        200: {"description": "标记成功", "model": MessageResponse},
        404: {"description": "错题不存在", "model": ErrorResponse},
    },
)
async def mark_reviewed(
    entry_id: str = FastAPIPath(..., description="错题ID"),
):
    ok = error_notebook.mark_reviewed(entry_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "错题不存在"})
    return {"message": "已标记为已复习"}


@app.patch(
    "/api/error-notebook/{entry_id}/unreview",
    summary="标记为未复习",
    description='将指定错题恢复为「未复习」状态。',
    tags=["错题本"],
    responses={
        200: {"description": "标记成功", "model": MessageResponse},
        404: {"description": "错题不存在", "model": ErrorResponse},
    },
)
@app.patch(
    "/api/error-notebook/{entry_id}/rename",
    summary="重命名错题",
    description="修改错题的题目名称（question_text）。",
    tags=["错题本"],
    response_model=MessageResponse,
)
async def rename_error(
    entry_id: str = FastAPIPath(..., description="错题ID"),
    body: RenameRequest = None,
):
    new_text = body.question_text if body else ""
    ok = error_notebook.rename_entry(entry_id, new_text)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "错题不存在"})
    return {"message": "重命名成功"}


@app.patch(
    "/api/error-notebook/{entry_id}/unreview",
    summary="标记为未复习",
    description='将指定错题恢复为「未复习」状态。',
    tags=["错题本"],
    responses={
        200: {"description": "标记成功", "model": MessageResponse},
        404: {"description": "错题不存在", "model": ErrorResponse},
    },
)
async def mark_unreviewed(
    entry_id: str = FastAPIPath(..., description="错题ID"),
):
    ok = error_notebook.mark_unreviewed(entry_id)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "错题不存在"})
    return {"message": "已标记为未复习"}


@app.patch(
    "/api/error-notebook/mark-all-reviewed",
    summary="全部标记为已复习",
    description="将错题本中所有错题标记为已复习。",
    tags=["错题本"],
    response_model=MessageResponse,
)
async def mark_all_reviewed():
    count = error_notebook.mark_all_reviewed()
    return {"message": f"已标记 {count} 条错题为已复习"}


# ========== 统计分析 API ==========

@app.get(
    "/api/stats",
    summary="获取错题统计数据",
    description="返回错题本的整体统计数据：按学科/知识点/难度分布、复习率等。",
    tags=["统计分析"],
    response_model=StatsResponse,
)
async def get_stats():
    return error_notebook.get_stats()


@app.get(
    "/api/subjects",
    summary="获取学科列表",
    description="返回错题本中出现过的所有学科（供前端筛选用）。",
    tags=["统计分析"],
)
async def get_subjects():
    subjects = error_notebook.get_subjects()
    return {"subjects": subjects}


# ========== 手动保存错题本 ==========

class SaveToNotebookRequest(BaseModel):
    task_id: str = Field(description="批改任务 ID")
    questions: List[dict] = Field(description="要保存的题目列表，每项含 save 标记")


@app.post(
    "/api/error-notebook/save-batch",
    summary="手动保存错题到错题本（带去重）",
    description="用户选择要保存的题目，系统自动去重。同题号+同学科+相同题目文本不会重复添加。",
    tags=["错题本"],
    response_model=MessageResponse,
)
async def save_to_notebook(body: SaveToNotebookRequest):
    def _auto_detect_subject(question_text: str, subject: str) -> str:
        """自动检测学科，如果模型没分类则根据文本内容推断"""
        if subject and subject != "未分类":
            return subject
        text = (question_text or "").lower()
        # 英文特征：大量英文字母
        eng_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        if eng_chars > 30:
            return "英语"
        # 数学特征
        if any(c in text for c in "0123456789"):
            if any(w in text for w in ["方程", "函数", "几何", "计算", "面积", "体积"]):
                return "数学"
        # 中文特征
        chn_chars = sum(1 for c in text if '一' <= c <= '鿿')
        if chn_chars > 10:
            return "语文"
        return "其他"

    saved = 0
    skipped_dup = 0
    for q in body.questions:
        if not q.get("save"):
            continue
        q_no = q.get("question_no", 0)
        q_text = q.get("ocr_text", q.get("comment", ""))
        subject = _auto_detect_subject(q_text, q.get("subject", "未分类"))

        if error_notebook.is_duplicate(q_no, subject, q_text):
            skipped_dup += 1
            logger.info(f"跳过重复错题: 题号{q_no} 学科{subject}")
            continue

        # 将小题详情格式化为文本
        subs = q.get("sub_questions", [])
        subs_text = ""
        for s in subs:
            status = "✓" if s.get("is_correct") else "✗"
            subs_text += f"[{status}] {s.get('label','')} ({s.get('score',0)}/{s.get('max_score',0)}分): {s.get('comment','')}\n"

        error_notebook.add_entry({
            "question_no": q_no,
            "question_text": q_text,
            "student_answer": q.get("student_answer", ""),
            "subject": subject,
            "topic": q.get("topic", "未分类"),
            "difficulty": q.get("difficulty", "中等"),
            "score": q.get("score", 0),
            "max_score": q.get("max_score", 0),
            "is_correct": q.get("is_correct", False),
            "comment": q.get("comment", ""),
            "explanation": subs_text if subs_text else q.get("comment", ""),
            "image_url": q.get("image_url", ""),
            "sub_questions": subs,
        })
        saved += 1

    msg = f"已保存 {saved} 道错题"
    if skipped_dup > 0:
        msg += f"（跳过 {skipped_dup} 道重复）"
    return {"message": msg}


@app.delete(
    "/api/error-notebook/clear-all/all",
    summary="清空错题本",
    description="⚠️ 永久删除所有错题记录，不可恢复。",
    tags=["错题本"],
    response_model=MessageResponse,
)
async def clear_all_errors():
    count = error_notebook.clear_all()
    return {"message": f"已清空所有 {count} 条错题"}
