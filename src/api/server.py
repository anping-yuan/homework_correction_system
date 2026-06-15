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
import base64
import hashlib
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
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# 错题本实例（全局单例，基于 MySQL）
from src.modules.error_notebook import ErrorNotebook
from src.modules.db_manager import DBManager
error_notebook = ErrorNotebook()
db = DBManager()

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
    db_connected: bool = Field(default=False, description="数据库连接状态")
    db_error: Optional[str] = Field(default=None, description="数据库错误信息")


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
    page: int = Field(default=1, description="所属页码（多页场景下标记该题目来自第几页）")


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
    grading_mode: Optional[str] = Field(default="vision", description="批改模式: vision(视觉模型直接批改) | ocr_llm(OCR提取文字+大模型批改) | vision_deepseek(Vision提取+DeepSeek判题)")
    force: bool = Field(default=False, description="强制重新批改，跳过缓存")


class RedoGradingRequest(BaseModel):
    task_id: str = Field(description="原批改任务ID")
    question_no: int = Field(description="题目编号")
    upload_id: Optional[str] = Field(default=None, description="当前上传会话ID（图片回退用）")
    sub_label: Optional[str] = Field(default=None, description="小题编号（可选，用于 Vision 模式指定重做哪个小题）")
    hint: Optional[str] = Field(default="", description="老师提示（可选）")


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


def compute_image_hash(file_path: str) -> str:
    """计算图片文件的 MD5 哈希（用于缓存去重）"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        # 分块读取，大文件也不怕
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _save_grading_results(
    upload_id: str,
    task_id: str,
    questions_results: list,
    annotated_path: str,
    questions: list,
    grading_mode: str,
    upload_info: dict,
    image_hash: str = "",
) -> None:
    """将批改结果持久化到 MySQL grading_records 表"""
    import json as _json

    annotated_url = f"/image/{os.path.basename(annotated_path)}"
    # 构建原图路径映射（question_no → original_image_url）
    page_images = upload_info.get("page_images", [])

    records = []
    for q in questions_results:
        sub_qs = q.get("sub_questions", [])
        # 查找对应 question 对象的坐标信息
        q_no = q["question_no"]
        matched = [r for r in questions if r.question_no == q_no]
        region = matched[0] if matched else None

        # 确定原图 URL
        q_page = getattr(region, "page", 1) if region else 1
        original_url = ""
        if page_images and q_page <= len(page_images):
            original_url = page_images[q_page - 1]

        records.append({
            "upload_id": upload_id,
            "task_id": task_id,
            "question_no": q_no,
            "page": q_page,
            "question_text": q.get("ocr_text", "") or "",
            # 学生答案：优先用模型返回的 student_answer，回退到 sub_questions 拼接
            "student_answer": q.get("student_answer", "") or
                "; ".join(sq.get("comment", "") for sq in sub_qs if not sq.get("is_correct")) or "",
            "is_correct": 1 if q.get("is_correct") else 0,
            "score": q.get("score", 0),
            "max_score": q.get("max_score", 0),
            "comment": q.get("comment", ""),
            "explanation": "",
            "subject": q.get("subject", "未分类"),
            "topic": q.get("topic", "未分类"),
            "difficulty": q.get("difficulty", "中等"),
            "sub_questions": _json.dumps(sub_qs, ensure_ascii=False) if sub_qs else None,
            "original_image": original_url,
            "annotated_image": annotated_url,
            "crop_image": q.get("image_url", ""),
            "region_x": region.x if region else None,
            "region_y": region.y if region else None,
            "region_width": region.width if region else None,
            "region_height": region.height if region else None,
            "grading_mode": grading_mode,
            "image_hash": image_hash,
            "in_notebook": 0,
            "is_reviewed": 0,
        })

    try:
        count = db.insert_batch(records)
        logger.info(f"批改结果已入库: task_id={task_id}, 共 {count} 条记录")
    except Exception as e:
        logger.exception(f"批改结果入库失败: {e}")


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


@app.get("/enhance-view", response_class=HTMLResponse, include_in_schema=False)
async def enhance_view(upload_id: str = "", page: int = 1):
    """返回图片修复预览独立页面"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>图片修复预览</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f0f2f5; color:#333; min-height:100vh; display:flex; flex-direction:column; }}
.header {{ background:linear-gradient(135deg,#10b981,#059669); color:#fff; padding:14px 28px; display:flex; justify-content:space-between; align-items:center; flex-shrink:0; }}
.header h1 {{ font-size:1.15rem; font-weight:600; letter-spacing:.02em; }}
.header a {{ color:#fff; text-decoration:none; font-size:.85rem; opacity:.85; transition:opacity .2s; }}
.header a:hover {{ opacity:1; }}
.toolbar {{ display:flex; gap:10px; padding:12px 28px; background:#fff; border-bottom:1px solid #e5e7eb; align-items:center; flex-shrink:0; }}
.toolbar label {{ font-size:.85rem; color:#666; }}
.toolbar select,.toolbar button {{ padding:7px 16px; border:1px solid #d1d5db; border-radius:6px; font-size:.85rem; cursor:pointer; background:#fff; }}
.toolbar button {{ background:#10b981; color:#fff; border-color:#10b981; font-weight:500; transition:background .2s; }}
.toolbar button:hover {{ background:#059669; }}
.compare {{ flex:1; display:flex; align-items:stretch; justify-content:center; gap:0; background:#e5e7eb; min-height:0; }}
.side {{ flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#fff; padding:32px 24px; position:relative; }}
.side + .side {{ border-left:2px solid #e5e7eb; }}
.side h2 {{ font-size:.85rem; font-weight:600; margin-bottom:16px; padding:5px 16px; border-radius:20px; letter-spacing:.03em; }}
.side h2.orig {{ background:#fef2f2; color:#dc2626; }}
.side h2.enh {{ background:#ecfdf5; color:#059669; }}
.img-wrap {{ display:flex; align-items:center; justify-content:center; flex:1; width:100%; }}
.side img {{ max-width:95%; max-height:75vh; object-fit:contain; border-radius:8px; box-shadow:0 2px 16px rgba(0,0,0,.06); }}
.loading {{ padding:60px 20px; text-align:center; color:#9ca3af; font-size:.9rem; }}
.steps {{ padding:12px 28px; background:#fff; border-top:1px solid #e5e7eb; font-size:.82rem; color:#666; display:flex; gap:12px; flex-wrap:wrap; align-items:center; flex-shrink:0; }}
.step-tag {{ padding:4px 12px; border-radius:12px; font-size:.75rem; font-weight:500; }}
.step-tag.on {{ background:#d1fae5; color:#059669; }}
@media(max-width:768px){{ .compare{{flex-direction:column;}} .side+.side{{border-left:0;border-top:2px solid #e5e7eb;}} }}
</style>
</head>
<body>
<div class="header">
    <h1>🔧 图片修复预览 — 原图 vs 增强后</h1>
    <a href="/" target="_blank">← 返回批改系统</a>
</div>
<div class="toolbar" id="toolbar">
    <label>📄 页面：</label>
    <select id="pageSelect" onchange="loadPage(this.value)"></select>
    <button onclick="loadPage(currentPage)">🔄 刷新</button>
    <span style="font-size:.8rem;color:#9ca3af;margin-left:auto;" id="infoSpan"></span>
</div>
<div class="compare">
    <div class="side">
        <h2 class="orig">📷 原图</h2>
        <div id="origLoading" class="loading">加载中…</div>
        <div class="img-wrap"><img id="origImg" style="display:none;" alt="原图"></div>
    </div>
    <div class="side">
        <h2 class="enh">✨ 增强后</h2>
        <div id="enhLoading" class="loading">处理中…</div>
        <div class="img-wrap"><img id="enhImg" style="display:none;" alt="增强后"></div>
    </div>
</div>
<div class="steps" id="stepsInfo">
    <span style="font-weight:600;">预处理步骤：</span><span id="stepsTags">等待加载…</span>
</div>
<script>
const UPLOAD_ID = '{upload_id}';
const START_PAGE = {page};
let currentPage = START_PAGE;
let pageImages = [];

const STEP_LABELS = {{
    illumination_norm: '✓ 光照归一化(去阴影)',
    bilateral_denoise: '✓ 双边滤波(保边去噪)',
    clahe: '✓ CLAHE对比度增强',
    usm_sharpen: '✓ USM锐化',
}};

async function init() {{
    if (!UPLOAD_ID) {{
        document.body.innerHTML = '<div style="padding:40px;text-align:center;"><h2>缺少 upload_id 参数</h2><p>请从批改系统点击"修复预览"进入</p><a href="/">返回首页</a></div>';
        return;
    }}
    // 获取页面信息
    try {{
        const res = await fetch('/enhance-image', {{
            method:'POST', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{upload_id: UPLOAD_ID, page: 1}})
        }});
        if (!res.ok) throw new Error((await res.json()).error);
        // 探测总页数
        let totalPages = 1;
        for (let p = 1; p <= 20; p++) {{
            const r = await fetch('/enhance-image', {{
                method:'POST', headers:{{'Content-Type':'application/json'}},
                body: JSON.stringify({{upload_id: UPLOAD_ID, page: p}})
            }});
            if (!r.ok) {{ totalPages = p - 1; break; }}
            totalPages = p;
        }}
        const sel = document.getElementById('pageSelect');
        for (let i = 1; i <= totalPages; i++) {{
            const opt = document.createElement('option');
            opt.value = i; opt.textContent = '第' + i + '页';
            if (i === START_PAGE) opt.selected = true;
            sel.appendChild(opt);
        }}
        if (totalPages <= 1) document.getElementById('toolbar').style.display = 'none';
        loadPage(START_PAGE);
    }} catch(e) {{
        document.getElementById('origLoading').textContent = '加载失败: ' + e.message;
        document.getElementById('enhLoading').textContent = '加载失败';
    }}
}}

async function loadPage(page) {{
    currentPage = page;
    document.getElementById('origImg').style.display = 'none';
    document.getElementById('enhImg').style.display = 'none';
    document.getElementById('origLoading').style.display = 'block';
    document.getElementById('enhLoading').style.display = 'block';
    document.getElementById('enhLoading').textContent = '处理中…';

    try {{
        const res = await fetch('/enhance-image', {{
            method:'POST', headers:{{'Content-Type':'application/json'}},
            body: JSON.stringify({{upload_id: UPLOAD_ID, page: page}})
        }});
        if (!res.ok) throw new Error((await res.json()).error);
        const data = await res.json();

        document.getElementById('origImg').src = data.original_url;
        document.getElementById('origImg').style.display = 'block';
        document.getElementById('origLoading').style.display = 'none';

        document.getElementById('enhImg').src = data.enhanced_base64;
        document.getElementById('enhImg').style.display = 'block';
        document.getElementById('enhLoading').style.display = 'none';

        document.getElementById('infoSpan').textContent =
            data.enhanced_width + 'x' + data.enhanced_height;
        const tags = Object.entries(data.preprocessing || {{}})
            .map(([k,v]) => '<span class="step-tag ' + (v?'on':'off') + '">' + (STEP_LABELS[k]||k) + '</span>')
            .join('');
        document.getElementById('stepsTags').innerHTML = tags;
    }} catch(e) {{
        document.getElementById('enhLoading').textContent = '失败: ' + e.message;
    }}
}}

init();
</script>
</body>
</html>"""


@app.get(
    "/health",
    summary="健康检查",
    description="检查服务是否正常运行，不消耗任何 API 额度。可用于监控系统可用性。",
    tags=["系统"],
    response_model=HealthResponse,
)
async def health_check():
    # 检测数据库连接池状态
    from src.modules.db_manager import _is_pool_available, _pool_error
    db_ok = False
    db_err = None
    try:
        db_ok = _is_pool_available()
    except Exception as e:
        db_err = str(e)
    if not db_ok:
        db_err = db_err or _pool_error or "数据库连接池未初始化"
    return {
        "status": "ok",
        "service": "作业批改系统",
        "version": "2.1.0",
        "db_connected": db_ok,
        "db_error": db_err if not db_ok else None,
    }


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
        page_image_urls = []    # 逐页图片访问 URL（用于前端逐页编辑）
        page_relative_questions = []  # 逐页题目区域（页面内相对坐标，未修正 Y 偏移）

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

            # 3a2. 保存逐页编辑用图到静态目录（前端逐页切换时显示）
            edit_page_name = f"{upload_id}_edit_page{page_idx+1}.jpg"
            edit_page_path = static_img_dir / edit_page_name
            cv2.imwrite(str(edit_page_path), img)
            page_image_urls.append(f"/static/uploads/{edit_page_name}")

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
            page_qs = []
            for q in questions:
                q_data = {
                    "question_no": 0,  # 稍后重新编号
                    "page": page_idx + 1,
                    "x": q.get("x", 0),
                    "y": q.get("y", 0),
                    "width": q.get("width", 0),
                    "height": q.get("height", 0),
                    "text": q.get("text", ""),
                    "y_offset": 0,  # 原始 Y（相对当前页），稍后修正
                }
                all_questions.append(q_data)
                page_qs.append(dict(q_data))  # 深拷贝保存逐页相对坐标
            page_relative_questions.append(page_qs)

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
            page_relative_questions = []  # 重建逐页数据
            for page_idx, img in enumerate(page_original_imgs):
                h, w = img.shape[:2]
                q_data = {
                    "question_no": page_idx + 1,
                    "page": page_idx + 1,
                    "x": 0,
                    "y": y_cumulative,
                    "width": w,
                    "height": h,
                    "text": f"第{page_idx+1}页 整页内容",
                    "y_offset": y_cumulative,
                }
                all_questions.append(q_data)
                page_relative_questions.append([{
                    "question_no": page_idx + 1,
                    "page": page_idx + 1,
                    "x": 0, "y": 0,
                    "width": w, "height": h,
                    "text": f"第{page_idx+1}页 整页内容",
                }])
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

        # 存储上传信息（grade-selected 使用拼接图 + 修正后的坐标，以及逐页数据）
        upload_store[upload_id] = {
            "image_path": str(stitched_path),
            "questions": all_questions,
            "original_pages": page_count,
            "page_images": page_image_urls,
            "page_questions": page_relative_questions,
            "page_heights": page_heights,
        }

        # 构建逐页题目模型（页面内相对坐标）
        page_question_models = []
        for page_idx, pqs in enumerate(page_relative_questions):
            page_question_models.append([
                QuestionRegion(
                    question_no=q["question_no"],
                    x=q["x"],
                    y=q["y"],
                    width=q["width"],
                    height=q["height"],
                    text=q.get("text", ""),
                )
                for q in pqs
            ])

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
            "page_images": page_image_urls,
            "page_questions": page_question_models,
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


class EnhanceImageRequest(BaseModel):
    upload_id: str = Field(..., min_length=1, description="上传任务标识")
    page: int = Field(default=1, ge=1, description="页码，从1开始")


class EnhanceImageResponse(BaseModel):
    upload_id: str
    page: int
    original_url: str
    enhanced_base64: str  # data:image/jpeg;base64,...
    enhanced_width: int
    enhanced_height: int
    preprocessing: dict


@app.post(
    "/enhance-image",
    summary="图片修复增强预览",
    description="对已上传的作业图片运行预处理流水线（纠偏/去噪/CLAHE/锐化），返回增强后的 base64 图片供前端预览对比。",
    tags=["交互式批改"],
    response_model=EnhanceImageResponse,
    responses={
        200: {"description": "增强成功"},
        404: {"description": "upload_id 不存在", "model": ErrorResponse},
        500: {"description": "处理失败", "model": FailedResponse},
    },
)
async def enhance_image(body: EnhanceImageRequest):
    """对已上传图片运行预处理增强，返回原图和增强图的对比数据"""
    from src.modules.image_processor import ImageProcessor

    upload_id = body.upload_id
    if upload_id not in upload_store:
        return JSONResponse(status_code=404, content={"error": f"上传任务 {upload_id} 不存在或已过期"})

    upload_info = upload_store[upload_id]
    page_images = upload_info.get("page_images", [])
    page = body.page

    if page < 1 or page > len(page_images):
        return JSONResponse(status_code=400, content={"error": f"页码 {page} 超出范围 (1-{len(page_images)})"})

    # 获取原图 URL，转换为本地文件路径
    original_url = page_images[page - 1]
    # URL 格式: /static/uploads/xxx.jpg → 去掉 /static/ 前缀匹配 static_dir
    url_path = original_url.split("?")[0]
    if url_path.startswith("/static/"):
        url_path = url_path[len("/static/"):]
    local_path = static_dir / url_path

    if not local_path.exists():
        return JSONResponse(status_code=404, content={"error": f"图片文件不存在: {local_path}"})

    try:
        import time
        t0 = time.time()

        # 读取原图
        img_array = np.fromfile(str(local_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(status_code=500, content={"error": "无法解码图片"})

        h, w = img.shape[:2]

        # 运行预处理增强管线：光照归一化 + 双边滤波 + CLAHE + USM锐化
        img_config = _app_config.get("image_processing", {})
        processor = ImageProcessor(config=img_config)

        enhanced = processor.enhance_for_vlm(img)

        # 编码为 base64 JPEG
        _, buf = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 90])
        enhanced_b64 = base64.b64encode(buf).decode("utf-8")
        enhanced_data_url = f"data:image/jpeg;base64,{enhanced_b64}"

        elapsed = (time.time() - t0) * 1000
        logger.info(f"图片增强完成: page={page}, size={w}x{h}, {elapsed:.0f}ms")

        # 实际使用的预处理步骤
        preprocess_steps = {
            "illumination_norm": True,
            "bilateral_denoise": True,
            "clahe": True,
            "usm_sharpen": True,
        }

        return EnhanceImageResponse(
            upload_id=upload_id,
            page=page,
            original_url=original_url,
            enhanced_base64=enhanced_data_url,
            enhanced_width=w,
            enhanced_height=h,
            preprocessing=preprocess_steps,
        )
    except Exception as e:
        logger.exception("图片增强失败")
        return JSONResponse(status_code=500, content={"error": f"图片增强失败: {str(e)}"})


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

        grading_mode = body.grading_mode or "vision"

        # ===== 图片哈希缓存：相同图片直接用历史结果（force=true 时跳过缓存）=====
        image_hash = compute_image_hash(image_path)
        cached = None
        if not getattr(body, "force", False):
            cached = db.get_by_image_hash(image_hash, grading_mode)
        if cached:
            logger.info(f"缓存命中: image_hash={image_hash[:12]}..., mode={grading_mode}")
            sub_qs = cached.get("sub_questions")
            if isinstance(sub_qs, str):
                try: sub_qs = json.loads(sub_qs)
                except: sub_qs = []
            questions_results = [{
                "question_no": cached["question_no"],
                "is_correct": cached.get("is_correct", False),
                "score": cached.get("score", 0),
                "max_score": cached.get("max_score", 0),
                "comment": cached.get("comment", ""),
                "ocr_text": cached.get("question_text", ""),
                "subject": cached.get("subject", "未分类"),
                "topic": cached.get("topic", "未分类"),
                "difficulty": cached.get("difficulty", "中等"),
                "student_answer_found": True,
                "sub_questions": sub_qs or [],
                "image_url": cached.get("crop_image", ""),
            }]
            # 对缓存结果也做一致性修正（旧数据可能矛盾）
            cached_result = {
                "is_correct": cached.get("is_correct", False),
                "score": cached.get("score", 0),
                "max_score": cached.get("max_score", 0),
                "comment": cached.get("comment", ""),
                "subject": cached.get("subject", "未分类"),
                "topic": cached.get("topic", "未分类"),
                "difficulty": cached.get("difficulty", "中等"),
                "student_answer_found": True,
                "sub_questions": sub_qs or [],
            }
            from src.modules.llm_grader import LLMGrader
            cached_result = LLMGrader._enforce_consistency(cached_result)
            fixed_subs = cached_result.get("sub_questions", [])
            fixed_score = cached_result.get("score", 0)
            fixed_max = cached_result.get("max_score", 0)
            fixed_correct = cached_result.get("is_correct", False)
            fixed_comment = cached_result.get("comment", "")

            questions_results = [{
                "question_no": cached["question_no"],
                "is_correct": fixed_correct,
                "score": fixed_score,
                "max_score": fixed_max,
                "comment": fixed_comment,
                "ocr_text": cached.get("question_text", ""),
                "subject": cached.get("subject", "未分类"),
                "topic": cached.get("topic", "未分类"),
                "difficulty": cached.get("difficulty", "中等"),
                "student_answer_found": True,
                "sub_questions": fixed_subs,
                "image_url": cached.get("crop_image", ""),
            }]
            # 修正后的数据写回 DB，保证下次缓存一致性
            try:
                db.update_grading_result(cached["id"], {
                    "is_correct": fixed_correct,
                    "score": fixed_score,
                    "max_score": fixed_max,
                    "comment": fixed_comment,
                    "sub_questions": fixed_subs,
                })
            except Exception:
                pass
            # 直接用 DB 里的批注图
            annotated_path = os.path.join("data/output", os.path.basename(cached.get("annotated_image", "") or ""))
            total_score = fixed_score
            total_max = fixed_max
            all_subs = fixed_subs
            correct_count = sum(1 for s in all_subs if s.get("is_correct")) if all_subs else (1 if fixed_correct else 0)
            total_question_count = len(all_subs) if all_subs else 1
            return {
                "task_id": cached.get("task_id", task_id),
                "message": "批改完成（使用历史结果）",
                "total_score": total_score,
                "total_max": total_max,
                "correct_count": correct_count,
                "questions_count": total_question_count,
                "questions": questions_results,
                "annotated_image_url": cached.get("annotated_image", ""),
                "cached": True,
            }

        questions_results = []
        corrections = []

        if grading_mode == "ocr_llm":
            # ===== OCR+LLM 模式：逐题裁剪 → OCR提取文字 → 文本模型批改 =====
            ocr = get_api_ocr()

            # 预加载逐页图片（用于按页裁剪题目），加载后自动增强
            page_images = upload_info.get("page_images", [])
            page_img_cache = {}
            from src.modules.image_processor import ImageProcessor
            img_proc = ImageProcessor(config=_app_config.get("image_processing", {}))
            for pi, p_url in enumerate(page_images):
                # 从 URL 反推本地路径
                p_name = Path(p_url).name
                p_path = static_dir / "uploads" / p_name
                if p_path.exists():
                    img = cv2.imread(str(p_path))
                    if img is None:
                        img_array = np.fromfile(str(p_path), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        # OCR+LLM 模式也做图片增强，提升 OCR 识别准确率
                        img = img_proc.enhance_for_vlm(img)
                        page_img_cache[pi + 1] = img  # key 为页码

            for q in questions:
                # 使用题目所属页码的图片进行裁剪
                q_page = getattr(q, "page", 1) or 1
                page_img = page_img_cache.get(q_page, image)
                ph, pw = page_img.shape[:2]

                x1 = max(0, min(q.x, pw - 1))
                y1 = max(0, min(q.y, ph - 1))
                x2 = min(pw, max(x1 + 1, q.x + q.width))
                y2 = min(ph, max(y1 + 1, q.y + q.height))
                cropped = page_img[y1:y2, x1:x2]
                crop_path = os.path.join(tempfile.gettempdir(), f"{upload_id}_q{q.question_no}.jpg")
                cv2.imwrite(crop_path, cropped)

                # OCR 提取文字
                ocr_text = ""
                try:
                    ocr_result = ocr.recognize_text(crop_path)
                    ocr_text = ocr.parse_result(ocr_result)
                except Exception as ocr_err:
                    logger.warning(f"第{q.question_no}题 OCR 失败: {ocr_err}")

                # 文本模型批改
                try:
                    grade_result = grader.grade(
                        ocr_text=ocr_text,
                        reference_answer="",
                    )
                except Exception as grade_err:
                    logger.exception(f"第{q.question_no}题批改失败: {grade_err}")
                    grade_result = {
                        "is_correct": False, "score": 0, "max_score": 0,
                        "comment": f"批改出错: {str(grade_err)[:100]}",
                        "subject": "未分类", "topic": "未分类", "difficulty": "中等",
                        "student_answer_found": False, "sub_questions": [],
                    }

                # 保存裁剪图用于错题本
                saved_crop_name = f"{upload_id}_q{q.question_no}.jpg"
                saved_crop_path = Path("data/output") / saved_crop_name
                Path("data/output").mkdir(parents=True, exist_ok=True)
                shutil.copy2(crop_path, str(saved_crop_path))

                sub_questions = grade_result.get("sub_questions", [])
                questions_results.append({
                    "question_no": q.question_no,
                    "is_correct": grade_result.get("is_correct", False),
                    "score": grade_result.get("score", 0),
                    "max_score": grade_result.get("max_score", 0),
                    "comment": grade_result.get("comment", ""),
                    "ocr_text": ocr_text or "",
                    "subject": grade_result.get("subject", "未分类"),
                    "topic": grade_result.get("topic", "未分类"),
                    "difficulty": grade_result.get("difficulty", "中等"),
                    "student_answer_found": grade_result.get("student_answer_found", True),
                    "sub_questions": sub_questions,
                    "image_url": f"/image/{saved_crop_name}",
                })

                corrections.append({
                    "position": (q.x + q.width // 2, q.y + q.height // 2),
                    "is_correct": grade_result.get("is_correct", False),
                    "comment": f"第{q.question_no}题: {grade_result.get('comment','')}",
                    "score": grade_result.get("score", 0),
                    "max_score": grade_result.get("max_score", 0),
                    "page": q_page,
                })

                # 清理临时裁剪文件
                try:
                    if os.path.exists(crop_path):
                        os.remove(crop_path)
                except Exception:
                    pass

            # OCR 模式：将逐页坐标转为拼接图坐标后渲染批注
            page_heights = upload_info.get("page_heights", [])
            # 计算每页在拼接图中的 Y 偏移
            y_offsets = []
            y_cum = 0
            sep_h = 20
            for i, ph in enumerate(page_heights):
                y_offsets.append(y_cum)
                y_cum += ph
                if i < len(page_heights) - 1:
                    y_cum += sep_h

            img_h, img_w = image.shape[:2]
            region_dicts = []
            for q in questions:
                q_page = getattr(q, "page", 1) or 1
                y_off = y_offsets[q_page - 1] if q_page <= len(y_offsets) else 0
                region_dicts.append({
                    "x": q.x, "y": q.y + y_off,
                    "width": q.width, "height": q.height,
                })
                # 同步修正 correction 位置
                for c in corrections:
                    if c.get("page") == q_page and c["position"] == (q.x + q.width // 2, q.y + q.height // 2):
                        c["position"] = (q.x + q.width // 2, q.y + q.height // 2 + y_off)
                        break
            if not region_dicts:
                region_dicts.append({"x": 10, "y": 10, "width": img_w - 20, "height": img_h - 20})
            annotated = renderer.render_all_with_regions(image, corrections, region_dicts)
            output_dir = Path("data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            annotated_path = str(output_dir / f"{upload_id}_annotated.jpg")
            cv2.imwrite(annotated_path, annotated)

        elif grading_mode == "vision_deepseek":
            # ===== Vision+DeepSeek 模式：Vision看图提取文字 → DeepSeek深度推理判题 =====
            full_page_path = os.path.join(tempfile.gettempdir(), f"{upload_id}_fullpage.jpg")
            cv2.imwrite(full_page_path, image)

            try:
                grade_result = grader.grade_vision_deepseek(
                    image_path=full_page_path,
                    reference_answer="",
                )
                q_text = grade_result.get("ocr_text", "")
            except Exception as grade_err:
                logger.exception(f"Vision+DeepSeek 批改失败: {grade_err}")
                q_text = "[批改失败]"
                grade_result = {
                    "is_correct": False, "score": 0, "max_score": 0,
                    "comment": f"批改出错: {str(grade_err)[:100]}",
                    "subject": "未分类", "topic": "未分类", "difficulty": "中等",
                    "student_answer_found": False, "sub_questions": [],
                    "ocr_text": "",
                }

            # 保留整图用于错题本
            saved_image_name = f"{upload_id}_original.jpg"
            saved_image_path = Path("data/output") / saved_image_name
            Path("data/output").mkdir(parents=True, exist_ok=True)
            shutil.copy2(full_page_path, str(saved_image_path))

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

            # 构建标记区域并渲染批注图
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

            # 清理临时文件
            try:
                if os.path.exists(full_page_path):
                    os.remove(full_page_path)
            except Exception:
                pass

        else:
            # ===== Vision 模式（默认）：整页发给视觉模型一次批改所有题目 =====
            full_page_path = os.path.join(tempfile.gettempdir(), f"{upload_id}_fullpage.jpg")
            cv2.imwrite(full_page_path, image)

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

        # —— 持久化到 MySQL ——
        _save_grading_results(
            upload_id=upload_id,
            task_id=task_id,
            questions_results=questions_results,
            annotated_path=annotated_path,
            questions=questions,
            grading_mode=grading_mode,
            upload_info=upload_info,
            image_hash=image_hash,
        )

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
    # 从数据库查询批改结果
    records = db.get_by_task(task_id)
    if records:
        # 重建响应格式（兼容旧接口）
        questions = []
        for r in records:
            import json as _json
            sub_qs = r.get("sub_questions")
            if isinstance(sub_qs, str):
                sub_qs = _json.loads(sub_qs) if sub_qs else []
            elif sub_qs is None:
                sub_qs = []
            questions.append({
                "question_no": r["question_no"],
                "is_correct": bool(r.get("is_correct")),
                "score": float(r.get("score", 0)),
                "max_score": float(r.get("max_score", 0)),
                "comment": r.get("comment", ""),
                "ocr_text": r.get("question_text", ""),
                "subject": r.get("subject", ""),
                "topic": r.get("topic", ""),
                "difficulty": r.get("difficulty", ""),
                "student_answer_found": True,
                "sub_questions": sub_qs,
                "image_url": r.get("crop_image", ""),
            })
        all_subs = [s for q in questions for s in q.get("sub_questions", [])]
        correct_count = sum(1 for s in all_subs if s.get("is_correct")) if all_subs else sum(1 for q in questions if q["is_correct"])
        total_count = len(all_subs) if all_subs else len(questions)

        result = {
            "task_id": task_id,
            "message": "批改完成",
            "total_score": sum(q["score"] for q in questions),
            "total_max": sum(q["max_score"] for q in questions),
            "correct_count": correct_count,
            "questions_count": total_count,
            "questions": questions,
            "annotated_image_url": records[0].get("annotated_image", "") if records else "",
        }
        return {"task_id": task_id, "result": result}

    # 回退到内存 store（兼容刚批改完还未入库的瞬时情况）
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


# ========== 重做批改 API ==========

@app.post(
    "/api/redo-grading",
    summary="重做批改",
    description="""
对批改结果中判错的题目重新批改。统一使用 Qwen-VL 视觉模型看图重判（当前准确率最高的方案）。
将上次的批改结果发回模型，告知「你做错了请重新检查」，支持老师附加纠错提示。

**请求参数**：
- `task_id`：原批改任务 ID
- `question_no`：要重做的题号
- `sub_label`（可选）：Vision 模式下指定重做哪个小题
- `hint`（可选）：老师提示，帮助模型纠错
""",
    tags=["批改"],
    responses={
        200: {"description": "重做成功，返回新结果"},
        404: {"description": "记录不存在", "model": ErrorResponse},
        400: {"description": "原图文件不存在", "model": ErrorResponse},
        500: {"description": "重做失败", "model": FailedResponse},
    },
)
async def redo_grading(body: RedoGradingRequest):
    # 1. 从 DB 查原记录
    records = db.get_by_task(body.task_id)
    target = None
    for r in records:
        if r["question_no"] == body.question_no:
            target = r
            break
    if not target:
        return JSONResponse(status_code=404, content={"error": f"任务 {body.task_id} 中未找到第 {body.question_no} 题"})

    # 2. 重做统一用全页图（upload_store 高清原图 > DB 记录）
    image_path = ""
    image_source = "unknown"
    if body.upload_id and body.upload_id in upload_store:
        image_path = upload_store[body.upload_id].get("image_path", "")
        image_source = "upload_store"
    if not image_path or not os.path.exists(image_path):
        # 回退到 DB 记录中的图片
        fallback_field = target.get("original_image", "") or target.get("crop_image", "")
        image_source = "db_original_image" if target.get("original_image") else "db_crop_image"
        image_path = fallback_field
        if image_path.startswith("/image/"):
            image_path = os.path.join("data/output", os.path.basename(image_path))
        elif image_path.startswith("/static/"):
            image_path = str(static_dir / (image_path[len("/static/"):]))
        elif image_path.startswith("/error_images/"):
            image_path = str(ERROR_NB_IMAGE_DIR / os.path.basename(image_path))
    if not image_path or not os.path.exists(image_path):
        return JSONResponse(status_code=400, content={"error": f"原图文件不存在: {image_path}"})
    logger.info(f"重做图片来源: {image_source}, 路径: {image_path}, 文件大小: {os.path.getsize(image_path) if os.path.exists(image_path) else 'N/A'} bytes")

    # 3. 构建 previous_result
    sub_qs = target.get("sub_questions")
    if isinstance(sub_qs, str):
        try:
            sub_qs = json.loads(sub_qs)
        except Exception:
            sub_qs = []
    previous = {
        "is_correct": bool(target.get("is_correct")),
        "score": float(target.get("score", 0)),
        "max_score": float(target.get("max_score", 0)),
        "comment": target.get("comment", ""),
        "sub_questions": sub_qs or [],
        "ocr_text": target.get("question_text", ""),
        "student_answer": target.get("student_answer", ""),
    }

    # 如果是重做特定小题，从 sub_questions 中提取
    sub_previous = None
    if body.sub_label and previous["sub_questions"]:
        for sq in previous["sub_questions"]:
            if sq.get("label") == body.sub_label:
                sub_previous = sq
                break

    # 4. 重做：统一用 Qwen-VL 视觉模型看图重判（当前准确率最高的方案）
    #    传入上次结果 + 老师提示 + 小题过滤，让模型知道「上次判错了请重新检查」
    grader = get_api_grader()
    try:
        new_result = grader.redo_question(
            image_path=image_path,
            previous_result=previous,
            hint=body.hint or "",
            sub_label=body.sub_label,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"重做失败: {str(e)}"})

    # 5. 对比新旧结果（精确到小题级别）
    #    优先比较 sub_label 指定的小题，其次匹配 question_no 对应的小题，
    #    都找不到时才用整页总分比较（保守但可能误判 magnitude）
    old_score = float(target.get("score", 0))
    new_score = float(new_result.get("score", 0))
    changed = abs(new_score - old_score) > 0.001  # 默认：整题比较

    if new_result.get("sub_questions"):
        # 尝试精确到小题级别比较
        lookup_label = body.sub_label if (sub_previous and body.sub_label) else str(body.question_no)
        for ns in new_result["sub_questions"]:
            if str(ns.get("label", "")) == lookup_label:
                matched_new = float(ns.get("score", 0))
                matched_old = float(sub_previous.get("score", 0)) if (sub_previous and body.sub_label) else old_score
                changed = abs(matched_new - matched_old) > 0.001
                break

    # 6. 合并小题并更新 DB
    #    _redo_vision 返回整页所有小题（与 grade_full_page 同格式），
    #    按 label 匹配合并。如果新旧 label 层级不同（OCR 记录的子题 label 是
    #    "(1)/(2)" 而 Vision 返回的是 "46/47"），直接用新结果替换。
    new_subs = new_result.get("sub_questions", [])
    old_subs = list(previous["sub_questions"])

    if new_subs and old_subs:
        # 检测新旧 label 是否处于同一层级（同为页面题号，或同为子题编号）
        old_labels = {sq.get("label", "") for sq in old_subs}
        new_labels = {ns.get("label", "") for ns in new_subs}
        same_level = bool(old_labels & new_labels)  # 有交集说明同层级

        if same_level:
            # 同层级：按 label 匹配合并
            label_map = {sq.get("label"): sq for sq in old_subs}
            for ns in new_subs:
                lbl = ns.get("label", "")
                if lbl in label_map:
                    label_map[lbl]["is_correct"] = ns.get("is_correct", False)
                    label_map[lbl]["score"] = ns.get("score", 0)
                    label_map[lbl]["max_score"] = ns.get("max_score", 0)
                    label_map[lbl]["comment"] = ns.get("comment", "")
                else:
                    label_map[lbl] = ns
                    logger.info(f"重做发现新小题: label={lbl}")
            merged_subs = list(label_map.values())
        else:
            # 不同层级（OCR→Vision 统一）：从整页结果中只提取匹配目标 question_no 的小题
            target_label = str(body.question_no)
            matched_sq = None
            for ns in new_subs:
                if str(ns.get("label", "")) == target_label:
                    matched_sq = ns
                    break
            if matched_sq:
                # 匹配到的子题如有嵌套 sub_questions 则展开，否则包装为单元素列表
                nested = matched_sq.get("sub_questions", [])
                if nested:
                    merged_subs = nested
                else:
                    merged_subs = [matched_sq]
                logger.info(
                    f"重做切换为 Vision 结果: question_no={body.question_no}, "
                    f"旧 labels={old_labels} → 提取到 {len(merged_subs)} 小题"
                )
            else:
                # 模型返回中未找到匹配 label，安全回退：使用全部新小题
                logger.warning(
                    f"重做未在 Vision 结果中找到 question_no={body.question_no}, "
                    f"可用 labels={new_labels}, 回退使用全部 {len(new_subs)} 小题"
                )
                merged_subs = new_subs
    elif new_subs:
        merged_subs = new_subs
    else:
        merged_subs = []

    all_correct = all(sq.get("is_correct", False) for sq in merged_subs) if merged_subs else new_result.get("is_correct", False)
    total_score = sum(float(sq.get("score", 0)) for sq in merged_subs) if merged_subs else float(new_result.get("score", 0))
    total_max = sum(float(sq.get("max_score", 0)) for sq in merged_subs) if merged_subs else float(new_result.get("max_score", 0))

    db.update_grading_result(target["id"], {
        "sub_questions": merged_subs,
        "is_correct": all_correct,
        "score": total_score,
        "max_score": total_max,
        "comment": new_result.get("comment", target.get("comment", "")),
        "student_answer": new_result.get("student_answer", ""),
    })
    logger.info(
        f"重做 DB 更新完成: id={target['id']}, "
        f"old_comment={target.get('comment','')[:50]} → new_comment={new_result.get('comment','')[:50]}, "
        f"old_subs={len(old_subs)} → new_subs={len(merged_subs)}, "
        f"changed={changed}"
    )

    # 7. 重新生成批注图（使用合并后的小题结果）
    try:
        renderer = get_api_renderer()
        img_array = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is not None:
            img_h, img_w = image.shape[:2]
            subs_for_annotation = merged_subs
            corrections = []
            if subs_for_annotation:
                for i, sq in enumerate(subs_for_annotation):
                    y_pos = int(img_h * (i + 1) / (len(subs_for_annotation) + 1))
                    corrections.append({
                        "position": (img_w // 2, y_pos),
                        "is_correct": sq.get("is_correct", False),
                        "comment": f"{sq.get('label','')}: {sq.get('comment','')}",
                        "score": sq.get("score", 0),
                        "max_score": sq.get("max_score", 0),
                    })
            if not corrections:
                corrections.append({
                    "position": (img_w // 2, img_h // 2),
                    "is_correct": new_result.get("is_correct", False),
                    "comment": new_result.get("comment", ""),
                    "score": new_result.get("score", 0),
                    "max_score": new_result.get("max_score", 0),
                })
            region_dicts = []
            for i in range(len(corrections)):
                y_pos = int(img_h * (i + 1) / (len(corrections) + 1))
                region_dicts.append({"x": 10, "y": y_pos - 15, "width": img_w - 20, "height": 30})
            annotated = renderer.render_all_with_regions(image, corrections, region_dicts)
            new_annotated_name = f"{body.task_id}_redo_annotated.jpg"
            new_annotated_path = str(Path("data/output") / new_annotated_name)
            cv2.imwrite(new_annotated_path, annotated)
            db.update_grading_result(target["id"], {"annotated_image": f"/image/{new_annotated_name}"})
            logger.info(f"重做批注图已更新: {new_annotated_name}")
    except Exception as e:
        logger.warning(f"重做批注图生成失败: {e}")

    # 8. 返回
    updated = db.get_by_id(target["id"])
    # 序列化 record 中的 datetime/Decimal 等类型
    if updated:
        for key in ("created_at", "updated_at", "notebook_saved_at", "reviewed_at"):
            if key in updated and updated[key] is not None:
                updated[key] = str(updated[key])
        if "score" in updated and updated["score"] is not None:
            updated["score"] = float(updated["score"])
        if "max_score" in updated and updated["max_score"] is not None:
            updated["max_score"] = float(updated["max_score"])
        updated["is_correct"] = bool(updated.get("is_correct", 0))
        updated["in_notebook"] = bool(updated.get("in_notebook", 0))
        updated["is_reviewed"] = bool(updated.get("is_reviewed", 0))
        # 反序列化 sub_questions
        sq = updated.get("sub_questions")
        if isinstance(sq, str):
            try:
                updated["sub_questions"] = json.loads(sq)
            except Exception:
                updated["sub_questions"] = []

    return {
        "success": True,
        "changed": changed,
        "question_no": body.question_no,
        "sub_label": body.sub_label,
        "record": updated,
    }


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
    difficulty: Optional[str] = Query(None, description="按难度筛选"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    result = error_notebook.get_all(
        subject=subject,
        reviewed=reviewed,
        keyword=keyword,
        difficulty=difficulty,
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


class UpdateTopicRequest(BaseModel):
    topic: str = Field(description="新的知识点")


@app.patch(
    "/api/error-notebook/{entry_id}/topic",
    summary="编辑知识点",
    description="修改错题的知识点分类。",
    tags=["错题本"],
    response_model=MessageResponse,
)
async def update_topic(
    entry_id: str = FastAPIPath(..., description="错题ID"),
    body: UpdateTopicRequest = None,
):
    new_topic = body.topic if body else ""
    ok = error_notebook.update_topic(entry_id, new_topic)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "错题不存在"})
    return {"message": "知识点已更新"}


class UpdateSubjectRequest(BaseModel):
    subject: str = Field(description="新的学科")


@app.patch(
    "/api/error-notebook/{entry_id}/subject",
    summary="编辑学科",
    description="修改错题的学科分类。",
    tags=["错题本"],
    response_model=MessageResponse,
)
async def update_subject(
    entry_id: str = FastAPIPath(..., description="错题ID"),
    body: UpdateSubjectRequest = None,
):
    new_subject = body.subject if body else ""
    ok = error_notebook.update_subject(entry_id, new_subject)
    if not ok:
        return JSONResponse(status_code=404, content={"error": "错题不存在"})
    return {"message": "学科已更新"}


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
    # 使用 llm_grader 中的统一学科检测函数（避免关键词库重复）
    from src.modules.llm_grader import auto_detect_subject

    saved = 0
    skipped_dup = 0

    # —— 从数据库查找该 task 的已有记录 ——
    existing_records = db.get_by_task(body.task_id)
    record_map = {r["question_no"]: r for r in existing_records}

    for q in body.questions:
        if not q.get("save"):
            continue
        q_no = q.get("question_no", 0)
        q_text = q.get("ocr_text", q.get("comment", ""))
        subject = auto_detect_subject(q_text, q.get("subject", "未分类"))

        # 去重检查
        if error_notebook.is_duplicate(q_no, subject, q_text):
            skipped_dup += 1
            logger.info(f"跳过重复错题: 题号{q_no} 学科{subject}")
            continue

        # 查找已入库的记录
        existing = record_map.get(q_no)
        if existing:
            # 直接标记为错题本
            db.mark_as_notebook(existing["id"])
            # 可选：更新学科（如果自动检测的更准）
            if subject and subject != existing.get("subject"):
                db.update_subject(existing["id"], subject)
            saved += 1
        else:
            # 记录尚未入库（异常情况），走旧逻辑新增
            subs = q.get("sub_questions", [])
            subs_text = ""
            for s in subs:
                status = "✓" if s.get("is_correct") else "✗"
                subs_text += f"[{status}] {s.get('label','')} ({s.get('score',0)}/{s.get('max_score',0)}分): {s.get('comment','')}\n"

            error_notebook.add_entry({
                "upload_id": body.task_id,
                "task_id": body.task_id,
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
