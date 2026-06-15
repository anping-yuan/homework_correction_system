"""
大模型批改模块
负责调用大语言模型对作业内容进行智能批改和评分。
v3: 支持从 OCR 文本中自动区分题目和答案、拆分多小题独立评分。
v4: 图片压缩 + API 重试机制 + 学科检测统一提取。
v5: 送 VL 模型前自动预处理增强画质（CLAHE/去噪/纠偏/锐化）。
"""
import json
import os
import re
import time
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from openai import OpenAI

from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)

# ===================== 学科自动检测（统一关键词库）=====================

# 各学科关键词（供 prompt 和后端回退算法共用）
SUBJECT_KEYWORDS = {
    "数学": ["方程", "函数", "几何", "计算", "面积", "体积", "复数", "向量",
             "概率", "数列", "导数", "积分", "三角", "坐标", "平方", "根号", "不等式",
             "正弦", "余弦", "正切", "圆", "直径", "半径", "周长", "角度", "平行",
             "垂直", "对称", "相似", "全等", "图像", "象限", "模", "|", "△"],
    "物理": ["力", "速度", "加速度", "质量", "电场", "磁场", "电流", "电压",
             "电阻", "功率", "能量", "动量", "光学", "声", "热", "牛顿", "欧姆", "焦耳",
             "压强", "浮力", "密度", "波长", "频率", "电路", "串联", "并联"],
    "化学": ["化学", "反应", "元素", "分子", "原子", "离子", "溶液", "酸", "碱",
             "盐", "氧化", "还原", "电解", "沉淀", "化合价", "摩尔", "pH", "h₂o", "co₂",
             "naoh", "hcl", "fe", "cu", "zn", "o₂", "h₂", "↓", "↑"],
    "生物": ["细胞", "基因", "遗传", "蛋白", "酶", "光合", "呼吸", "神经",
             "激素", "免疫", "生态", "物种", "进化", "dna", "rna", "染色体", "器官"],
    "语文": ["古文", "诗词", "文言", "作者", "修辞", "描写", "表达", "主旨",
             "意境", "情感", "诗人", "句子", "成语", "拼音", "笔画", "部首"],
}

# 数学强特征字符
MATH_CHARS = set("0123456789+-×÷=<>√^∑∫∏")


def auto_detect_subject(question_text: str, model_subject: str = "") -> str:
    """自动检测学科：先信任模型分类，如果明显矛盾则按内容修正。

    供 llm_grader 和 server.py 共享使用，避免关键词库重复。
    """
    text = (question_text or "").lower()

    def detect_by_content(t: str) -> str:
        scores = {}
        for subj, keywords in SUBJECT_KEYWORDS.items():
            scores[subj] = sum(1 for w in keywords if w in t)
        # 英语：大量 ASCII 字母
        eng_chars = sum(1 for c in t if c.isascii() and c.isalpha())
        scores["英语"] = 1 if eng_chars > 50 else 0

        best = max(scores, key=scores.get)
        return best if scores[best] >= 1 else ""

    model_subj = model_subject if model_subject and model_subject != "未分类" else ""
    content_subj = detect_by_content(text)

    # 模型与内容矛盾时以内容为准
    if model_subj and content_subj and model_subj != content_subj:
        # 数学 vs 语文 常见误判修正
        math_indicators = sum(1 for c in text if c in MATH_CHARS) + \
            sum(1 for w in ["方程", "函数", "几何", "计算", "复数", "向量", "概率",
                            "数列", "导数", "积分", "三角", "坐标", "不等式"] if w in text)
        if model_subj == "语文" and math_indicators >= 3:
            return "数学"
        if model_subj == "数学" and sum(1 for c in text if c in "0123456789") == 0:
            if content_subj != "数学":
                return content_subj

    return model_subj or content_subj or "其他"


class LLMGrader:
    """大模型批改器"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base_url: Optional[str] = None,
    ):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        if config_path is None:
            config_path = project_root / "config" / "config.json"
        else:
            config_path = Path(config_path)
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        llm_config = config.get("llm", {})
        self.api_key = (
            api_key
            or llm_config.get("api_key", "")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.model = (
            model
            or llm_config.get("model", "deepseek-v4-pro")
            or os.environ.get("OPENAI_MODEL")
        )
        self.base_url = (
            api_base_url
            or llm_config.get("api_base_url", "https://api.deepseek.com")
            or os.environ.get("OPENAI_API_BASE_URL")
        )
        if not self.api_key:
            raise ValueError("没有找到OpenAI API密钥,请配置.")
        self.grading_prompt = ""

        # 初始化视觉模型客户端（通义千问 VL）
        vl_config = config.get("dashscope", {})
        self.vl_api_key = vl_config.get("api_key", "") or os.environ.get("DASHSCOPE_API_KEY")
        self.vl_model = vl_config.get("model", "qwen-vl-max")
        self.vl_base_url = vl_config.get("api_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 超时配置（默认 120 秒，可通过 config 覆盖）
        vl_timeout = vl_config.get("request_timeout", 60)
        llm_timeout = llm_config.get("request_timeout", 60)

        self.vl_client = None
        if self.vl_api_key:
            self.vl_client = OpenAI(api_key=self.vl_api_key, base_url=self.vl_base_url, timeout=vl_timeout)
            logger.info(f"视觉模型初始化成功（model: {self.vl_model}, timeout={vl_timeout}s）")
        else:
            logger.warning("未配置 DashScope API Key，视觉批改不可用")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=llm_timeout)
        logger.info(f"大模型服务初始化成功（model: {self.model}, timeout={llm_timeout}s）")

        # 初始化图像预处理器（用于送 VL 模型前增强画质）
        img_config = config.get("image_processing", {})
        self.preprocess_enabled = img_config.get("preprocess_for_vl", True)
        self.image_processor = ImageProcessor(img_config)
        if self.preprocess_enabled:
            logger.info(f"图片预处理已启用: 光照归一化+双边滤波+CLAHE+USM锐化")

    def _prepare_image_for_api(self, image_path: str, max_size_kb: int = 3500,
                               preprocess: bool = True) -> str:
        """准备图片用于 API 调用：预处理增强画质 + 压缩大图后返回 base64 data URL。

        处理链路：读取 → 预处理(纠偏/去噪/CLAHE/锐化) → 压缩 → base64。
        通义千问 VL 图片大小限制约 5MB，此方法将超过 max_size_kb 的图片
        等比缩小直到满足大小要求。
        """
        ext = Path(image_path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        # 读取原图
        img_array = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            # 回退：直接读取原始字节
            with open(image_path, "rb") as f:
                raw = f.read()
            size_kb = len(raw) / 1024
            if size_kb > max_size_kb * 2:
                logger.warning(f"图片过大 ({size_kb:.0f}KB) 且无法解码，仍将发送原始文件")
            return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"

        # ===== 预处理增强画质（送 VL 模型前） =====
        if preprocess and self.preprocess_enabled:
            t0 = time.time()
            img = self.image_processor.enhance_for_vlm(img)
            elapsed = (time.time() - t0) * 1000
            logger.info(f"图片预处理完成: {image_path!r}, {img.shape[1]}x{img.shape[0]} ({elapsed:.0f}ms)")

        h, w = img.shape[:2]
        # 先检查原始编码大小
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 92] if mime == "image/jpeg" else []
        _, buf = cv2.imencode(ext, img, encode_param)
        size_kb = len(buf) / 1024

        if size_kb <= max_size_kb:
            b64 = base64.b64encode(buf).decode("utf-8")
            return f"data:{mime};base64,{b64}"

        # 需要压缩：逐步降低质量和尺寸
        logger.info(f"图片需压缩: {size_kb:.0f}KB → 目标 ≤{max_size_kb}KB")
        quality = 85
        for attempt in range(5):
            if w > 2048:
                ratio = 2048 / w
                w, h = int(w * ratio), int(h * ratio)
                img = cv2.resize(img, (w, h))
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buf = cv2.imencode(".jpg", img, encode_param)
            size_kb = len(buf) / 1024
            if size_kb <= max_size_kb:
                break
            quality = max(30, quality - 15)

        logger.info(f"图片压缩完成: {size_kb:.0f}KB (quality={quality})")
        b64 = base64.b64encode(buf).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _call_with_retry(fn, max_retries: int = 2):
        """API 调用重试：网络瞬时错误时自动重试，指数退避"""
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = (attempt + 1) * 2
                    logger.warning(f"API 调用失败 (attempt {attempt+1}/{max_retries+1}), "
                                   f"{wait}s 后重试: {e}")
                    time.sleep(wait)
                else:
                    raise last_err

    @staticmethod
    def _enforce_consistency(result: dict) -> dict:
        """修正模型输出矛盾：父题与子题统一、判错不拿满分"""
        sub_qs = result.get("sub_questions", [])
        if sub_qs:
            # 父题数据完全从子题重算，防止模型给矛盾的顶层值
            total_score = sum(float(sq.get("score", 0)) for sq in sub_qs)
            total_max = sum(float(sq.get("max_score", 0)) for sq in sub_qs)
            all_ok = all(sq.get("is_correct", False) for sq in sub_qs)
            result["score"] = total_score
            result["max_score"] = total_max
            result["is_correct"] = all_ok
            # 汇总评语
            parts = []
            correct_n = sum(1 for s in sub_qs if s.get("is_correct"))
            parts.append(f"共{len(sub_qs)}小题，{correct_n}✓/{len(sub_qs)-correct_n}✗")
            for sq in sub_qs:
                parts.append(f"[{sq.get('label','?')}] {sq.get('comment','')}")
            result["comment"] = "；".join(parts) if not result.get("comment") or result.get("comment").startswith("[系统") else result.get("comment", "")
        else:
            # 无子题：修正父题自身的矛盾
            is_correct = result.get("is_correct", False)
            score = result.get("score", 0)
            max_score = result.get("max_score", 1)
            if not is_correct and score >= max_score > 0:
                result["score"] = 0
                result["comment"] = "[系统修正：判错但给了满分，已归零] " + result.get("comment", "")
            elif is_correct and score <= 0 and max_score > 0:
                result["score"] = max_score
        # 小题内部一致性
        for sq in sub_qs:
            sq_ok = sq.get("is_correct", False)
            sq_score = sq.get("score", 0)
            sq_max = sq.get("max_score", 1)
            if not sq_ok and sq_score >= sq_max > 0:
                sq["score"] = 0
            elif sq_ok and sq_score <= 0 and sq_max > 0:
                sq["score"] = sq_max
        return result

    def load_prompt(self, prompt_template: str) -> None:
        """加载批改提示词模板"""
        self.grading_prompt = prompt_template

    def build_evaluation_context(self, ocr_text: str) -> str:
        """构建批改上下文。
        ocr_text 是 OCR 从作业图片中提取的原始文本，
        同时包含了题目和学生的答案，LLM 需要自行区分。
        """
        return "\n".join([
            "以下是从作业图片中通过 OCR 识别出的文本内容。",
            "这段文本混合了题目和学生的作答（可能为手写体 OCR 结果）。",
            "请仔细区分题目内容和学生答案，然后逐题批改。",
            "",
            "=== OCR 识别文本开始 ===",
            ocr_text.strip() or "[OCR 未识别到文字]",
            "=== OCR 识别文本结束 ===",
        ])

    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return "\n".join([
            "你是一位专业的作业批改老师。你会收到 OCR 从作业图片中提取的文本，",
            "这段文本混合了题目内容和学生的作答（可能为手写体 OCR 结果）。",
            "",
            "请按以下步骤完成批改：",
            "",
            "## 步骤1：区分题目和答案",
            "- 仔细阅读 OCR 文本，区分哪些是题目，哪些是学生写下的答案",
            "- 学生答案通常出现在题目下方、题号旁边或空白区域",
            "- 注意手写 OCR 可能不够准确，请根据上下文推断真实内容",
            "",
            "## 步骤2：拆分小题（重要！）",
            "- 如果一道题包含多个小题（如 1.(a) 1.(b) 或 ① ② ③），必须将每个小题单独评估",
            "- 每个小题独立判断对错、独立给分、独立写详细评语",
            "",
            "## 步骤3：逐题批改",
            "- 给分规则：简单题5分，中等题10分，困难题15分（含小题时总分在小题间平分）",
            "- 评语要求（每题必须有内容）：",
            "  · 正确：写\"✓ 正确\"，补充解题思路或表扬",
            "  · 错误：必须包含 (1)正确答案 (2)为什么错 (3)正确思路",
            "",
            "## 步骤4：分类",
            "- subject：数学/语文/英语/物理/化学/生物/历史/地理/政治/其他",
            "- topic：具体知识点，如\"现在完成时\"、\"一元二次方程\"",
            "- difficulty：简单/中等/困难",
            "",
            "## 输出格式（纯 JSON，不要用 ``` 包裹）：",
            "{",
            '  "is_correct": true/false,',
            '  "score": 总得分,',
            '  "max_score": 总满分,',
            '  "comment": "整体评语，必须要有内容",',
            '  "subject": "学科（必填！根据题目内容判断：出现数字/公式/方程→数学，出现古文/诗词/阅读理解(中文)→语文，出现英文段落/语法→英语，出现物理量/力学/电路→物理，出现化学式/反应→化学，出现细胞/遗传/生态→生物，出现朝代/事件→历史，出现地图/气候→地理，出现制度/法律→政治，无法判断→其他）",',
            '  "topic": "知识点",',
            '  "difficulty": "难度",',
            '  "student_answer_found": true/false,',
            '  "sub_questions": [',
            "    {",
            '      "label": "小题编号，如 (a)/①，单题则填\\"主\\"",',
            '      "is_correct": true/false,',
            '      "score": 得分,',
            '      "max_score": 满分,',
            '      "comment": "该小题详细评语"',
            "    }",
            "  ]",
            "}",
            "",
            "重要：",
            "- 如果 OCR 文本中确实没有学生作答痕迹，设 student_answer_found 为 false",
            "- comment 绝不能为空",
            "- 英文作业请用中文批注",
        ])

    def _extract_json(self, text: str) -> Optional[Dict]:
        """从 LLM 响应中尝试多种方式提取 JSON，失败返回 None"""
        # 策略1：直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略2：去掉 markdown 代码块
        cleaned = text
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # 策略3：在文本中搜索 JSON 对象（找第一个 { 和最后一个 }）
        #    如果没有 }，说明输出可能被截断，取从 { 到文本末尾
        candidate = None
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start:end + 1]
        elif start >= 0:
            # 可能被截断，没有闭合的 }
            candidate = cleaned[start:]
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 策略4：修复截断的 JSON（max_tokens 不足导致输出被截断时，补全缺失的 } ] "）
        if candidate is not None:
            try:
                open_braces = candidate.count("{") - candidate.count("}")
                open_brackets = candidate.count("[") - candidate.count("]")
                in_string = candidate.count('"') % 2 == 1
                fixed = candidate
                if in_string:
                    # 被截断的字符串：补闭合引号 + 截断标记
                    fixed += '…[截断]"'
                    open_braces += 0  # 截断标记在字符串内，括号计数不变
                fixed += "]" * open_brackets
                fixed += "}" * open_braces
                result = json.loads(fixed)
                # 检查 comment 字段是否被截断（末尾无句号/感叹号/问号）
                for key in ("comment",):
                    if key in result and isinstance(result[key], str):
                        val = result[key].rstrip()
                        if val and val[-1] not in "。！？….!?\n":
                            result[key] = val + "…[截断]"
                return result
            except json.JSONDecodeError:
                pass

        # 策略5：替换单引号为双引号
        if candidate is not None:
            try:
                fixed = candidate.replace("'", '"')
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        # 策略6：修复常见 JSON 错误（尾部逗号、未转义字符）
        target = candidate if candidate is not None else cleaned
        try:
            # 移除尾部逗号
            target = re.sub(r',\s*}', '}', target)
            target = re.sub(r',\s*]', ']', target)
            # 修复转义问题
            target = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', target)
            return json.loads(target)
        except json.JSONDecodeError:
            pass

        # 策略7：修复截断+尾部逗号+注释的组合问题
        if candidate is not None:
            try:
                fixed = re.sub(r'//[^\n]*', '', candidate)
                fixed = re.sub(r',\s*}', '}', fixed)
                fixed = re.sub(r',\s*]', ']', fixed)
                open_braces = fixed.count("{") - fixed.count("}")
                open_brackets = fixed.count("[") - fixed.count("]")
                in_string = fixed.count('"') % 2 == 1
                if in_string:
                    fixed += '"'
                fixed += "]" * open_brackets
                fixed += "}" * open_braces
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        logger.error(f"_extract_json 全部 7 策略失败，原始文本前200字: {text[:200]}")
        return None

    def grade(self, ocr_text: str, reference_answer: Optional[str] = None) -> Dict:
        """批改 OCR 文本中的作业。

        ocr_text: OCR 从作业图片中提取的文本，同时包含题目和学生答案。
        LLM 自行区分题目和答案，并拆分多小题。
        """
        system_prompt = self.grading_prompt or self._default_system_prompt()
        user_content = self.build_evaluation_context(ocr_text)
        if reference_answer:
            user_content += "\n\n【参考答案】：" + reference_answer

        fallback_result = {
            "is_correct": False,
            "score": 0,
            "max_score": 0,
            "comment": "大模型批改失败",
            "subject": "未分类",
            "topic": "未分类",
            "difficulty": "中等",
            "student_answer_found": False,
            "sub_questions": [],
        }

        try:
            response = self._call_with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=3072,
            ))
            reply = response.choices[0].message.content
        except Exception as api_err:
            logger.error(f"大模型 API 调用失败: {api_err}")
            result = dict(fallback_result)
            result["comment"] = f"API 调用失败: {str(api_err)[:150]}"
            return result

        reply = reply.strip()
        # 尝试多种方式提取 JSON
        result = self._extract_json(reply)
        if result is None:
            logger.error(f"JSON 解析失败，模型原始输出(前300字): {reply[:300]}")
            result = dict(fallback_result)
            result["comment"] = f"模型输出格式异常，无法解析。原始响应: {reply[:200]}"
            result["sub_questions"] = []

        # 确保必要字段存在
        result.setdefault("is_correct", False)
        result.setdefault("score", 0)
        result.setdefault("max_score", 0)
        result.setdefault("comment", "")
        result.setdefault("subject", "未分类")
        result.setdefault("topic", "未分类")
        result.setdefault("difficulty", "中等")
        result.setdefault("student_answer_found", True)
        result.setdefault("sub_questions", [])

        # 交叉验证学科分类
        result["subject"] = auto_detect_subject(
            # 从 ocr_text 中取前500字符用于学科检测
            (user_content or "")[:500], result.get("subject", "")
        )

        logger.info(
            f"批改完成: is_correct={result.get('is_correct')}, "
            f"得分={result.get('score')}/{result.get('max_score')}, "
            f"小题数={len(result.get('sub_questions', []))}, "
            f"找到作答={result.get('student_answer_found')}"
        )
        result = self._enforce_consistency(result)
        return result

    def grade_vision(self, image_path: str, reference_answer: Optional[str] = None) -> Dict:
        """使用视觉模型直接看图片批改，跳过 OCR。

        image_path: 作业图片路径（裁剪后的题目区域或整页图）
        使用通义千问 VL 模型直接分析图片中的题目和学生答案。
        """
        if not self.vl_client:
            raise RuntimeError("视觉模型未初始化，请检查 DashScope API Key 配置")

        # 准备图片（自动压缩大图）
        data_url = self._prepare_image_for_api(image_path)

        system_prompt = "\n".join([
            "你是一位专业的作业批改老师。你会看到一张作业图片，",
            "图片中可能包含题目和学生的作答（印刷或手写）。",
            "",
            "请完成以下任务：",
            "1. 识别图片中的所有文字内容（印刷体和手写体）",
            "2. 区分题目内容和学生的答案",
            "3. 判断学生答案是否正确",
            "4. 给出得分和详细评语",
            "5. 如果一道题有多个小题，请拆分并分别评分",
            "",
            "给分规则：简单题5分，中等题10分，困难题15分",
            "评语要求：",
            "  - 正确：写'✓ 正确'，补充解题思路",
            "  - 错误：包含正确答案 + 错误原因 + 正确思路",
            "",
            "输出纯 JSON（不要用 ``` 包裹）：",
            "{",
            '  "is_correct": true/false,',
            '  "score": 总得分,',
            '  "max_score": 总满分,',
            '  "comment": "整体评语",',
            '  "subject": "学科（必填！根据题目内容判断：出现数字/公式/方程→数学，出现古文/诗词/阅读理解(中文)→语文，出现英文段落/语法→英语，出现物理量/力学/电路→物理，出现化学式/反应→化学，出现细胞/遗传/生态→生物，出现朝代/事件→历史，出现地图/气候→地理，出现制度/法律→政治，无法判断→其他）",',
            '  "topic": "知识点",',
            '  "difficulty": "简单/中等/困难",',
            '  "student_answer_found": true/false,',
            '  "ocr_text": "图片中识别出的全部文字内容",',
            '  "sub_questions": [',
            "    {",
            '      "label": "小题编号",',
            '      "is_correct": true/false,',
            '      "score": 得分,',
            '      "max_score": 满分,',
            '      "comment": "该小题评语"',
            "    }",
            "  ]",
            "}",
        ])

        user_text = "请批改这张作业图片中的题目。"

        fallback_result = {
            "is_correct": False,
            "score": 0,
            "max_score": 0,
            "comment": "视觉模型批改失败",
            "subject": "未分类",
            "topic": "未分类",
            "difficulty": "中等",
            "student_answer_found": False,
            "sub_questions": [],
            "ocr_text": "",
        }

        try:
            response = self._call_with_retry(lambda: self.vl_client.chat.completions.create(
                model=self.vl_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=3072,
            ))
            reply = response.choices[0].message.content
        except Exception as api_err:
            logger.error(f"视觉模型 API 调用失败: {api_err}")
            result = dict(fallback_result)
            result["comment"] = f"视觉 API 调用失败: {str(api_err)[:150]}"
            return result

        # JSON 提取
        result = self._extract_json(reply.strip())
        if result is None:
            logger.error(f"视觉模型 JSON 解析失败，原始输出(前300字): {reply[:300]}")
            result = dict(fallback_result)
            result["comment"] = f"模型输出格式异常。原始响应: {reply[:200]}"
            result["sub_questions"] = []

        result.setdefault("is_correct", False)
        result.setdefault("score", 0)
        result.setdefault("max_score", 0)
        result.setdefault("comment", "")
        result.setdefault("subject", "未分类")
        result.setdefault("topic", "未分类")
        result.setdefault("difficulty", "中等")
        result.setdefault("student_answer_found", True)
        result.setdefault("sub_questions", [])
        result.setdefault("ocr_text", "")

        # 交叉验证学科分类
        result["subject"] = auto_detect_subject(
            result.get("ocr_text", ""), result.get("subject", "")
        )

        logger.info(
            f"视觉批改完成: is_correct={result.get('is_correct')}, "
            f"得分={result.get('score')}/{result.get('max_score')}, "
            f"小题数={len(result.get('sub_questions', []))}"
        )
        result = self._enforce_consistency(result)
        return result

    def grade_full_page(self, image_path: str, reference_answer: Optional[str] = None) -> Dict:
        """整页批改：将完整作业页面发给视觉模型，一次识别所有题目和答案。

        与 grade_vision 不同，此方法要求模型找出页面中 ALL 题目，
        并以 sub_questions 数组返回每一题的独立结果。
        适用于：阅读理解（文章+多道选择题同一页）、试卷等场景。
        """
        if not self.vl_client:
            raise RuntimeError("视觉模型未初始化，请检查 DashScope API Key 配置")

        # 准备图片（自动压缩大图）
        data_url = self._prepare_image_for_api(image_path)

        system_prompt = "\n".join([
            "你是一位专业的中小学作业批改老师。请仔细查看这张完整的作业图片。",
            "",
            "## 你的任务",
            "1. 识别图片中所有文字（印刷体和手写体）",
            "2. 区分哪些是题目/文章，哪些是学生写下的答案",
            "3. 注意：阅读理解题会有文章段落 + 多道选择题，请找到每道题的学生选项",
            "4. 逐题判断学生答案是否正确",
            "5. 给出得分和详细解析",
            "",
            "## 重要：仔细核对学生的每个答案！",
            "- 仔细看学生选的是 A/B/C/D 哪个选项",
            "- 根据文章内容判断该选项是否正确",
            "- 如果学生选错了，必须明确指出：正确答案是什么、为什么学生选错了",
            "- 不要轻易判断为'正确'，请认真对照文章内容核实",
            "",
            "## 给分规则",
            "- 简单题5分，中等题10分，困难题15分",
            "- 如果图片中有多道题（如阅读理解46-50题），每道题单独评分",
            "",
            "## 输出格式（纯 JSON，不要 ``` 包裹）",
            "{",
            '  "is_correct": 整体是否全对,',
            '  "score": 总得分,',
            '  "max_score": 总满分,',
            '  "comment": "整体评语（汇总各题情况）",',
            '  "subject": "学科，必须填！选：数学/语文/英语/物理/化学/生物/历史/地理/政治/其他",',
            '  "topic": "知识点",',
            '  "difficulty": "简单/中等/困难",',
            '  "student_answer_found": 是否找到学生作答(true/false),',
            '  "ocr_text": "图片中识别出的文字内容摘要",',
            '  "sub_questions": [',
            "    {",
            '      "label": "题号（如46/47/48 或 1a/1b）",',
            '      "is_correct": true/false,',
            '      "score": 得分,',
            '      "max_score": 满分,',
            '      "comment": "详细评语：正确则写解题思路，错误则写(1)正确答案(2)为什么错(3)正确思路"',
            "    }",
            "  ]",
            "}",
            "",
            "注意：",
            "- sub_questions 必须包含图片中每一道有学生作答的题目",
            "- 如果图片中没有学生作答痕迹，student_answer_found 设为 false",
            "- comment 不能为空",
            "- 英文作业用中文评语",
            "- ⚠️ comment 必须控制在 150 字以内！只写结论，不要展开详细解题过程。",
        ])

        fallback_result = {
            "is_correct": False, "score": 0, "max_score": 0,
            "comment": "整页批改失败", "subject": "未分类", "topic": "未分类",
            "difficulty": "中等", "student_answer_found": False,
            "sub_questions": [], "ocr_text": "",
        }

        try:
            response = self._call_with_retry(lambda: self.vl_client.chat.completions.create(
                model=self.vl_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": "请批改这张完整的作业图片。仔细找每道题、看学生选了哪个选项、判断对错。"},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=4096,
            ))
            reply = response.choices[0].message.content
        except Exception as api_err:
            logger.error(f"整页批改 API 调用失败: {api_err}")
            result = dict(fallback_result)
            result["comment"] = f"视觉 API 调用失败: {str(api_err)[:150]}"
            return result

        result = self._extract_json(reply.strip())
        if result is None:
            logger.error(f"整页批改 JSON 解析失败，原始输出(前300字): {reply[:300]}")
            result = dict(fallback_result)
            result["comment"] = f"模型输出格式异常。原始响应: {reply[:200]}"

        result.setdefault("is_correct", False)
        result.setdefault("score", 0)
        result.setdefault("max_score", 0)
        result.setdefault("comment", "")
        result.setdefault("subject", "未分类")
        result.setdefault("topic", "未分类")
        result.setdefault("difficulty", "中等")
        result.setdefault("student_answer_found", True)
        result.setdefault("sub_questions", [])
        result.setdefault("ocr_text", "")

        # 交叉验证学科分类
        result["subject"] = auto_detect_subject(
            result.get("ocr_text", ""), result.get("subject", "")
        )

        logger.info(
            f"整页批改完成: 总分={result.get('score')}/{result.get('max_score')}, "
            f"小题数={len(result.get('sub_questions', []))}"
        )
        result = self._enforce_consistency(result)
        return result

    # ===================== 重做批改 =====================

    def _build_redo_prompt(self, previous_result: dict, hint: str = "",
                           sub_label: str = None) -> str:
        """构建重做时的纠错指令块，注入到 system prompt 中"""
        parts = [
            "❌ CRITICAL: A teacher has carefully reviewed your previous grading "
            "and determined it contains ERRORS. You MUST re-examine the image "
            "thoroughly and produce a DIFFERENT, corrected assessment.",
            "",
            "Do NOT simply repeat your previous answer. Look at the student's "
            "actual handwritten answers in the image — you may have missed or "
            "misread them the first time. Pay special attention to:",
            "- Which option (A/B/C/D) the student actually circled or wrote down",
            "- Handwritten numbers and symbols that you may have overlooked",
            "- Small markings or corrections near the student's answer area",
            "",
            "⚠️ CRITICAL: Keep ALL comment fields under 100 characters. "
            "Write ONLY the conclusion (e.g. '选A，正确答案C，模长为1'). "
            "Do NOT write step-by-step derivations or LaTeX — they cause "
            "output truncation and CORRUPT the JSON structure.",
        ]
        if sub_label:
            parts.append(
                f"Focus your re-evaluation on sub-question ({sub_label}) ONLY. "
                "Do NOT change the grading of other sub-questions."
            )
        if hint:
            parts.append(
                f"Teacher's specific hint (you MUST follow this): {hint}"
            )
        if previous_result:
            # 只传关键字段，避免信息过载
            prev_brief = {
                k: previous_result.get(k)
                for k in ("is_correct", "score", "max_score", "comment",
                          "sub_questions", "student_answer")
                if k in previous_result
            }
            parts.append(
                "Your previous (INCORRECT) grading result was:\n"
                + json.dumps(prev_brief, ensure_ascii=False, indent=2)
                + "\n\nIMPORTANT: The teacher says this result is WRONG. "
                "Find what you missed and correct it."
            )
        return "\n\n".join(parts)

    def _redo_vision(self, image_path: str, previous_result: dict,
                     hint: str = "", sub_label: str = None) -> Dict:
        """Vision 模式重做：发整页图 + 纠错指令给 Qwen-VL"""
        if not self.vl_client:
            raise RuntimeError("视觉模型未初始化，请检查 DashScope API Key 配置")

        data_url = self._prepare_image_for_api(image_path)

        # 复用 grade_full_page 的 system prompt 结构，前置纠错指令
        system_prompt = "\n".join([
            self._build_redo_prompt(previous_result, hint, sub_label),
            "",
            "---",
            "",
            "你是一位专业的中小学作业批改老师。请仔细查看这张完整的作业图片。",
            "",
            "## 你的任务",
            "1. 识别图片中所有文字（印刷体和手写体）",
            "2. 区分哪些是题目/文章，哪些是学生写下的答案",
            "3. 注意：阅读理解题会有文章段落 + 多道选择题，请找到每道题的学生选项",
            "4. 逐题判断学生答案是否正确",
            "5. 给出得分和详细解析",
            "",
            "## 重要：仔细核对学生的每个答案！",
            "- 仔细看学生选的是 A/B/C/D 哪个选项",
            "- 根据文章内容判断该选项是否正确",
            "- 如果学生选错了，必须明确指出：正确答案是什么、为什么学生选错了",
            "- 不要轻易判断为正确，请认真对照文章内容核实",
            "",
            "## 给分规则",
            "- 简单题5分，中等题10分，困难题15分",
            "- 如果图片中有多道题（如阅读理解46-50题），每道题单独评分",
            "",
            "## 输出格式（纯 JSON，不要 ``` 包裹）",
            "{",
            '  "is_correct": 整体是否全对,',
            '  "score": 总得分,',
            '  "max_score": 总满分,',
            '  "comment": "整体评语（汇总各题情况）",',
            '  "subject": "学科，必须填！",',
            '  "topic": "知识点",',
            '  "difficulty": "简单/中等/困难",',
            '  "student_answer_found": 是否找到学生作答(true/false),',
            '  "ocr_text": "图片中识别出的文字内容摘要",',
            '  "sub_questions": [',
            "    {",
            '      "label": "题号",',
            '      "is_correct": true/false,',
            '      "score": 得分,',
            '      "max_score": 满分,',
            '      "comment": "详细评语"',
            "    }",
            "  ]",
            "}",
            "",
            "注意：",
            "- sub_questions 必须包含图片中每一道有学生作答的题目",
            "- 如果图片中没有学生作答痕迹，student_answer_found 设为 false",
            "- comment 不能为空",
            "- 英文作业用中文评语",
            "- ⚠️ comment 必须控制在 150 字以内！只写结论，不要展开详细解题过程。",
        ])

        fallback = {
            "is_correct": False, "score": 0, "max_score": 0,
            "comment": "重做批改失败", "subject": "未分类", "topic": "未分类",
            "difficulty": "中等", "student_answer_found": False,
            "sub_questions": [], "ocr_text": "",
        }

        try:
            response = self._call_with_retry(lambda: self.vl_client.chat.completions.create(
                model=self.vl_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": (
                        "请极其仔细地重新查看这张图片中的学生作答。"
                        "你上次的批改被老师复核发现是【错误】的！"
                        "仔细看学生实际写了什么、选了哪个选项——你可能漏看了学生的笔迹。"
                        "必须给出与上次不同的、正确的批改结果。"
                        "comment 限制在200字以内，避免输出过长。"
                    )},
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=8192,
                timeout=120,
            ), max_retries=2)
            reply = response.choices[0].message.content
        except Exception as api_err:
            logger.error(f"重做 Vision 批改 API 失败: {api_err}")
            result = dict(fallback)
            result["comment"] = f"重做 API 失败: {str(api_err)[:150]}"
            return result

        result = self._extract_json(reply.strip())
        if result is None:
            logger.error(f"重做 Vision 批改 JSON 解析失败，前300字: {reply[:300]}")
            result = dict(fallback)
            result["comment"] = f"模型输出格式异常。原始响应: {reply[:200]}"

        result.setdefault("is_correct", False)
        result.setdefault("score", 0)
        result.setdefault("max_score", 0)
        result.setdefault("comment", "")
        result.setdefault("subject", "未分类")
        result.setdefault("topic", "未分类")
        result.setdefault("difficulty", "中等")
        result.setdefault("student_answer_found", True)
        result.setdefault("sub_questions", [])
        result.setdefault("ocr_text", "")

        result["subject"] = auto_detect_subject(
            result.get("ocr_text", ""), result.get("subject", "")
        )

        logger.info(
            f"重做 Vision 批改完成: is_correct={result.get('is_correct')}, "
            f"得分={result.get('score')}/{result.get('max_score')}, "
            f"小题数={len(result.get('sub_questions', []))}"
        )
        result = self._enforce_consistency(result)
        return result

    def _redo_text(self, previous_result: dict, hint: str = "",
                   sub_label: str = None) -> Dict:
        """OCR+LLM 文本模式重做：OCR 文本 + 纠错指令给 DeepSeek"""
        ocr_text = previous_result.get("ocr_text", "")
        system_prompt = "\n".join([
            self._build_redo_prompt(previous_result, hint, sub_label),
            "",
            "---",
            "",
            self._default_system_prompt(),
        ])

        user_content = self.build_evaluation_context(ocr_text)
        user_content += "\n\n【请重新批改，你上次的批改结果经老师复核认为是错误的。请非常仔细地逐题重新判断。】"

        fallback = {
            "is_correct": False, "score": 0, "max_score": 0,
            "comment": "重做批改失败", "subject": "未分类", "topic": "未分类",
            "difficulty": "中等", "student_answer_found": False,
            "sub_questions": [], "ocr_text": "",
        }

        try:
            response = self._call_with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=3072,
                timeout=60,
            ), max_retries=1)
            reply = response.choices[0].message.content
        except Exception as api_err:
            logger.error(f"重做文本批改 API 失败: {api_err}")
            result = dict(fallback)
            result["comment"] = f"重做 API 失败: {str(api_err)[:150]}"
            return result

        result = self._extract_json(reply.strip())
        if result is None:
            logger.error(f"重做文本批改 JSON 解析失败，前300字: {reply[:300]}")
            result = dict(fallback)
            result["comment"] = f"模型输出格式异常。原始响应: {reply[:200]}"

        result.setdefault("is_correct", False)
        result.setdefault("score", 0)
        result.setdefault("max_score", 0)
        result.setdefault("comment", "")
        result.setdefault("subject", "未分类")
        result.setdefault("topic", "未分类")
        result.setdefault("difficulty", "中等")
        result.setdefault("student_answer_found", True)
        result.setdefault("sub_questions", [])

        result["subject"] = auto_detect_subject(
            (ocr_text or "")[:500], result.get("subject", "")
        )

        logger.info(
            f"重做文本批改完成: is_correct={result.get('is_correct')}, "
            f"得分={result.get('score')}/{result.get('max_score')}"
        )
        result = self._enforce_consistency(result)
        return result

    def _redo_vision_deepseek(self, image_path: str, previous_result: dict,
                              hint: str = "", sub_label: str = None) -> Dict:
        """Vision+DeepSeek 双阶段重做：VL 提取文字 + DeepSeek 深度判题"""
        if not self.vl_client:
            raise RuntimeError("视觉模型未初始化")

        data_url = self._prepare_image_for_api(image_path)

        # Stage 1: VL 提取文字（带重做指令）
        stage1_prompt = "\n".join([
            "You are an OCR assistant. Extract ALL text from this homework image.",
            self._build_redo_prompt(previous_result, hint, sub_label),
            "",
            "Output format (JSON):",
            '{"full_text": "ALL text in the image", "questions": [{"label":"1","question_text":"...","student_answer":"..."}]}',
        ])

        fallback = {
            "is_correct": False, "score": 0, "max_score": 0,
            "comment": "Vision+DeepSeek 重做失败", "subject": "未分类",
            "topic": "未分类", "difficulty": "中等",
            "student_answer_found": False, "sub_questions": [], "ocr_text": "",
        }

        try:
            resp1 = self._call_with_retry(lambda: self.vl_client.chat.completions.create(
                model=self.vl_model,
                messages=[
                    {"role": "system", "content": stage1_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "Extract ALL text from this image for re-grading."},
                    ]},
                ],
                temperature=0.05,
                max_tokens=4096,
            ), max_retries=1)
            stage1_reply = resp1.choices[0].message.content
        except Exception as e:
            logger.error(f"重做 Stage1 VL 提取失败: {e}")
            fallback["comment"] = f"重做 VL 提取失败: {str(e)[:150]}"
            return fallback

        extracted = self._extract_json(stage1_reply.strip())
        if extracted is None:
            extracted = {"full_text": stage1_reply, "questions": []}
        full_text = extracted.get("full_text", stage1_reply)
        questions = extracted.get("questions", [])

        if not questions:
            questions = [{"label": "1", "question_text": full_text[:500], "student_answer": ""}]

        # Stage 2: DeepSeek 判题
        q_lines = []
        for i, q in enumerate(questions):
            label = q.get("label", str(i + 1))
            q_text = q.get("question_text", "")
            s_ans = q.get("student_answer", "")
            options = q.get("options", {})
            q_block = f"### Question {label}\n**Problem**: {q_text}\n"
            if options:
                opt_str = "  ".join(f"{k}. {v}" for k, v in sorted(options.items()))
                q_block += f"**Choices**: {opt_str}\n"
            q_block += f"**Student's answer**: {s_ans or '(not found)'}\n"
            q_lines.append(q_block)

        stage2_prompt = "\n".join([
            "你是一位专业的中小学作业批改老师。",
            self._build_redo_prompt(previous_result, hint, sub_label),
            "",
            "请逐题判断对错，给出评分和详细解析。",
            "输出纯 JSON：",
            '{"is_correct":bool,"score":0,"max_score":0,"comment":"","subject":"","topic":"","difficulty":"中等","student_answer_found":true,',
            '"sub_questions":[{"label":"1","is_correct":true/false,"score":0,"max_score":0,"comment":"详细评语"}]}',
        ])

        try:
            resp2 = self._call_with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": stage2_prompt},
                    {"role": "user", "content": f"## Reading\n{full_text}\n\n## Questions\n{''.join(q_lines)}"},
                ],
                temperature=0.1,
                max_tokens=3072,
            ), max_retries=1)
            result = self._extract_json(resp2.choices[0].message.content.strip())
        except Exception as e:
            logger.error(f"重做 Stage2 DeepSeek 判题失败: {e}")
            fallback["comment"] = f"重做 DeepSeek 判题失败: {str(e)[:150]}"
            fallback["ocr_text"] = full_text
            return fallback

        if result is None:
            fallback["ocr_text"] = full_text
            fallback["comment"] = "重做判题 JSON 解析失败"
            return fallback

        result.setdefault("is_correct", False)
        result.setdefault("score", 0)
        result.setdefault("max_score", 0)
        result.setdefault("comment", "")
        result.setdefault("subject", "未分类")
        result.setdefault("topic", "未分类")
        result.setdefault("difficulty", "中等")
        result.setdefault("student_answer_found", True)
        result.setdefault("sub_questions", [])
        result.setdefault("ocr_text", full_text)
        result["subject"] = auto_detect_subject(full_text, result.get("subject", ""))

        logger.info(f"重做 Vision+DeepSeek 完成: 得分={result.get('score')}/{result.get('max_score')}")
        result = self._enforce_consistency(result)
        return result

    def redo_question(self, image_path: str, previous_result: dict,
                      hint: str = "", sub_label: str = None) -> Dict:
        """重做批改：统一使用 Qwen-VL 视觉模型看图重判（当前准确率最高方案）

        不再区分 grading_mode —— Vision 模型 69% 准确率远超 OCR+LLM 的 56%，
        且一次 API 调用即可完成识别+判题，无需 OCR 中间环节。

        Args:
            image_path: 整页图片路径
            previous_result: 上次批改结果 dict（用于纠错指令）
            hint: 老师提示（可选）
            sub_label: 小题编号（可选，聚焦重做特定小题）

        Returns:
            新的批改结果 dict（与 grade_full_page 格式一致）
        """
        return self._redo_vision(image_path, previous_result, hint, sub_label)

    def grade_vision_deepseek(self, image_path: str,
                               reference_answer: Optional[str] = None) -> Dict:
        """Vision+DeepSeek 双阶段批改。

        Stage 1: Qwen-VL 看图提取题目文字 + 学生答案文字（不做判断）
        Stage 2: DeepSeek 文本模型做深度推理判题（比 VL 模型数学推理更强）

        优势：视觉模型擅长读图识字，文本模型擅长逻辑推理判对错，各取所长。
        """
        if not self.vl_client:
            raise RuntimeError("视觉模型未初始化，请检查 DashScope API Key 配置")

        data_url = self._prepare_image_for_api(image_path)

        # ═══════ Stage 1: Qwen-VL 提取文字 ═══════
        stage1_prompt = "\n".join([
            "You are an OCR assistant specialized in math homework. Extract ALL text from this image.",
            "DO NOT grade or judge — only extract what you see.",
            "",
            "## What to extract",
            "1. PRINTED problem text — copy the original math problem word-for-word in English",
            "2. HANDWRITTEN student work — transcribe ALL handwritten numbers, symbols, and text",
            "3. Math formulas → LaTeX inline (e.g. $x^2 + 3x = 0$)",
            "4. If the handwriting is messy, do your best. Mark uncertain parts with (?)",
            "5. Never make up content that isn't in the image",
            "",
            "## Important: Separate problem from student work",
            "- The printed text is the QUESTION",
            "- The handwritten text is the STUDENT's ANSWER",
            "- Extract BOTH completely and separately",
            "",
            "## JSON output format (no markdown fences)",
            "{",
            '  "full_text": "ALL text in the image (problem + student work)",',
            '  "questions": [',
            "    {",
            '      "label": "1",',
            '      "question_text": "The printed problem text (original English)",',
            '      "student_answer": "The HANDWRITTEN work — ALL numbers, formulas, text the student wrote"',
            "    }",
            "  ]",
            "}",
        ])

        fallback = {
            "is_correct": False, "score": 0, "max_score": 0,
            "comment": "Vision+DeepSeek 批改失败", "subject": "未分类",
            "topic": "未分类", "difficulty": "中等",
            "student_answer_found": False, "sub_questions": [], "ocr_text": "",
        }

        # --- Stage 1 调用 ---
        try:
            resp1 = self._call_with_retry(lambda: self.vl_client.chat.completions.create(
                model=self.vl_model,
                messages=[
                    {"role": "system", "content": stage1_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": "Extract ALL text from this homework image. Separate the printed problem from the handwritten student work."},
                    ]},
                ],
                temperature=0.05,
                max_tokens=4096,
            ))
            stage1_reply = resp1.choices[0].message.content
        except Exception as e:
            logger.error(f"Stage1 Qwen-VL 提取失败: {e}")
            fallback["comment"] = f"视觉提取失败: {str(e)[:150]}"
            return fallback

        extracted = self._extract_json(stage1_reply.strip())
        if extracted is None:
            logger.error(f"Stage1 JSON 解析失败: {stage1_reply[:300]}")
            # 回退：将原始回复作为 ocr_text 做文本批改
            extracted = {"full_text": stage1_reply, "questions": []}

        passage = extracted.get("passage", "")
        full_text = extracted.get("full_text", stage1_reply)
        questions = extracted.get("questions", [])

        if not questions:
            questions = [{
                "label": "1",
                "question_text": full_text[:500],
                "student_answer": "",
                "options": {},
            }]

        logger.info(
            f"Stage1 完成: 提取到 {len(questions)} 道题, "
            f"passage长度={len(passage)}, full_text长度={len(full_text)}"
        )

        # ═══════ Stage 2: DeepSeek 深度推理判题 ═══════
        # 构建阅读文章（如果有）
        passage_block = ""
        if passage:
            passage_block = f"## Reading Passage\n{passage}\n\n"

        # 构建题目列表（英文格式）
        q_lines = []
        for i, q in enumerate(questions):
            label = q.get("label", str(i + 1))
            q_text = q.get("question_text", "")
            s_ans = q.get("student_answer", "")
            options = q.get("options", {})
            q_block = f"### Question {label}\n**Problem**: {q_text}\n"
            if options:
                opt_str = "  ".join(f"{k}. {v}" for k, v in sorted(options.items()))
                q_block += f"**Choices**: {opt_str}\n"
            q_block += f"**Student's answer**: {s_ans or '(not found)'}\n"
            q_lines.append(q_block)
        questions_block = "\n".join(q_lines)

        stage2_prompt = "\n".join([
            "你是一位专业的中小学作业批改老师。下面是从作业图片中提取的内容。",
            "你的任务是逐题判断对错，给出评分和详细解析。",
            "",
            "## 批改要求",
            "1. 【阅读理解】仔细阅读文章，根据文章内容判断每道题的正确答案，再对比学生选择",
            "2. 【选择题】从选项中找出正确答案，看学生选对没有",
            "3. 【解答题】自行验证计算过程，判断学生答案对错",
            "4. 每题给出详细评语：正确→解题思路；错误→(1)正确答案(2)为什么错(3)正确思路",
            "",
            "## 给分规则",
            "- 简单题5分，中等题10分，困难题15分",
            "- 没有学生作答→0分，comment注明",
            "",
            "## 输出格式（纯 JSON，不要 ``` 包裹）",
            "{",
            '  "is_correct": 整体是否全对,',
            '  "score": 总得分,',
            '  "max_score": 总满分,',
            '  "comment": "整体评语（汇总各题情况）",',
            '  "subject": "学科，必须填！选：数学/语文/英语/物理/化学/生物/历史/地理/政治/其他",',
            '  "topic": "知识点",',
            '  "difficulty": "简单/中等/困难",',
            '  "student_answer_found": 是否找到学生作答,',
            '  "ocr_text": "视觉提取的原始文字",',
            '  "sub_questions": [',
            "    {",
            '      "label": "题号",',
            '      "is_correct": true/false,',
            '      "score": 得分,',
            '      "max_score": 满分,',
            '      "comment": "详细评语（必须包含判断理由）"',
            "    }",
            "  ]",
            "}",
        ])

        # 构建最终用户消息
        user_msg = f"{passage_block}## Questions\n{questions_block}"

        # ═══════ Stage 2: DeepSeek 三路推理投票 ═══════
        VOTER_PROMPTS = [
            "\n".join([
                "You are a math teacher grading homework. Below is text extracted from a homework image.",
                "The 'question_text' is the problem. The 'student_answer' is what the student wrote.",
                "For each question: solve it yourself first, then compare with the student's answer.",
                "Each question is worth 5 points. Output ONLY this JSON:",
                '{"sub_questions":[{"label":"1","is_correct":true/false,"score":5,"max_score":5,"comment":"Explain your reasoning in English"}]}',
                "Important: really SOLVE the problem yourself, don't just guess.",
            ]),
            "\n".join([
                "You are a rigorous exam reviewer. Below is text from a homework image.",
                "Assume the student's answer is WRONG until you find clear evidence it is correct.",
                "Solve each problem yourself, then check the student's work step by step.",
                "Each question 5 points. Output ONLY this JSON:",
                '{"sub_questions":[{"label":"1","is_correct":true/false,"score":5,"max_score":5,"comment":"Show your verification in English"}]}',
            ]),
            "\n".join([
                "You are a math tutor. Below is text from a homework image.",
                "Judge whether the student understands the concept, not just the final answer.",
                "Solve the problem, then evaluate the student's approach.",
                "Each question 5 points. Output ONLY this JSON:",
                '{"sub_questions":[{"label":"1","is_correct":true/false,"score":5,"max_score":5,"comment":"Give helpful feedback in English"}]}',
            ]),
        ]

        def _call_voter(idx):
            try:
                resp = self._call_with_retry(lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": VOTER_PROMPTS[idx]},
                        {"role": "user", "content": f"Grade this homework:\n\n{user_msg}"},
                    ],
                    temperature=0.1 if idx == 0 else 0.2,
                    max_tokens=3072,
                ))
                return self._extract_json(resp.choices[0].message.content.strip())
            except Exception as e:
                logger.warning(f"投票路径{idx}失败: {e}")
                return None

        import concurrent.futures
        voter_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(_call_voter, i) for i in range(3)]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                if r:
                    voter_results.append(r)

        if not voter_results:
            # 回退：用单个简单调用再试一次
            logger.warning("三路投票全部失败，尝试单次回退...")
            try:
                resp = self._call_with_retry(lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是作业批改老师。输出纯JSON: {\"sub_questions\":[{\"label\":\"1\",\"is_correct\":true/false,\"score\":0,\"max_score\":0,\"comment\":\"评语\"}]}"},
                        {"role": "user", "content": f"批改以下题目:\n\n{user_msg}"},
                    ],
                    temperature=0.1,
                    max_tokens=3072,
                ))
                r = self._extract_json(resp.choices[0].message.content.strip())
                if r:
                    voter_results = [r]
                    logger.info("单次回退成功")
            except Exception as e:
                logger.error(f"回退也失败: {e}")
            if not voter_results:
                logger.error("三路投票+回退全部失败")
                fallback["comment"] = "三路推理全部失败"
                fallback["ocr_text"] = full_text
                return fallback

        # ── 标签归一化 + 按题汇总投票 ──
        import re

        def _norm_label(label: str) -> str:
            """统一题号格式: '46' '第46题' '46题' 'Q46' → '46'"""
            m = re.search(r'(\d+)', str(label))
            return m.group(1) if m else str(label)

        all_votes = {}
        for vr in voter_results:
            for sq in vr.get("sub_questions", []):
                key = _norm_label(sq.get("label", "?"))
                all_votes.setdefault(key, []).append(sq)

        final_subs = []
        total_score = 0
        total_max = 0
        SCORE_PER_Q = 5  # 阅读题每题5分

        for key in sorted(all_votes.keys(), key=lambda k: int(k) if k.isdigit() else 99):
            votes = all_votes[key]
            yes = sum(1 for v in votes if v.get("is_correct"))
            no = len(votes) - yes
            is_correct = yes > no
            # 用多数派中第一条最详细的评语
            majority_votes = [v for v in votes if v.get("is_correct") == is_correct]
            best_comment = max((v.get("comment", "") for v in majority_votes), key=len) if majority_votes else ""
            if len(best_comment) > 150:
                best_comment = best_comment[:150] + "…"

            final_subs.append({
                "label": key,
                "is_correct": is_correct,
                "score": SCORE_PER_Q if is_correct else 0,
                "max_score": SCORE_PER_Q,
                "comment": best_comment,
            })
            total_score += SCORE_PER_Q if is_correct else 0
            total_max += SCORE_PER_Q

        # 用第一条结果填充元信息
        base = voter_results[0]
        base.setdefault("subject", "未分类")
        base.setdefault("topic", "未分类")
        base.setdefault("difficulty", "中等")
        base["ocr_text"] = full_text
        base["student_answer_found"] = len(final_subs) > 0
        base["sub_questions"] = final_subs
        base["is_correct"] = all(sq["is_correct"] for sq in final_subs) if final_subs else fallback["is_correct"]
        base["score"] = total_score
        base["max_score"] = total_max
        n_correct = sum(1 for s in final_subs if s['is_correct'])
        base["comment"] = f"三路投票: {n_correct}✓/{len(final_subs)-n_correct}✗ (共{len(final_subs)}题)"
        base["subject"] = auto_detect_subject(full_text, base.get("subject", ""))
        return base

    def batch_grade(self, questions: List[Dict]) -> List[Dict]:
        """批量批改多道题目"""
        results = []
        for item in questions:
            result = self.grade(
                ocr_text=item.get("ocr_text", item.get("question", "")),
                reference_answer=item.get("reference_answer"),
            )
            results.append(result)
        return results

    def calculate_total_score(self, grading_results: List[Dict]) -> Tuple[float, float]:
        """计算总分和得分"""
        total_max = sum(item.get("max_score", 0) for item in grading_results)
        earned_score = sum(item.get("score", 0) for item in grading_results)
        return earned_score, total_max

    def generate_feedback(self, grading_result: List[Dict]) -> str:
        """根据批改结果生成总结评语"""
        correct_count = sum(1 for r in grading_result if r.get("is_correct", False))
        total_count = len(grading_result)
        total_earned = sum(r.get("score", 0) for r in grading_result)
        total_max = sum(r.get("max_score", 0) for r in grading_result)

        comments = [
            r.get("comment", "")
            for r in grading_result
            if not r.get("is_correct", False)
        ]

        prompt = f"""共{total_count}道题，{correct_count}道题正确，得分{total_earned}/{total_max}
错误题目评语:{'; '.join(comments)}
请用一句话写出鼓励性总结评语."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()
