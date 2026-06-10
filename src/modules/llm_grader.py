"""
大模型批改模块
负责调用大语言模型对作业内容进行智能批改和评分。
v3: 支持从 OCR 文本中自动区分题目和答案、拆分多小题独立评分。
"""
import json
import re
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


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
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.grading_prompt = ""

        # 初始化视觉模型客户端（通义千问 VL）
        vl_config = config.get("dashscope", {})
        self.vl_api_key = vl_config.get("api_key", "") or os.environ.get("DASHSCOPE_API_KEY")
        self.vl_model = vl_config.get("model", "qwen-vl-max")
        self.vl_base_url = vl_config.get("api_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.vl_client = None
        if self.vl_api_key:
            self.vl_client = OpenAI(api_key=self.vl_api_key, base_url=self.vl_base_url)
            logger.info(f"视觉模型初始化成功（model: {self.vl_model}）")
        else:
            logger.warning("未配置 DashScope API Key，视觉批改不可用")

        logger.info(f"大模型服务初始化成功（model: {self.model}）")

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
            '  "subject": "学科",',
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
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 策略4：替换单引号为双引号
        try:
            fixed = candidate.replace("'", '"') if 'candidate' in dir() else cleaned.replace("'", '"')
            return json.loads(fixed)
        except (json.JSONDecodeError, UnboundLocalError):
            pass

        # 策略5：修复常见 JSON 错误（尾部逗号、未转义字符）
        try:
            target = candidate if 'candidate' in dir() else cleaned
            # 移除尾部逗号
            target = re.sub(r',\s*}', '}', target)
            target = re.sub(r',\s*]', ']', target)
            # 修复转义问题
            target = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', target)
            return json.loads(target)
        except (json.JSONDecodeError, UnboundLocalError):
            pass

        logger.error(f"_extract_json 全部策略失败，原始文本前200字: {text[:200]}")
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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
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

        logger.info(
            f"批改完成: is_correct={result.get('is_correct')}, "
            f"得分={result.get('score')}/{result.get('max_score')}, "
            f"小题数={len(result.get('sub_questions', []))}, "
            f"找到作答={result.get('student_answer_found')}"
        )
        return result

    def grade_vision(self, image_path: str, reference_answer: Optional[str] = None) -> Dict:
        """使用视觉模型直接看图片批改，跳过 OCR。

        image_path: 作业图片路径（裁剪后的题目区域或整页图）
        使用通义千问 VL 模型直接分析图片中的题目和学生答案。
        """
        if not self.vl_client:
            raise RuntimeError("视觉模型未初始化，请检查 DashScope API Key 配置")

        # 将图片编码为 base64
        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        # 根据扩展名确定 MIME 类型
        ext = Path(image_path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        data_url = f"data:{mime};base64,{img_b64}"

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
            '  "subject": "学科",',
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
            response = self.vl_client.chat.completions.create(
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
                max_tokens=2048,
            )
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

        logger.info(
            f"视觉批改完成: is_correct={result.get('is_correct')}, "
            f"得分={result.get('score')}/{result.get('max_score')}, "
            f"小题数={len(result.get('sub_questions', []))}"
        )
        return result

    def grade_full_page(self, image_path: str, reference_answer: Optional[str] = None) -> Dict:
        """整页批改：将完整作业页面发给视觉模型，一次识别所有题目和答案。

        与 grade_vision 不同，此方法要求模型找出页面中 ALL 题目，
        并以 sub_questions 数组返回每一题的独立结果。
        适用于：阅读理解（文章+多道选择题同一页）、试卷等场景。
        """
        if not self.vl_client:
            raise RuntimeError("视觉模型未初始化，请检查 DashScope API Key 配置")

        import base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        data_url = f"data:{mime};base64,{img_b64}"

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
        ])

        fallback_result = {
            "is_correct": False, "score": 0, "max_score": 0,
            "comment": "整页批改失败", "subject": "未分类", "topic": "未分类",
            "difficulty": "中等", "student_answer_found": False,
            "sub_questions": [], "ocr_text": "",
        }

        try:
            response = self.vl_client.chat.completions.create(
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
                max_tokens=3072,
            )
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

        logger.info(
            f"整页批改完成: 总分={result.get('score')}/{result.get('max_score')}, "
            f"小题数={len(result.get('sub_questions', []))}"
        )
        return result

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
