"""
大模型批改模块
负责调用大语言模型对作业内容进行智能批改和评分。
"""
# 用于解析大模型返回的json格式结果
import json
# 类型注解
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
        logger.info(f"大模型服务初始化成功（model: {self.model}）")

    def load_prompt(self, prompt_template: str) -> None:
        """加载批改提示词模板"""
        self.grading_prompt = prompt_template

    def build_evaluation_context(self, question: str, student_answer: str, reference_answer: Optional[str] = None) -> str:
        """构建批改上下文"""
        context_parts = [f"题目：{question}", f"学生答案：{student_answer}"]
        if reference_answer:
            context_parts.append(f"参考答案：{reference_answer}")
        return "\n".join(context_parts)

    def grade(self, question: str, student_answer: str, reference_answer: Optional[str] = None) -> Dict:
        """对单道题目进行批改
        大模型对话接口通常接受3种角色的消息
        1.system(系统)
        2.user(用户)
        3.assistant(助手)
        """
        if self.grading_prompt:
            system_prompt = self.grading_prompt
        else:
            # 创建系统提示 大模型需要按照其中的要求来完成任务
            system_prompt = """你是一位专业的作业批改老师。请根据以下规则批改学生答案：
            1. 判断学生答案是否正确
            2. 给出得分（满分由题目难度决定，简单题5分，中等题10分，困难题15分）
            3. 评语要求：
               - 正确：只写"正确"二字
               - 错误：必须包含三部分：(1)正确答案是什么 (2)为什么是这个答案 (3)正确的解题思路或方法
            4. 以JSON格式输出：{"is_correct": true/false, "score": 分数, "max_score": 满分, "comment": "评语"}"""
        # 用户的指令
        user_content = self.build_evaluation_context(question, student_answer, reference_answer)
        # 调用大模型api
        response = self.client.chat.completions.create(
            # 指定大模型名称 在__init__方法种初始化好了
            model = self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            # temperature 控制随机性 0 完全确定 1 完全随机
            temperature = 0.2,
            # 最大生成长度 token是 大模型处理文本的基本单位
            max_tokens = 1024,
        )
        # reply 是一个字符串'{is_correct:false,:,}'等外面是单引号,里面是一个字典
        reply = response.choices[0].message.content

        try:
            result  = json.loads(reply)
        except json.JSONDecodeError:
            logger.error(f"解析大模型输出失败：{reply}")
            result = {
                "is_correct": False,
                "score": 0,
                "max_score": 0,
                "comment": "解析大模型输出失败",
            }
        result["question"] = question
        result["student_answer"] = student_answer
        logger.info(f"批改完成:{result.get('is_correct')},得分:{result.get('score')}/{result.get('max_score')}")
        return result

    def batch_grade(self, questions: List[Dict]) -> List[Dict]:
        """批量批改多道题目"""
        results = []
        for item in questions:
            result = self.grade(
                question=item.get("question", ""),
                student_answer=item.get("student_answer", ""),
                reference_answer=item.get("reference_answer"),
            )
            results.append(result)
        return results

    def calculate_total_score(self, grading_results: List[Dict]) -> Tuple[float, float]:
        """计算总分和得分"""
        total_score = sum(item.get("max_score", 0) for item in grading_results)
        earned_score = sum(item.get("score", 0) for item in grading_results)
        return earned_score, total_score

    def generate_feedback(self, grading_result: List[Dict]) -> str:
        """根据批改结果生成评语"""
        correct_count = sum(1 for r in grading_result if r.get("is_correct", False))
        total_count = len(grading_result)
        total_earned = sum(r.get("score",0) for r in grading_result)
        total_max = sum(r.get("max_score",0) for r in grading_result)

        comments = [r.get("comment", "") for r in grading_result if not r.get("is_correct", False)]

        prompt =f"""共{total_count}道题，{correct_count}道题正确，得分{total_earned}/{total_max}
        错误题目评语:{'; '.join(comments)}
        请用一句话写出鼓励性总结评语.
        """
        response = self.client.chat.completions.create(
            model = self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature = 0.5,
            max_tokens = 100,
        )
        return response.choices[0].message.content.strip()