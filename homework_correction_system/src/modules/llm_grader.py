"""
大模型批改模块
负责调用大语言模型对作业内容进行智能批改和评分。
"""

import json
from typing import Dict, List, Optional, Tuple


class LLMGrader:
    """大模型批改器"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.grading_prompt = ""

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
        """对单道题目进行批改"""
        pass

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

    def generate_feedback(self, grading_result: Dict) -> str:
        """根据批改结果生成评语"""
        pass