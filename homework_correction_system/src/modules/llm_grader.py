import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logger = logging.getLogger("homework_correction.llm_grader")

DEFAULT_GRADING_PROMPT = """你是一位专业的中小学教师，需要批改学生的作业题目。请仔细判断学生的答案是否正确，并给出详细的批改意见。

批改要求：
1. 判断学生答案是否正确（is_correct: true/false）
2. 给出得分（score，满分以题目标注为准，如无标注默认每题满分10分）
3. 给出简短评价（comment，指出对错及改进建议，不超过50字）
4. 如果答案不完整或部分正确，给出部分分数

请严格按以下JSON格式输出，不要输出其他内容：
{
    "is_correct": true/false,
    "score": 数字,
    "max_score": 数字,
    "comment": "评语"
}"""

GENERAL_FEEDBACK_PROMPT = """你是一位专业的中小学教师，需要根据学生作业的整体批改结果，生成一段综合评语。

根据以下批改结果生成评语：
- 总题数：{total_questions}
- 正确题数：{correct_count}
- 总得分：{earned_score}/{total_score}
- 各题详情：{details}

要求：
1. 先肯定学生的优点和进步
2. 指出需要改进的地方
3. 给出针对性的学习建议
4. 语言亲切、鼓励为主，适合中小学生
5. 评语长度控制在100-200字之间"""


class LLMGrader:
    def __init__(
            self,
            api_key: str,
            model: str = "gpt-4",
            api_base_url: str = "https://api.openai.com/v1",
            temperature: float = 0.3,
            max_tokens: int = 2048,
            request_timeout: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.grading_prompt = DEFAULT_GRADING_PROMPT

    def load_prompt(self, prompt_template: str) -> None:
        self.grading_prompt = prompt_template

    def build_evaluation_context(
            self,
            question: str,
            student_answer: str,
            reference_answer: Optional[str] = None,
    ) -> str:
        context_parts = [f"题目：{question}", f"学生答案：{student_answer}"]
        if reference_answer:
            context_parts.append(f"参考答案：{reference_answer}")
        return "\n".join(context_parts)

    def _call_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        url = f"{self.api_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_llm_response(content)
            except requests.exceptions.Timeout:
                last_error = Exception("LLM 请求超时")
                logger.warning(f"LLM 请求超时 (尝试 {attempt + 1}/3)")
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.warning(f"LLM 请求失败 (尝试 {attempt + 1}/3): {e}")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(f"LLM 响应解析失败 (尝试 {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(1.0 * (2 ** attempt))
        raise last_error

    @staticmethod
    def _parse_llm_response(content: str) -> Dict[str, Any]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:]) if len(lines) > 1 else content
            if content.endswith("```"):
                content = content[: -3]
            content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start: end + 1])
                except json.JSONDecodeError:
                    pass
            return {
                "is_correct": False,
                "score": 0,
                "max_score": 10,
                "comment": content[:100],
            }

    def grade(
            self,
            question: str,
            student_answer: str,
            reference_answer: Optional[str] = None,
            max_score: float = 10.0,
    ) -> Dict:
        if not student_answer or not student_answer.strip():
            return {
                "is_correct": False,
                "score": 0,
                "max_score": max_score,
                "comment": "未作答",
                "question": question,
                "student_answer": student_answer,
            }
        context = self.build_evaluation_context(
            question, student_answer, reference_answer
        )
        system_msg = (
            f"{self.grading_prompt}\n\n"
            f"本题满分：{max_score}分。请根据学生答案的正确程度给出相应分数。"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": context},
        ]
        try:
            result = self._call_llm(messages)
        except Exception as e:
            logger.error(f"批改失败: {e}")
            result = {
                "is_correct": False,
                "score": 0,
                "max_score": max_score,
                "comment": f"批改异常: {str(e)[:50]}",
            }
        result["question"] = question
        result["student_answer"] = student_answer
        result.setdefault("max_score", max_score)
        result.setdefault("score", 0)
        result.setdefault("is_correct", False)
        return result

    def batch_grade(
            self,
            questions: List[Dict],
            max_workers: int = 3,
    ) -> List[Dict]:
        results = [None] * len(questions)

        def grade_task(idx: int, item: Dict):
            try:
                results[idx] = self.grade(
                    question=item.get("question", item.get("question_text", "")),
                    student_answer=item.get("student_answer", item.get("answer_text", "")),
                    reference_answer=item.get("reference_answer"),
                    max_score=item.get("max_score", 10.0),
                )
            except Exception as e:
                logger.error(f"题目 {idx + 1} 批改异常: {e}")
                results[idx] = {
                    "is_correct": False,
                    "score": 0,
                    "max_score": item.get("max_score", 10.0),
                    "comment": f"批改失败: {str(e)[:50]}",
                    "question": item.get("question", item.get("question_text", "")),
                    "student_answer": item.get("student_answer", item.get("answer_text", "")),
                }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(grade_task, i, questions[i]): i
                for i in range(len(questions))
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"批改任务 {idx + 1} 异常: {e}")

        return [r for r in results if r is not None]

    def calculate_total_score(self, grading_results: List[Dict]) -> Tuple[float, float]:
        total_score = sum(item.get("max_score", 0) for item in grading_results)
        earned_score = sum(item.get("score", 0) for item in grading_results)
        return earned_score, total_score

    def generate_feedback(self, grading_results: List[Dict]) -> str:
        correct_count = sum(1 for r in grading_results if r.get("is_correct"))
        total_questions = len(grading_results)
        earned, total = self.calculate_total_score(grading_results)

        details = []
        for i, r in enumerate(grading_results):
            details.append(
                f"第{i + 1}题: {'正确' if r.get('is_correct') else '错误'} "
                f"({r.get('score', 0)}/{r.get('max_score', 0)}分) - {r.get('comment', '')}"
            )

        prompt = GENERAL_FEEDBACK_PROMPT.format(
            total_questions=total_questions,
            correct_count=correct_count,
            earned_score=earned,
            total_score=total,
            details="\n".join(details),
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请根据以上批改结果生成综合评语。"},
        ]
        try:
            result = self._call_llm(messages)
            return result.get("comment", "") or result.get("raw_data", "")
        except Exception as e:
            logger.error(f"生成评语失败: {e}")
            if total > 0:
                ratio = earned / total
                if ratio >= 0.9:
                    return "非常棒！继续保持！"
                elif ratio >= 0.75:
                    return "表现良好，还有提升空间，继续加油！"
                elif ratio >= 0.6:
                    return "需要加强练习，找出薄弱环节，争取更大进步！"
                else:
                    return "要更加努力学习哦，多复习巩固基础知识！"
            return "请继续努力！"

    def grade_pipeline(
            self,
            regions: List[Dict],
            max_workers: int = 3,
    ) -> Dict[str, Any]:
        logger.info(f"开始批改 {len(regions)} 道题目 (并发数: {max_workers})")
        grade_inputs = []
        for region in regions:
            ocr_result = region.get("ocr_result", {})
            question_text = region.get("question_text", "")
            answer_text = region.get("answer_text", "")
            if ocr_result:
                questions = ocr_result.get("questions", [])
                if questions:
                    q_info = questions[0].get("questionInfo", [])
                    a_info = questions[0].get("answerInfo", [])
                    question_text = " ".join(
                        item.get("content", "") for item in q_info
                    ) or question_text
                    answer_text = " ".join(
                        item.get("content", "") for item in a_info
                    ) or answer_text
            if not question_text:
                prism_words = ocr_result.get("prism_wordsInfo", []) if ocr_result else []
                if prism_words:
                    question_text = " ".join(w.get("word", "") for w in prism_words)
            grade_inputs.append({
                "question": question_text,
                "student_answer": answer_text or "",
                "max_score": region.get("max_score", 10.0),
            })

        grading_results = self.batch_grade(grade_inputs, max_workers=max_workers)
        earned_score, total_score = self.calculate_total_score(grading_results)
        feedback = self.generate_feedback(grading_results)

        logger.info(
            f"批改完成, 得分: {earned_score}/{total_score}, "
            f"正确率: {earned_score / total_score * 100:.1f}%"
            if total_score > 0 else "批改完成"
        )

        return {
            "grading_results": grading_results,
            "earned_score": earned_score,
            "total_score": total_score,
            "feedback": feedback,
        }