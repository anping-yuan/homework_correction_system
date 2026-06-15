"""
错题本模块
负责错误题目的持久化存储、查询检索、统计分析。
数据存储在 MySQL grading_records 表中（通过 in_notebook 标记区分）。
"""

from typing import Dict, List, Optional

from src.modules.db_manager import DBManager


class ErrorNotebook:
    """错题本管理器 — 基于 MySQL 的持久化存储"""

    def __init__(self, storage_path: str = None):
        """
        storage_path 参数保留用于兼容旧代码，不再使用。
        数据现在存储在 MySQL grading_records 表中。
        """
        self.db = DBManager()

    # ===================== 增删改查 =====================

    def add_entry(self, question_data: dict) -> str:
        """添加一条错题记录，返回自增 ID（字符串）"""
        import json as _json

        sub_qs = question_data.get("sub_questions", [])
        sub_json = _json.dumps(sub_qs, ensure_ascii=False) if sub_qs else None

        record = {
            "upload_id": question_data.get("upload_id", ""),
            "task_id": question_data.get("task_id", ""),
            "question_no": question_data.get("question_no", 0),
            "page": question_data.get("page", 1),
            "question_text": question_data.get("question_text", ""),
            "student_answer": question_data.get("student_answer", ""),
            "is_correct": 1 if question_data.get("is_correct") else 0,
            "score": question_data.get("score", 0),
            "max_score": question_data.get("max_score", 0),
            "comment": question_data.get("comment", ""),
            "explanation": question_data.get("explanation", ""),
            "subject": question_data.get("subject", "未分类"),
            "topic": question_data.get("topic", "未分类"),
            "difficulty": question_data.get("difficulty", "中等"),
            "sub_questions": sub_json,
            "original_image": question_data.get("image_url", ""),
            "annotated_image": None,
            "crop_image": None,
            "region_x": question_data.get("region_x"),
            "region_y": question_data.get("region_y"),
            "region_width": question_data.get("region_width"),
            "region_height": question_data.get("region_height"),
            "grading_mode": question_data.get("grading_mode", "vision"),
            "in_notebook": 1,
            "is_reviewed": 0,
        }
        new_id = self.db.insert_record(record)
        return str(new_id)

    def get_all(
        self,
        subject: Optional[str] = None,
        reviewed: Optional[bool] = None,
        keyword: Optional[str] = None,
        difficulty: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """分页查询错题列表"""
        result = self.db.get_notebook_entries(
            subject=subject,
            reviewed=reviewed,
            keyword=keyword,
            difficulty=difficulty,
            page=page,
            page_size=page_size,
        )
        # 转换字段名兼容旧接口
        for e in result["entries"]:
            e["id"] = str(e.pop("id", ""))
            e["created_at"] = str(e.get("created_at", ""))
            e["image_url"] = e.get("original_image", "")
        return result

    def get_by_id(self, entry_id: str) -> Optional[dict]:
        """根据 ID 查找单条错题"""
        try:
            rid = int(entry_id)
        except (ValueError, TypeError):
            return None
        row = self.db.get_by_id(rid)
        if row is None:
            return None
        # 转换字段名兼容旧接口
        row["id"] = str(row.pop("id", ""))
        row["created_at"] = str(row.get("created_at", ""))
        row["image_url"] = row.get("original_image", "")
        return row

    def is_duplicate(self, question_no: int, subject: str, question_text: str) -> bool:
        """检查是否已存在相同错题"""
        return self.db.is_duplicate(question_no, subject, question_text)

    def rename_entry(self, entry_id: str, new_text: str) -> bool:
        """重命名错题"""
        try:
            return self.db.rename_entry(int(entry_id), new_text)
        except (ValueError, TypeError):
            return False

    def update_topic(self, entry_id: str, new_topic: str) -> bool:
        """更新知识点"""
        try:
            return self.db.update_topic(int(entry_id), new_topic)
        except (ValueError, TypeError):
            return False

    def update_subject(self, entry_id: str, new_subject: str) -> bool:
        """更新学科"""
        try:
            return self.db.update_subject(int(entry_id), new_subject)
        except (ValueError, TypeError):
            return False

    def delete_entry(self, entry_id: str) -> bool:
        """删除单条"""
        try:
            return self.db.delete_record(int(entry_id))
        except (ValueError, TypeError):
            return False

    def batch_delete(self, entry_ids: List[str]) -> int:
        """批量删除"""
        int_ids = []
        for eid in entry_ids:
            try:
                int_ids.append(int(eid))
            except (ValueError, TypeError):
                pass
        return self.db.batch_delete(int_ids)

    def mark_reviewed(self, entry_id: str) -> bool:
        """标记为已复习"""
        try:
            return self.db.mark_reviewed(int(entry_id))
        except (ValueError, TypeError):
            return False

    def mark_unreviewed(self, entry_id: str) -> bool:
        """标记为未复习"""
        try:
            return self.db.mark_unreviewed(int(entry_id))
        except (ValueError, TypeError):
            return False

    def mark_all_reviewed(self) -> int:
        """全部标记为已复习"""
        return self.db.mark_all_reviewed()

    # ===================== 统计分析 =====================

    def get_stats(self) -> dict:
        """错题本统计数据"""
        return self.db.get_stats()

    def get_subjects(self) -> List[str]:
        """获取所有学科列表"""
        return self.db.get_subjects()

    def clear_all(self) -> int:
        """清空所有错题"""
        return self.db.clear_all_notebook()
