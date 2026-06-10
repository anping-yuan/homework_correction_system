"""
错题本模块
负责错误题目的持久化存储、查询检索、统计分析。
数据以 JSON 文件形式存储在 data/error_notebook.json。
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path


class ErrorNotebook:
    """错题本管理器 — 持久化存储错题并提供检索与统计"""

    def __init__(self, storage_path: str = "data/error_notebook.json"):
        self.storage_path = storage_path
        self._ensure_storage()

    # ===================== 内部工具 =====================

    def _ensure_storage(self) -> None:
        """确保存储文件及目录存在"""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.storage_path):
            self._save_data({"entries": []})

    def _load_data(self) -> dict:
        """读取整个存储文件"""
        with open(self.storage_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_data(self, data: dict) -> None:
        """写入整个存储文件"""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ===================== 增删改查 =====================

    def add_entry(self, question_data: dict) -> str:
        """添加一条错题记录，返回生成的 entry_id"""
        entry_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        entry = {
            "id": entry_id,
            "created_at": now,
            "question_no": question_data.get("question_no", 0),
            "question_text": question_data.get("question_text", ""),
            "student_answer": question_data.get("student_answer", ""),
            "subject": question_data.get("subject", "未分类"),
            "topic": question_data.get("topic", "未分类"),
            "difficulty": question_data.get("difficulty", "中等"),
            "score": question_data.get("score", 0),
            "max_score": question_data.get("max_score", 0),
            "is_correct": question_data.get("is_correct", False),
            "comment": question_data.get("comment", ""),
            "explanation": question_data.get("explanation", ""),
            "image_url": question_data.get("image_url", ""),
            "is_reviewed": False,
            "reviewed_at": None,
        }
        data = self._load_data()
        data["entries"].append(entry)
        self._save_data(data)
        return entry_id

    def get_all(
        self,
        subject: Optional[str] = None,
        reviewed: Optional[bool] = None,
        keyword: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """分页查询错题列表，支持按学科 / 复习状态 / 关键词筛选"""
        data = self._load_data()
        entries = data["entries"]

        if subject and subject != "全部":
            entries = [e for e in entries if e.get("subject") == subject]
        if reviewed is not None:
            entries = [e for e in entries if e.get("is_reviewed") == reviewed]
        if keyword:
            kw = keyword.lower()
            entries = [
                e for e in entries
                if kw in e.get("question_text", "").lower()
                or kw in e.get("topic", "").lower()
                or kw in e.get("comment", "").lower()
            ]

        # 按创建时间倒序排列
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)

        total = len(entries)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "entries": entries[start:end],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    def get_by_id(self, entry_id: str) -> Optional[dict]:
        """根据 ID 查找单条错题"""
        data = self._load_data()
        for entry in data["entries"]:
            if entry["id"] == entry_id:
                return entry
        return None

    def is_duplicate(self, question_no: int, subject: str, question_text: str) -> bool:
        """检查是否已存在相同的错题（同题号+同学科+相似文本）"""
        data = self._load_data()
        q_text_short = (question_text or "").strip()[:50]
        for entry in data["entries"]:
            if entry.get("question_no") == question_no and entry.get("subject") == subject:
                existing_text = (entry.get("question_text", "") or "").strip()[:50]
                if q_text_short and existing_text and q_text_short == existing_text:
                    return True
        return False

    def rename_entry(self, entry_id: str, new_text: str) -> bool:
        """重命名错题的题目名称"""
        if not new_text or not new_text.strip():
            return False
        data = self._load_data()
        for entry in data["entries"]:
            if entry["id"] == entry_id:
                entry["question_text"] = new_text.strip()
                self._save_data(data)
                return True
        return False

    def delete_entry(self, entry_id: str) -> bool:
        """删除单条错题"""
        data = self._load_data()
        original_len = len(data["entries"])
        data["entries"] = [e for e in data["entries"] if e["id"] != entry_id]
        if len(data["entries"]) < original_len:
            self._save_data(data)
            return True
        return False

    def batch_delete(self, entry_ids: List[str]) -> int:
        """批量删除错题，返回实际删除数量"""
        data = self._load_data()
        ids_set = set(entry_ids)
        original_len = len(data["entries"])
        data["entries"] = [e for e in data["entries"] if e["id"] not in ids_set]
        deleted = original_len - len(data["entries"])
        if deleted > 0:
            self._save_data(data)
        return deleted

    def mark_reviewed(self, entry_id: str) -> bool:
        """将某条错题标记为"已复习" """
        data = self._load_data()
        for entry in data["entries"]:
            if entry["id"] == entry_id:
                entry["is_reviewed"] = True
                entry["reviewed_at"] = datetime.now().isoformat()
                self._save_data(data)
                return True
        return False

    def mark_unreviewed(self, entry_id: str) -> bool:
        """将某条错题标记为"未复习" """
        data = self._load_data()
        for entry in data["entries"]:
            if entry["id"] == entry_id:
                entry["is_reviewed"] = False
                entry["reviewed_at"] = None
                self._save_data(data)
                return True
        return False

    def mark_all_reviewed(self) -> int:
        """将所有错题标记为已复习，返回更新的数量"""
        data = self._load_data()
        now = datetime.now().isoformat()
        count = 0
        for entry in data["entries"]:
            if not entry.get("is_reviewed"):
                entry["is_reviewed"] = True
                entry["reviewed_at"] = now
                count += 1
        if count > 0:
            self._save_data(data)
        return count

    # ===================== 统计分析 =====================

    def get_stats(self) -> dict:
        """生成错题本的汇总统计数据"""
        data = self._load_data()
        entries = data["entries"]
        total = len(entries)

        if total == 0:
            return {
                "total": 0,
                "reviewed": 0,
                "unreviewed": 0,
                "review_rate": 0,
                "by_subject": {},
                "by_topic": {},
                "by_difficulty": {},
                "recent_week": 0,
                "recent_month": 0,
            }

        reviewed = sum(1 for e in entries if e.get("is_reviewed"))

        # 按学科统计
        subjects: dict = {}
        for e in entries:
            subj = e.get("subject", "未分类")
            if subj not in subjects:
                subjects[subj] = {"total": 0, "reviewed": 0}
            subjects[subj]["total"] += 1
            if e.get("is_reviewed"):
                subjects[subj]["reviewed"] += 1

        # 按知识点统计（取 TOP 20）
        topics: dict = {}
        for e in entries:
            topic = e.get("topic", "未分类")
            topics[topic] = topics.get(topic, 0) + 1
        topics = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:20])

        # 按难度统计
        difficulties: dict = {}
        for e in entries:
            diff = e.get("difficulty", "中等")
            difficulties[diff] = difficulties.get(diff, 0) + 1

        # 最近 7 天 / 30 天
        now = datetime.now()
        week_ago = (now - timedelta(days=7)).isoformat()
        month_ago = (now - timedelta(days=30)).isoformat()
        recent_week = sum(1 for e in entries if e.get("created_at", "") >= week_ago)
        recent_month = sum(1 for e in entries if e.get("created_at", "") >= month_ago)

        return {
            "total": total,
            "reviewed": reviewed,
            "unreviewed": total - reviewed,
            "review_rate": round(reviewed / total * 100, 1),
            "by_subject": subjects,
            "by_topic": topics,
            "by_difficulty": difficulties,
            "recent_week": recent_week,
            "recent_month": recent_month,
        }

    def get_subjects(self) -> List[str]:
        """获取所有出现过的学科列表（供前端下拉筛选用）"""
        data = self._load_data()
        subjects = set()
        for e in data["entries"]:
            subj = e.get("subject", "未分类")
            subjects.add(subj)
        return sorted(subjects)

    def clear_all(self) -> int:
        """清空所有错题，返回被删除的数量"""
        data = self._load_data()
        count = len(data["entries"])
        data["entries"] = []
        self._save_data(data)
        return count
