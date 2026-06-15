"""
数据库管理模块
使用 PyMySQL + DBUtils 连接池管理 MySQL，执行 grading_records 表的 CRUD 操作。
替代原有的 JSON 文件存储和内存 dict 存储。
"""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB  # type: ignore[import-untyped]

logger = logging.getLogger("db_manager")


def _load_db_config() -> dict:
    """从 config.json 加载 MySQL 连接配置"""
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            return cfg.get("mysql", {})
    return {}


_db_config = _load_db_config()

DB_HOST = _db_config.get("host", "localhost")
DB_PORT = _db_config.get("port", 3308)
DB_USER = _db_config.get("user", "root")
DB_PASSWORD = _db_config.get("password", "123456")
DB_NAME = _db_config.get("database", "homework")
DB_CHARSET = _db_config.get("charset", "utf8mb4")

# 连接池：默认 min=2, max=10 个连接
_pool: Optional[PooledDB] = None
_pool_error: Optional[str] = None


def _init_pool():
    """惰性初始化连接池（失败时记录错误，不阻断应用启动）"""
    global _pool, _pool_error
    if _pool is not None:
        return
    try:
        _pool = PooledDB(
            creator=pymysql,
            maxconnections=10,
            mincached=2,
            maxcached=5,
            blocking=True,
            maxusage=1000,
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset=DB_CHARSET,
            cursorclass=DictCursor,
            autocommit=False,
        )
        # 测试连接
        test_conn = _pool.connection()
        test_conn.close()
        logger.info(f"MySQL 连接池初始化成功 (host={DB_HOST}:{DB_PORT}, db={DB_NAME})")
    except Exception as e:
        _pool_error = str(e)
        _pool = None
        logger.warning(f"MySQL 连接池初始化失败，服务将以降级模式运行: {e}")


def _is_pool_available() -> bool:
    """检查连接池是否可用"""
    if _pool is None:
        _init_pool()
    return _pool is not None


@contextmanager
def get_connection():
    """从连接池获取数据库连接（上下文管理器，自动归还）"""
    if not _is_pool_available() or _pool is None:
        raise RuntimeError(f"数据库不可用: {_pool_error or '连接池未初始化'}")
    conn = _pool.connection()
    try:
        # 自动重连检测
        conn.ping(reconnect=True)
        yield conn
    finally:
        conn.close()


class DBManager:
    """批改记录数据库管理器"""

    # ===================== 写入 =====================

    @staticmethod
    def insert_record(data: dict) -> int:
        """插入一条批改记录，返回自增 ID"""
        sql = """
            INSERT INTO grading_records (
                upload_id, task_id, question_no, page,
                question_text, student_answer,
                is_correct, score, max_score, comment, explanation,
                subject, topic, difficulty,
                sub_questions,
                original_image, annotated_image, crop_image,
                region_x, region_y, region_width, region_height,
                grading_mode, image_hash,
                in_notebook, is_reviewed
            ) VALUES (
                %(upload_id)s, %(task_id)s, %(question_no)s, %(page)s,
                %(question_text)s, %(student_answer)s,
                %(is_correct)s, %(score)s, %(max_score)s, %(comment)s, %(explanation)s,
                %(subject)s, %(topic)s, %(difficulty)s,
                %(sub_questions)s,
                %(original_image)s, %(annotated_image)s, %(crop_image)s,
                %(region_x)s, %(region_y)s, %(region_width)s, %(region_height)s,
                %(grading_mode)s, %(image_hash)s,
                %(in_notebook)s, %(is_reviewed)s
            )
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, data)
                conn.commit()
                return cur.lastrowid

    @staticmethod
    def insert_batch(records: List[dict]) -> int:
        """批量插入批改记录，返回插入条数"""
        if not records:
            return 0

        sql = """
            INSERT INTO grading_records (
                upload_id, task_id, question_no, page,
                question_text, student_answer,
                is_correct, score, max_score, comment, explanation,
                subject, topic, difficulty,
                sub_questions,
                original_image, annotated_image, crop_image,
                region_x, region_y, region_width, region_height,
                grading_mode, image_hash,
                in_notebook, is_reviewed
            ) VALUES (
                %(upload_id)s, %(task_id)s, %(question_no)s, %(page)s,
                %(question_text)s, %(student_answer)s,
                %(is_correct)s, %(score)s, %(max_score)s, %(comment)s, %(explanation)s,
                %(subject)s, %(topic)s, %(difficulty)s,
                %(sub_questions)s,
                %(original_image)s, %(annotated_image)s, %(crop_image)s,
                %(region_x)s, %(region_y)s, %(region_width)s, %(region_height)s,
                %(grading_mode)s, %(image_hash)s,
                %(in_notebook)s, %(is_reviewed)s
            )
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.executemany(sql, records)
                conn.commit()
                return affected

    # ===================== 错题本标记 =====================

    @staticmethod
    def mark_as_notebook(record_id: int) -> bool:
        """将一条记录标记为错题本"""
        sql = """
            UPDATE grading_records
            SET in_notebook=1, notebook_saved_at=NOW()
            WHERE id=%(id)s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, {"id": record_id})
                conn.commit()
                return affected > 0

    @staticmethod
    def mark_as_notebook_by_task(task_id: str, question_indices: List[int]) -> int:
        """将某次批改任务中的指定题目标记为错题本，返回标记数量"""
        if not question_indices:
            return 0
        placeholders = ",".join(["%s"] * len(question_indices))
        sql = f"""
            UPDATE grading_records
            SET in_notebook=1, notebook_saved_at=NOW()
            WHERE task_id=%(task_id)s AND question_no IN ({placeholders})
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, [task_id] + question_indices)
                conn.commit()
                return affected

    @staticmethod
    def mark_reviewed(record_id: int) -> bool:
        """标记为已复习"""
        sql = """
            UPDATE grading_records
            SET is_reviewed=1, reviewed_at=NOW()
            WHERE id=%(id)s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, {"id": record_id})
                conn.commit()
                return affected > 0

    @staticmethod
    def mark_unreviewed(record_id: int) -> bool:
        """标记为未复习"""
        sql = """
            UPDATE grading_records
            SET is_reviewed=0, reviewed_at=NULL
            WHERE id=%(id)s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, {"id": record_id})
                conn.commit()
                return affected > 0

    @staticmethod
    def mark_all_reviewed() -> int:
        """将所有错题本记录标记为已复习"""
        sql = """
            UPDATE grading_records
            SET is_reviewed=1, reviewed_at=NOW()
            WHERE in_notebook=1 AND is_reviewed=0
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql)
                conn.commit()
                return affected

    # ===================== 更新字段 =====================

    @staticmethod
    def update_topic(record_id: int, new_topic: str) -> bool:
        """更新知识点"""
        sql = "UPDATE grading_records SET topic=%(topic)s WHERE id=%(id)s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, {"id": record_id, "topic": new_topic})
                conn.commit()
                return affected > 0

    @staticmethod
    def update_subject(record_id: int, new_subject: str) -> bool:
        """更新学科"""
        sql = "UPDATE grading_records SET subject=%(subject)s WHERE id=%(id)s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, {"id": record_id, "subject": new_subject})
                conn.commit()
                return affected > 0

    @staticmethod
    def rename_entry(record_id: int, new_text: str) -> bool:
        """重命名错题（修改题目文本）"""
        sql = "UPDATE grading_records SET question_text=%(text)s WHERE id=%(id)s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, {"id": record_id, "text": new_text})
                conn.commit()
                return affected > 0

    @staticmethod
    def update_grading_result(record_id: int, data: dict) -> bool:
        """更新一条记录的批改结果字段（score, is_correct, comment, sub_questions 等）

        用于重做批改后覆盖旧结果。只更新 data 中提供的字段，不会把未提供的字段置空。
        """
        import json as _json

        allowed = {'is_correct', 'score', 'max_score', 'comment',
                   'explanation', 'sub_questions', 'student_answer',
                   'annotated_image', 'crop_image'}
        update_data = {}
        for k, v in data.items():
            if k not in allowed:
                continue
            if k == 'sub_questions' and isinstance(v, (list, dict)):
                update_data[k] = _json.dumps(v, ensure_ascii=False)
            elif k == 'is_correct':
                update_data[k] = 1 if v else 0
            else:
                update_data[k] = v

        if not update_data:
            return False

        set_parts = [f"{k}=%({k})s" for k in update_data]
        sql = f"UPDATE grading_records SET {', '.join(set_parts)} WHERE id=%(id)s"
        update_data['id'] = record_id

        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, update_data)
                conn.commit()
                return affected > 0

    # ===================== 删除 =====================

    @staticmethod
    def delete_record(record_id: int) -> bool:
        """删除单条记录"""
        sql = "DELETE FROM grading_records WHERE id=%(id)s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, {"id": record_id})
                conn.commit()
                return affected > 0

    @staticmethod
    def batch_delete(record_ids: List[int]) -> int:
        """批量删除"""
        if not record_ids:
            return 0
        placeholders = ",".join(["%s"] * len(record_ids))
        sql = f"DELETE FROM grading_records WHERE id IN ({placeholders})"
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql, record_ids)
                conn.commit()
                return affected

    @staticmethod
    def clear_all_notebook() -> int:
        """清空错题本（只删 in_notebook=1 的记录）"""
        sql = "DELETE FROM grading_records WHERE in_notebook=1"
        with get_connection() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(sql)
                conn.commit()
                return affected

    # ===================== 查询 =====================

    @staticmethod
    def get_by_id(record_id: int) -> Optional[dict]:
        """按主键查询"""
        sql = "SELECT * FROM grading_records WHERE id=%(id)s"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"id": record_id})
                return cur.fetchone()

    @staticmethod
    def get_by_upload(upload_id: str) -> List[dict]:
        """查询某次上传的所有记录"""
        sql = """
            SELECT * FROM grading_records
            WHERE upload_id=%(upload_id)s
            ORDER BY question_no
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"upload_id": upload_id})
                return cur.fetchall()

    @staticmethod
    def get_by_task(task_id: str) -> List[dict]:
        """查询某次批改任务的所有记录"""
        sql = """
            SELECT * FROM grading_records
            WHERE task_id=%(task_id)s
            ORDER BY question_no
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"task_id": task_id})
                return cur.fetchall()

    @staticmethod
    def get_notebook_entries(
        subject: Optional[str] = None,
        reviewed: Optional[bool] = None,
        keyword: Optional[str] = None,
        difficulty: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        """分页查询错题本，支持筛选"""
        conditions = ["in_notebook=1"]
        params: Dict[str, Any] = {}

        if subject and subject != "全部":
            conditions.append("subject=%(subject)s")
            params["subject"] = subject
        if reviewed is not None:
            conditions.append("is_reviewed=%(reviewed)s")
            params["reviewed"] = 1 if reviewed else 0
        if difficulty:
            conditions.append("difficulty=%(difficulty)s")
            params["difficulty"] = difficulty
        if keyword:
            kw = f"%{keyword}%"
            conditions.append(
                "(question_text LIKE %(kw1)s OR topic LIKE %(kw2)s OR comment LIKE %(kw3)s)"
            )
            params["kw1"] = kw
            params["kw2"] = kw
            params["kw3"] = kw

        where = " AND ".join(conditions)

        # 总数
        count_sql = f"SELECT COUNT(*) AS cnt FROM grading_records WHERE {where}"
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(count_sql, params)
                total = cur.fetchone()["cnt"]

        # 分页数据
        offset = (page - 1) * page_size
        data_sql = f"""
            SELECT * FROM grading_records
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT %(limit)s OFFSET %(offset)s
        """
        params["limit"] = page_size
        params["offset"] = offset
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(data_sql, params)
                entries = cur.fetchall()

        for e in entries:
            e["created_at"] = str(e["created_at"]) if e.get("created_at") else ""
            e["updated_at"] = str(e["updated_at"]) if e.get("updated_at") else ""
            e["notebook_saved_at"] = str(e["notebook_saved_at"]) if e.get("notebook_saved_at") else ""
            e["reviewed_at"] = str(e["reviewed_at"]) if e.get("reviewed_at") else ""
            # 将 TINYINT 转为 Python bool
            e["is_correct"] = bool(e.get("is_correct", 0))
            e["in_notebook"] = bool(e.get("in_notebook", 0))
            e["is_reviewed"] = bool(e.get("is_reviewed", 0))
            # 将 Decimal 转为 float
            if "score" in e and e["score"] is not None:
                e["score"] = float(e["score"])
            if "max_score" in e and e["max_score"] is not None:
                e["max_score"] = float(e["max_score"])

        return {
            "entries": entries,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    @staticmethod
    def is_duplicate(question_no: int, subject: str, question_text: str) -> bool:
        """检查是否已存在相同的错题（同题号+同学科+相似文本）"""
        q_text_short = (question_text or "").strip()[:50]
        if not q_text_short:
            return False

        sql = """
            SELECT id FROM grading_records
            WHERE in_notebook=1 AND question_no=%(qno)s AND subject=%(subj)s
            AND question_text LIKE %(txt)s
            LIMIT 1
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "qno": question_no,
                    "subj": subject,
                    "txt": q_text_short + "%",
                })
                return cur.fetchone() is not None

    # ===================== 统计 =====================

    @staticmethod
    def get_stats() -> dict:
        """错题本统计数据"""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 总数 & 复习数
                cur.execute("""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN is_reviewed=1 THEN 1 ELSE 0 END) AS reviewed
                    FROM grading_records WHERE in_notebook=1
                """)
                row = cur.fetchone()
                total = row["total"] or 0
                reviewed = row["reviewed"] or 0
                unreviewed = total - reviewed
                review_rate = round(reviewed / total * 100, 1) if total > 0 else 0

                # 按学科
                cur.execute("""
                    SELECT subject, COUNT(*) AS cnt,
                           SUM(CASE WHEN is_reviewed=1 THEN 1 ELSE 0 END) AS rv
                    FROM grading_records WHERE in_notebook=1
                    GROUP BY subject
                """)
                by_subject = {}
                for r in cur.fetchall():
                    by_subject[r["subject"]] = {"total": r["cnt"], "reviewed": r["rv"]}

                # 按知识点 TOP 20
                cur.execute("""
                    SELECT topic, COUNT(*) AS cnt
                    FROM grading_records WHERE in_notebook=1
                    GROUP BY topic ORDER BY cnt DESC LIMIT 20
                """)
                by_topic = {r["topic"]: r["cnt"] for r in cur.fetchall()}

                # 按难度
                cur.execute("""
                    SELECT difficulty, COUNT(*) AS cnt
                    FROM grading_records WHERE in_notebook=1
                    GROUP BY difficulty
                """)
                by_difficulty = {r["difficulty"]: r["cnt"] for r in cur.fetchall()}

                # 最近 7/30 天
                cur.execute("""
                    SELECT
                        SUM(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 ELSE 0 END) AS recent_week,
                        SUM(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 ELSE 0 END) AS recent_month
                    FROM grading_records WHERE in_notebook=1
                """)
                recent = cur.fetchone()

        return {
            "total": total,
            "reviewed": reviewed,
            "unreviewed": unreviewed,
            "review_rate": review_rate,
            "by_subject": by_subject,
            "by_topic": by_topic,
            "by_difficulty": by_difficulty,
            "recent_week": recent["recent_week"] or 0,
            "recent_month": recent["recent_month"] or 0,
        }

    @staticmethod
    def get_subjects() -> List[str]:
        """获取所有出现过的学科"""
        sql = """
            SELECT DISTINCT subject FROM grading_records
            WHERE in_notebook=1 ORDER BY subject
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return [r["subject"] for r in cur.fetchall()]

    # ===================== 图片哈希缓存 =====================

    @staticmethod
    def get_by_image_hash(image_hash: str, grading_mode: str = "vision") -> Optional[dict]:
        """按图片哈希查找最近的批改结果（缓存命中）"""
        sql = """
            SELECT * FROM grading_records
            WHERE image_hash=%(hash)s AND grading_mode=%(mode)s
            ORDER BY created_at DESC LIMIT 1
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"hash": image_hash, "mode": grading_mode})
                row = cur.fetchone()
                if row:
                    row["created_at"] = str(row["created_at"]) if row.get("created_at") else ""
                    row["is_correct"] = bool(row.get("is_correct", 0))
                    if row.get("score") is not None:
                        row["score"] = float(row["score"])
                    if row.get("max_score") is not None:
                        row["max_score"] = float(row["max_score"])
                return row

    # ===================== 上传信息 =====================

    @staticmethod
    def save_upload_info(upload_id: str, page_count: int, page_images: List[str]) -> None:
        """
        保存上传会话信息（可选，用于追溯）。
        实际题目数据在批改时才会 INSERT。
        这里只做轻量记录 — 不插入 grading_records，仅靠后续批量插入。
        """
        # 上传信息目前只用于 grade-selected 中查找原图，
        # 改为从 grading_records 中按 upload_id 查已有记录的图片路径即可。
        # 这里保留方法体作为扩展点。
        pass

    @staticmethod
    def get_upload_images(upload_id: str) -> Dict[str, Any]:
        """获取某个上传的图片路径信息（用于批改时定位图片）"""
        sql = """
            SELECT DISTINCT original_image, page
            FROM grading_records
            WHERE upload_id=%(uid)s AND original_image IS NOT NULL
            ORDER BY page
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"uid": upload_id})
                rows = cur.fetchall()
                return {
                    "page_images": [r["original_image"] for r in rows],
                    "pages": len(rows),
                }

    # ===================== 基准评测 =====================

    @staticmethod
    def save_benchmark_run(data: dict) -> int:
        """保存一次评测结果"""
        sql = """
            INSERT INTO benchmark_runs (
                sample_count, mode, accuracy, total_valid,
                false_positives, false_negatives,
                recall_rate, precision_rate, avg_time_sec,
                concurrency, note
            ) VALUES (
                %(sample_count)s, %(mode)s, %(accuracy)s, %(total_valid)s,
                %(false_positives)s, %(false_negatives)s,
                %(recall_rate)s, %(precision_rate)s, %(avg_time_sec)s,
                %(concurrency)s, %(note)s
            )
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, data)
                conn.commit()
                return cur.lastrowid

    @staticmethod
    def get_benchmark_history(limit: int = 20) -> List[dict]:
        """查询最近的评测记录"""
        sql = """
            SELECT * FROM benchmark_runs
            ORDER BY run_time DESC LIMIT %(limit)s
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"limit": limit})
                return cur.fetchall()
