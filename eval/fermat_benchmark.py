"""
FERMAT 基准评测脚本
================================
用 FERMAT 手写数学数据集评测作业批改系统的两种模式（Vision / OCR+LLM）。

用法：
    python eval/fermat_benchmark.py                        # 默认 100 条，两种模式都跑
    python eval/fermat_benchmark.py --n 50                 # 只跑 50 条
    python eval/fermat_benchmark.py --n 200 --mode vision  # 只跑 Vision 模式
    python eval/fermat_benchmark.py --workers 15           # 并发 15

输出：
    eval/fermat_checkpoint.json   ← 断点续跑（中断后自动跳过已完成的）
    eval/fermat_report.json       ← 详细结果
    eval/fermat_summary.txt       ← 可读报告
"""

import argparse
import json
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
import cv2
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fermat_benchmark")

# 静默所有第三方库的日志输出
for lib in ["openai", "urllib3", "httpx", "httpcore", "src.modules", "llm_grader",
            "aliyun_ocr", "db_manager", "api", "__main__"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger(lib).propagate = False

# 只让评测脚本自己的日志可见
logging.getLogger("fermat_benchmark").setLevel(logging.INFO)


# ===================== 配置 =====================

# ── 可调参数（改这里就行）──
SAMPLE_COUNT = 100        # 评测多少张图片
GRADING_MODE = "both"     # "vision" / "ocr_llm" / "vision_deepseek" / "both"
CONCURRENCY  = 8          # 并发数
RESUME       = True       # 断点续跑

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
PARQUET_DIR = Path("C:/Users/anpingyuan/Desktop/ceshiphoto/FERMAT_dataset/data")

checkpoint_file = EVAL_DIR / "fermat_checkpoint.json"
report_json = EVAL_DIR / "fermat_report.json"
report_txt = EVAL_DIR / "fermat_summary.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================== 数据加载 =====================

def load_fermat_data(n: int, seed: int = 42) -> List[Dict]:
    """从 parquet 文件加载前 N 条 FERMAT 数据，随机打乱保证样本多样性"""
    logger.info(f"加载 FERMAT 数据（目标 {n} 条）...")

    parquet_files = sorted(PARQUET_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {PARQUET_DIR}")

    all_rows = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        all_rows.extend(df.to_dict("records"))

    logger.info(f"共 {len(all_rows)} 条记录（{len(parquet_files)} 个 parquet）")

    # 随机打乱保证样本多样性
    import random
    random.seed(seed)
    random.shuffle(all_rows)

    return all_rows[:n]


def extract_image(row: Dict, index: int) -> Optional[str]:
    """从 parquet 行提取图片，保存为临时文件，返回路径"""
    img_bytes = row.get("image", {}).get("bytes")
    if img_bytes is None:
        return None

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None

    save_dir = OUTPUT_DIR / "fermat_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / f"fermat_{index:04d}.jpg")
    cv2.imwrite(save_path, img)
    return save_path


# ===================== 批改适配 =====================

# 全局单例，避免每次初始化打日志
_grader = None
_ocr = None

def _get_grader():
    global _grader
    if _grader is None:
        from src.modules.llm_grader import LLMGrader
        _grader = LLMGrader()
    return _grader

def _get_ocr():
    global _ocr
    if _ocr is None:
        from src.modules.aliyun_ocr import AliyunOCR
        _ocr = AliyunOCR()
    return _ocr

def grade_vision(image_path: str) -> Dict:
    """Vision 模式：整张图发 Qwen-VL 直接批改"""
    grader = _get_grader()
    try:
        result = grader.grade_full_page(image_path)
        return {
            "is_correct": result.get("is_correct", False),
            "score": result.get("score", 0),
            "max_score": result.get("max_score", 0),
            "comment": result.get("comment", "")[:300],
            "subject": result.get("subject", ""),
            "sub_count": len(result.get("sub_questions", [])),
            "raw_ocr_text": result.get("ocr_text", "")[:500],
        }
    except Exception as e:
        logger.error(f"Vision 批改失败 [{image_path}]: {e}")
        return {"error": str(e), "is_correct": None}


def grade_ocr_llm(image_path: str) -> Dict:
    """OCR+LLM 模式：阿里云 OCR 提取文字 → DeepSeek 文本批改"""
    ocr = _get_ocr()
    grader = _get_grader()

    try:
        # Step 1: OCR 提取文字
        ocr_result = ocr.recognize_text(image_path)
        ocr_text = ocr.parse_result(ocr_result)

        if not ocr_text or len(ocr_text.strip()) < 5:
            logger.warning(f"OCR 文字过短 [{image_path}]: {ocr_text[:100]}")
            return {"error": "OCR 文字过短或为空", "is_correct": None, "ocr_text": ocr_text[:200]}

        # Step 2: 文本模型批改
        grade_result = grader.grade(ocr_text=ocr_text, reference_answer="")

        return {
            "is_correct": grade_result.get("is_correct", False),
            "score": grade_result.get("score", 0),
            "max_score": grade_result.get("max_score", 0),
            "comment": grade_result.get("comment", "")[:300],
            "subject": grade_result.get("subject", ""),
            "sub_count": len(grade_result.get("sub_questions", [])),
            "ocr_text": ocr_text[:500],
        }
    except Exception as e:
        logger.error(f"OCR+LLM 批改失败 [{image_path}]: {e}")
        return {"error": str(e), "is_correct": None}


def grade_vision_deepseek(image_path: str) -> Dict:
    """Vision+DeepSeek 模式：Vision 看图提取 → DeepSeek 三路投票判题"""
    grader = _get_grader()
    try:
        result = grader.grade_vision_deepseek(image_path)
        return {
            "is_correct": result.get("is_correct", False),
            "score": result.get("score", 0),
            "max_score": result.get("max_score", 0),
            "comment": result.get("comment", "")[:300],
            "subject": result.get("subject", ""),
            "sub_count": len(result.get("sub_questions", [])),
            "raw_ocr_text": result.get("ocr_text", "")[:500],
        }
    except Exception as e:
        logger.error(f"Vision+DeepSeek 批改失败 [{image_path}]: {e}")
        return {"error": str(e), "is_correct": None}


# ===================== 评测逻辑 =====================

def evaluate_single(index: int, row: Dict, mode: str, checkpoint: Dict) -> Optional[Dict]:
    """评测单条 FERMAT 数据，返回结果记录。已完成的跳过。"""
    # 断点续跑检查
    key = f"{index}_{mode}"
    if key in checkpoint:
        return None  # 已跑过，跳过

    ground_truth = row["has_error"]  # True = 有错, False = 没错
    grade = row.get("grade", "unknown")
    domain = row.get("domain_code", "unknown")

    # 提取图片
    img_path = extract_image(row, index)
    if img_path is None:
        record = {
            "index": index, "mode": mode, "grade": grade, "domain": domain,
            "ground_truth_has_error": ground_truth,
            "error": "图片提取失败",
            "model_is_correct": None, "match": None,
        }
        checkpoint[key] = record
        return record

    # 调用批改
    t0 = time.time()
    if mode == "vision":
        grading = grade_vision(img_path)
    elif mode == "vision_deepseek":
        grading = grade_vision_deepseek(img_path)
    else:
        grading = grade_ocr_llm(img_path)

    elapsed = round(time.time() - t0, 1)

    model_is_correct = grading.get("is_correct")

    # 匹配判断：
    # ground_truth has_error=True   → 模型应该判 is_correct=False  → match: model判错
    # ground_truth has_error=False  → 模型应该判 is_correct=True   → match: model判对
    if model_is_correct is None:
        match = None  # 批改失败
    else:
        match = (model_is_correct == (not ground_truth))

    record = {
        "index": index,
        "mode": mode,
        "grade": grade,
        "domain": str(domain),
        "ground_truth_has_error": ground_truth,
        "model_is_correct": model_is_correct,
        "match": match,
        "model_score": grading.get("score"),
        "model_max_score": grading.get("max_score"),
        "model_subject": grading.get("subject", ""),
        "model_sub_count": grading.get("sub_count", 0),
        "model_comment": grading.get("comment", "")[:200],
        "ocr_text": grading.get("ocr_text", "")[:200],
        "error": grading.get("error"),
        "elapsed_sec": elapsed,
        "image_path": img_path,
    }
    checkpoint[key] = record
    return record


# ===================== 统计报告 =====================

def compute_metrics(records: List[Dict]) -> Dict:
    """计算准确率等指标"""
    valid = [r for r in records if r.get("match") is not None]
    if not valid:
        return {"total_valid": 0}

    total = len(valid)
    correct = sum(1 for r in valid if r["match"])
    accuracy = round(correct / total * 100, 1)

    # 按 has_error 分组
    has_err = [r for r in valid if r["ground_truth_has_error"]]
    no_err = [r for r in valid if not r["ground_truth_has_error"]]

    # False Positive: 模型说有错但实际没错 (has_error=False, is_correct=False)
    fp = [r for r in valid if r["ground_truth_has_error"] is False and r["model_is_correct"] is False]
    # False Negative: 模型说没错但实际有错 (has_error=True, is_correct=True)
    fn = [r for r in valid if r["ground_truth_has_error"] is True and r["model_is_correct"] is True]

    recall = round(sum(1 for r in has_err if r["match"]) / len(has_err) * 100, 1) if has_err else 0
    precision = round(sum(1 for r in has_err if r["match"]) / (sum(1 for r in has_err if r["match"]) + len(fp)) * 100, 1) if (sum(1 for r in has_err if r["match"]) + len(fp)) > 0 else 0

    # 按年级分组
    by_grade = {}
    for r in valid:
        g = r.get("grade", "unknown")
        if g not in by_grade:
            by_grade[g] = {"total": 0, "correct": 0}
        by_grade[g]["total"] += 1
        if r["match"]:
            by_grade[g]["correct"] += 1
    for g in by_grade:
        by_grade[g]["accuracy"] = round(by_grade[g]["correct"] / by_grade[g]["total"] * 100, 1)

    # 按领域分组
    by_domain = {}
    for r in valid:
        d = r.get("domain", "unknown")
        if d not in by_domain:
            by_domain[d] = {"total": 0, "correct": 0}
        by_domain[d]["total"] += 1
        if r["match"]:
            by_domain[d]["correct"] += 1
    for d in by_domain:
        by_domain[d]["accuracy"] = round(by_domain[d]["correct"] / by_domain[d]["total"] * 100, 1)

    # 平均耗时
    avg_time = round(sum(r.get("elapsed_sec", 0) for r in valid) / total, 1)

    return {
        "total_evaluated": len(records),
        "total_valid": total,
        "total_errors": len(records) - total,
        "accuracy": accuracy,
        "correct": correct,
        "wrong": total - correct,
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "recall_has_error": recall,
        "precision_has_error": precision,
        "avg_time_sec": avg_time,
        "by_grade": by_grade,
        "by_domain": by_domain,
    }


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="FERMAT 基准评测 — 作业批改系统")
    parser.add_argument("--n", type=int, default=SAMPLE_COUNT, help=f"评测样本数（默认{SAMPLE_COUNT}）")
    parser.add_argument("--mode", type=str, default=GRADING_MODE,
                        choices=["vision", "ocr_llm", "vision_deepseek", "both"],
                        help=f"批改模式（默认 {GRADING_MODE}）")
    parser.add_argument("--workers", type=int, default=CONCURRENCY, help=f"并发数（默认{CONCURRENCY}）")
    parser.add_argument("--resume", action="store_true", default=RESUME,
                        help="断点续跑（默认开启）")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="从头开始跑，丢弃已有 checkpoint")
    parser.add_argument("--output", type=str, default="fermat",
                        help="输出文件名前缀（默认 fermat，可设为 fermat_v2 等避免覆盖旧数据）")
    args = parser.parse_args()

    # 根据 output 前缀确定文件路径
    checkpoint_file = PROJECT_ROOT / "eval" / f"{args.output}_checkpoint.json"
    report_json = PROJECT_ROOT / "eval" / f"{args.output}_report.json"
    report_txt = PROJECT_ROOT / "eval" / f"{args.output}_summary.txt"

    print("=" * 60)
    print("  FERMAT 基准评测 — 作业批改系统")
    print(f"  样本数: {args.n}  并发: {args.workers}  模式: {args.mode}")
    print(f"  输出: {args.output}_*")
    print("=" * 60)

    # 加载 checkpoint
    checkpoint = {}
    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        # 只恢复本次需要的 mode
        done_vision = sum(1 for k, v in checkpoint.items()
                          if k.endswith("_vision") and v.get("match") is not None)
        done_ocr = sum(1 for k, v in checkpoint.items()
                       if k.endswith("_ocr_llm") and v.get("match") is not None)
        print(f"  📂 断点续跑: 已跑 vision={done_vision} ocr_llm={done_ocr}")
    else:
        if args.resume:
            print(f"  🆕 首次评测")

    # 加载数据
    rows = load_fermat_data(args.n)
    print(f"  📊 加载 {len(rows)} 条数据\n")

    # 确定要跑的模式
    modes = []
    if args.mode in ("vision", "both"):
        modes.append("vision")
    if args.mode in ("ocr_llm", "both"):
        modes.append("ocr_llm")
    if args.mode in ("vision_deepseek", "both"):
        modes.append("vision_deepseek")

    t_start = time.time()

    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"  🚀 开始 {mode.upper()} 模式评测")
        print(f"{'─' * 60}")

        records = []
        # 过滤出该 mode 未完成的
        pending = []
        for i, row in enumerate(rows):
            if f"{i}_{mode}" not in checkpoint:
                pending.append((i, row))

        if not pending:
            print(f"  ✅ 全部已完成，跳过")
            continue

        print(f"  待跑: {len(pending)} 条  已完成: {len(rows) - len(pending)} 条")

        batch_size = args.workers
        total_batches = (len(pending) + batch_size - 1) // batch_size
        batch_num = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending_list = list(pending)
            while pending_list:
                batch_num += 1
                batch = pending_list[:batch_size]
                pending_list = pending_list[batch_size:]

                t_batch = time.time()
                futures = {
                    executor.submit(evaluate_single, i, row, mode, checkpoint): i
                    for i, row in batch
                }

                batch_results = []
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        records.append(result)

                acc = sum(1 for r in records if r.get("match", False)) / max(len(records), 1) * 100
                elapsed = time.time() - t_batch
                total_elapsed = time.time() - t_start
                done = len(records) + (len(rows) - len(pending))
                eta = (total_elapsed / done * (len(pending) - len(records))) if done > 0 else 0

                print(f"  🚀 第{batch_num}/{total_batches}轮 | "
                      f"已完成 {done}/{len(rows)} | "
                      f"本轮 {elapsed:.0f}s | "
                      f"准确率 {acc:.1f}% | "
                      f"预计剩余 {eta:.0f}s")

            # 定期保存 checkpoint
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)

        # 合并已完成的所有记录
        all_mode_records = [
            checkpoint[k] for k in sorted(checkpoint.keys(), key=lambda x: int(x.split("_")[0]))
            if k.endswith(f"_{mode}") and checkpoint[k].get("match") is not None
        ]

        # 计算指标
        metrics = compute_metrics(all_mode_records)

        print(f"\n  📊 {mode.upper()} 评测结果:")
        print(f"    有效评测: {metrics['total_valid']}/{metrics['total_evaluated']}")
        print(f"    准确率:   {metrics['accuracy']}%")
        print(f"    正确判对: {metrics['correct'] - metrics['false_negatives']} 道")
        print(f"    误判(FP): {metrics['false_positives']} 道（实际没错被判错）")
        print(f"    漏判(FN): {metrics['false_negatives']} 道（实际有错被判对）")
        print(f"    召回率:   {metrics['recall_has_error']}%")
        print(f"    精确率:   {metrics['precision_has_error']}%")
        print(f"    平均耗时: {metrics['avg_time_sec']}秒/题")

        if metrics.get("by_grade"):
            print(f"\n    按年级:")
            for g in sorted(metrics["by_grade"].keys()):
                d = metrics["by_grade"][g]
                bar = "█" * int(d["accuracy"] / 5)
                print(f"      {g}: {d['accuracy']:5.1f}% {bar} ({d['correct']}/{d['total']})")

        if metrics.get("by_domain"):
            print(f"\n    按领域:")
            for dm in sorted(metrics["by_domain"].keys(), key=lambda x: -metrics["by_domain"][x]["accuracy"]):
                d = metrics["by_domain"][dm]
                bar = "█" * int(d["accuracy"] / 5)
                print(f"      {dm:12s}: {d['accuracy']:5.1f}% {bar} ({d['correct']}/{d['total']})")

    t_total = round(time.time() - t_start, 0)
    print(f"\n{'=' * 60}")
    print(f"  ✅ 评测完成！总耗时: {t_total:.0f}秒 ({t_total/60:.1f}分钟)")

    # 生成完整报告
    report = {
        "config": {"n": args.n, "mode": args.mode, "workers": args.workers},
        "total_time_sec": t_total,
        "modes": {},
    }

    for mode in modes:
        all_mode_records = [
            checkpoint[k] for k in sorted(checkpoint.keys(), key=lambda x: int(x.split("_")[0]))
            if k.endswith(f"_{mode}")
        ]
        report["modes"][mode] = {
            "metrics": compute_metrics(all_mode_records),
            "records": all_mode_records,
        }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 写可读报告
    lines = []
    lines.append("=" * 60)
    lines.append("  FERMAT 基准评测报告")
    lines.append(f"  样本数: {args.n}  并发: {args.workers}  耗时: {t_total:.0f}s")
    lines.append("=" * 60)
    for mode in modes:
        m = report["modes"][mode]["metrics"]
        lines.append(f"\n── {mode.upper()} ──")
        lines.append(f"  准确率: {m['accuracy']}% ({m['correct']}/{m['total_valid']})")
        lines.append(f"  漏判(FN): {m['false_negatives']}  误判(FP): {m['false_positives']}")
        lines.append(f"  召回: {m['recall_has_error']}%  精确: {m['precision_has_error']}%")
        lines.append(f"  平均: {m['avg_time_sec']}s/题")
        if m.get("by_grade"):
            lines.append(f"  按年级:")
            for g in sorted(m["by_grade"].keys()):
                d = m["by_grade"][g]
                lines.append(f"    {g}: {d['accuracy']}% ({d['correct']}/{d['total']})")

    # 出错样例
    for mode in modes:
        records = report["modes"][mode]["records"]
        errors = [r for r in records if r.get("match") is False]
        if errors:
            lines.append(f"\n── {mode.upper()} 出错样例 (前5条) ──")
            for r in errors[:5]:
                lines.append(f"  [{r['index']}] 实际{'有错' if r['ground_truth_has_error'] else '没错'} "
                             f"→ 模型判{'对' if r['model_is_correct'] else '错'} "
                             f"| {r['model_comment'][:80]}")

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── 写入 MySQL benchmark_runs 表 ──
    db_saved = 0
    try:
        from src.modules.db_manager import DBManager
        db = DBManager()
        for mode in modes:
            m = report["modes"][mode]["metrics"]
            db.save_benchmark_run({
                "sample_count": args.n,
                "mode": mode,
                "accuracy": m["accuracy"],
                "total_valid": m["total_valid"],
                "false_positives": m["false_positives"],
                "false_negatives": m["false_negatives"],
                "recall_rate": m["recall_has_error"],
                "precision_rate": m["precision_has_error"],
                "avg_time_sec": m["avg_time_sec"],
                "concurrency": args.workers,
                "note": "",
            })
            db_saved += 1
    except Exception as e:
        print(f"  ⚠️ 数据库写入失败（文件报告正常）: {e}")

    print(f"  报告已保存:")
    print(f"    {report_json}")
    print(f"    {report_txt}")
    print(f"    {checkpoint_file}")
    print(f"    MySQL benchmark_runs: +{db_saved} 条")


if __name__ == "__main__":
    main()
