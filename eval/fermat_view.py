"""
FERMAT 评测结果可视化
================================
从 checkpoint JSON 读取数据，生成可视化图表。

用法：
    python eval/fermat_view.py                # 打开交互式网页
    python eval/fermat_view.py --port 8001    # 指定端口
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_FILE = PROJECT_ROOT / "eval" / "fermat_checkpoint.json"

# ===================== 数据加载 =====================

def load_data():
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    vision, ocr, vd = [], [], []
    for k, v in raw.items():
        if k.endswith("_vision"):
            vision.append(v)
        elif k.endswith("_ocr_llm"):
            ocr.append(v)
        elif k.endswith("_vision_deepseek"):
            vd.append(v)

    return vision, ocr, vd


def stats(records):
    """计算统计指标"""
    valid = [r for r in records if r.get("match") is not None]
    errors = [r for r in records if r.get("match") is None]
    if not valid:
        return {}

    total = len(valid)
    matched = sum(1 for r in valid if r["match"])
    accuracy = round(matched / total * 100, 1)

    has_err = [r for r in valid if r["ground_truth_has_error"]]
    no_err = [r for r in valid if not r["ground_truth_has_error"]]
    fp = [r for r in no_err if r["model_is_correct"] is False]   # 没错被判错
    fn = [r for r in has_err if r["model_is_correct"] is True]   # 有错被判对

    recall = round(sum(1 for r in has_err if r["match"]) / len(has_err) * 100, 1) if has_err else 0
    prec_denom = sum(1 for r in has_err if r["match"]) + len(fp)
    precision = round(sum(1 for r in has_err if r["match"]) / prec_denom * 100, 1) if prec_denom else 0

    # 按年级
    by_grade = {}
    for r in valid:
        g = r.get("grade", "unknown")
        by_grade.setdefault(g, {"total": 0, "correct": 0})
        by_grade[g]["total"] += 1
        if r["match"]:
            by_grade[g]["correct"] += 1

    # 按领域
    by_domain = {}
    for r in valid:
        d = r.get("domain", "unknown")
        by_domain.setdefault(d, {"total": 0, "correct": 0})
        by_domain[d]["total"] += 1
        if r["match"]:
            by_domain[d]["correct"] += 1

    avg_time = round(sum(r.get("elapsed_sec", 0) for r in valid) / total, 1)
    # 估算实际墙钟耗时（并发后每轮等最慢那道）
    concurrency = 8
    total_rounds = max(1, (total + concurrency - 1) // concurrency)
    wall_time = round(avg_time * total_rounds, 0)

    return {
        "total": len(records),
        "valid": total,
        "errors": len(errors),
        "accuracy": accuracy,
        "matched": matched,
        "wrong": total - matched,
        "fp": len(fp),         # 本来没错但判错了
        "fn": len(fn),         # 本来有错但判对了
        "recall": recall,
        "precision": precision,
        "avg_time": avg_time,
        "wall_time": wall_time,
        "total_rounds": total_rounds,
        "by_grade": {g: {**d, "accuracy": round(d["correct"]/d["total"]*100, 1) if d["total"] else 0}
                     for g, d in sorted(by_grade.items())},
        "by_domain": {d: {**v, "accuracy": round(v["correct"]/v["total"]*100, 1) if v["total"] else 0}
                      for d, v in sorted(by_domain.items(), key=lambda x: -x[1]["total"])},
    }


# ===================== HTML 页面 =====================

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FERMAT 评测报告</title>
<style>
:root {
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8; --green: #22c55e;
    --red: #ef4444; --amber: #f59e0b; --blue: #3b82f6;
    --purple: #a855f7; --pink: #ec4899;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, "PingFang SC", sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }
.container { max-width:1200px; margin:0 auto; padding:32px 24px; }
h1 { font-size:1.6rem; font-weight:700; margin-bottom:8px; }
.sub { color:var(--muted); font-size:0.9rem; margin-bottom:32px; }

/* 概览卡片 */
.overview { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px,1fr)); gap:16px; margin-bottom:32px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:24px; }
.card-label { font-size:0.82rem; color:var(--muted); margin-bottom:6px; }
.card-value { font-size:2rem; font-weight:700; }
.green { color:var(--green); } .red { color:var(--red); } .amber { color:var(--amber); }
.blue { color:var(--blue); } .purple { color:var(--purple); }

/* 模式对比 */
.compare { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:32px; }
@media(max-width:700px) { .compare { grid-template-columns:1fr; } }
.mode-card { background:var(--card); border:1px solid var(--border); border-radius:16px; padding:28px; }
.mode-title { font-size:1.1rem; font-weight:700; margin-bottom:20px; display:flex; align-items:center; gap:8px; }
.mode-badge { font-size:0.72rem; padding:3px 10px; border-radius:12px; }
.badge-vision { background:rgba(59,130,246,0.2); color:var(--blue); }
.badge-ocr { background:rgba(168,85,247,0.2); color:var(--purple); }

/* 准确率大圆环 */
.gauge-wrap { text-align:center; margin:16px 0; }
.gauge { display:inline-block; position:relative; }
.gauge svg { transform:rotate(-90deg); }
.gauge-text { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center; }
.gauge-text .big { font-size:2.5rem; font-weight:800; line-height:1; }
.gauge-text .small { font-size:0.78rem; color:var(--muted); }

/* 四象限 */
.quad { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:16px 0; }
.quad-item { background:rgba(255,255,255,0.03); border-radius:10px; padding:16px; text-align:center; }
.quad-item .num { font-size:1.5rem; font-weight:700; }
.quad-item .lbl { font-size:0.75rem; color:var(--muted); margin-top:2px; }

/* 柱状图 */
.section { margin-bottom:32px; }
.section h2 { font-size:1.1rem; font-weight:600; margin-bottom:16px; }
.bar-row { display:flex; align-items:center; gap:12px; margin:6px 0; }
.bar-label { width:100px; text-align:right; font-size:0.85rem; color:var(--muted); flex-shrink:0; }
.bar-track { flex:1; height:24px; background:rgba(255,255,255,0.05); border-radius:12px; overflow:hidden; }
.bar-fill { height:100%; border-radius:12px; transition:width 0.8s ease; display:flex; align-items:center; padding-left:10px; font-size:0.75rem; font-weight:600; }
.bar-val { width:70px; font-size:0.82rem; color:var(--muted); flex-shrink:0; }

/* 混淆矩阵 */
.confusion { display:grid; grid-template-columns:1fr 1fr; gap:2px; max-width:300px; margin:16px auto; border-radius:12px; overflow:hidden; }
.cm-cell { padding:24px 16px; text-align:center; }
.cm-cell .num { font-size:1.8rem; font-weight:700; }
.cm-cell .lbl { font-size:0.72rem; margin-top:4px; opacity:0.8; }
.cm-header { background:rgba(255,255,255,0.03); padding:8px; text-align:center; font-size:0.75rem; color:var(--muted); }

/* 错题样例 */
.error-sample { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px; margin:8px 0; font-size:0.84rem; }
.error-sample .tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.72rem; font-weight:600; margin-right:6px; }
.tag-fp { background:rgba(239,68,68,0.15); color:var(--red); }
.tag-fn { background:rgba(245,158,11,0.15); color:var(--amber); }
.error-sample .detail { color:var(--muted); margin-top:6px; }
</style>
</head>
<body>
<div class="container">

<h1>📊 FERMAT 基准评测报告</h1>
<div class="sub">样本数: SAMPLE_COUNT 条 ｜ 并发: CONCURRENCY ｜ 三种模式对比</div>

<!-- 概览 -->
<div class="overview">
  <div class="card"><div class="card-label">🔵 Vision 准确率</div><div class="card-value blue">VISION_ACC%</div></div>
  <div class="card"><div class="card-label">🟣 OCR+LLM 准确率</div><div class="card-value purple">OCR_ACC%</div></div>
  <div class="card"><div class="card-label">🟢 V+DS 准确率</div><div class="card-value" style="color:#10b981;">VD_ACC%</div></div>
  <div class="card"><div class="card-label">⏱ Vision</div><div class="card-value green">VISION_TIME</div></div>
  <div class="card"><div class="card-label">⏱ OCR+LLM</div><div class="card-value amber">OCR_TIME</div></div>
  <div class="card"><div class="card-label">⏱ V+DS</div><div class="card-value" style="color:#10b981;">VD_TIME</div></div>
</div>

<!-- 三模式对比 -->
<div class="compare">
  <div class="mode-card">
    <div class="mode-title">🔵 Vision <span class="mode-badge badge-vision">Qwen-VL</span></div>
    VISION_GAUGE VISION_QUAD
    <div style="font-size:0.78rem;color:var(--muted);margin-top:8px;">✅VISION_OK ❌VISION_WRONG ⚠️VISION_ERR</div>
  </div>
  <div class="mode-card">
    <div class="mode-title">🟣 OCR+LLM <span class="mode-badge badge-ocr">OCR+DS</span></div>
    OCR_GAUGE OCR_QUAD
    <div style="font-size:0.78rem;color:var(--muted);margin-top:8px;">✅OCR_OK ❌OCR_WRONG ⚠️OCR_ERR</div>
  </div>
  <div class="mode-card">
    <div class="mode-title">🟢 Vision+DeepSeek <span class="mode-badge" style="background:rgba(16,185,129,0.2);color:#10b981;">V+DS</span></div>
    VD_GAUGE VD_QUAD
    <div style="font-size:0.78rem;color:var(--muted);margin-top:8px;">✅VD_OK ❌VD_WRONG ⚠️VD_ERR</div>
  </div>
</div>

<!-- 按年级 -->
<div class="section"><h2>📐 按年级分组准确率</h2>GRADE_BARS</div>

<!-- 混淆矩阵 -->
<div class="section"><h2>🔍 错误类型分析</h2>
  <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:16px;">
    <div><h3 style="font-size:0.85rem;color:#3b82f6;">Vision</h3>VISION_CONFUSION</div>
    <div><h3 style="font-size:0.85rem;color:#a855f7;">OCR+LLM</h3>OCR_CONFUSION</div>
    <div><h3 style="font-size:0.85rem;color:#10b981;">Vision+DeepSeek</h3>VD_CONFUSION</div>
  </div>
</div>

<!-- 错误汇总 -->
<div class="section"><h2>📋 评测结果明细</h2>ERROR_SUMMARY</div>

</div>
</body>
</html>"""


def gauge_svg(accuracy, color):
    """生成 SVG 圆环"""
    r, stroke = 56, 10
    c = 2 * 3.14159 * r
    offset = c * (1 - accuracy / 100)
    return f"""
    <div class="gauge-wrap">
      <div class="gauge">
        <svg width="150" height="150" viewBox="0 0 150 150">
          <circle cx="75" cy="75" r="{r}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="{stroke}"/>
          <circle cx="75" cy="75" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
                  stroke-dasharray="{c}" stroke-dashoffset="{offset}" stroke-linecap="round"/>
        </svg>
        <div class="gauge-text">
          <div class="big">{accuracy}%</div>
          <div class="small">准确率</div>
        </div>
      </div>
    </div>"""


def quad_html(m):
    valid = m["valid"]
    # TN + TP = matched; FP + FN = wrong
    # TP: 实际有错判成有错 (正确的检出)
    # TN: 实际没错判成没错 (正确的放过)
    tp = m["matched"] - m["fp"]  # 近似：match=True中ground_truth=True的
    tn = valid - m["fp"] - m["fn"] - tp
    # 简化计算
    matched = m["matched"]
    wrong = m["wrong"]
    return f"""
    <div class="quad">
      <div class="quad-item"><div class="num green">{matched - m['fn']}</div><div class="lbl">正确识别</div></div>
      <div class="quad-item"><div class="num red">{m['fp']}</div><div class="lbl">误判(没错判错)</div></div>
      <div class="quad-item"><div class="num amber">{m['fn']}</div><div class="lbl">漏判(有错判对)</div></div>
      <div class="quad-item"><div class="num green">{m['matched']}</div><div class="lbl">总正确数</div></div>
    </div>"""


def confusion_html(m, records):
    """混淆矩阵 HTML + 精确率召回率等指标"""
    tn = [r for r in records if r.get("match") is True and r["ground_truth_has_error"] is False]
    fp = [r for r in records if r.get("match") is False and r["ground_truth_has_error"] is False]
    fn = [r for r in records if r.get("match") is False and r["ground_truth_has_error"] is True]
    tp = [r for r in records if r.get("match") is True and r["ground_truth_has_error"] is True]

    TN, FP, FN, TP = len(tn), len(fp), len(fn), len(tp)
    acc = round((TP + TN) / (TP + TN + FP + FN) * 100, 1) if (TP + TN + FP + FN) else 0
    prec = round(TP / (TP + FP) * 100, 1) if (TP + FP) else 0
    rec = round(TP / (TP + FN) * 100, 1) if (TP + FN) else 0
    f1 = round(2 * prec * rec / (prec + rec), 1) if (prec + rec) else 0
    spec = round(TN / (TN + FP) * 100, 1) if (TN + FP) else 0

    return f"""
    <div class="confusion">
      <div class="cm-header" style="color:#22c55e;">✓ 模型判对</div><div class="cm-header" style="color:#ef4444;">✗ 模型判错</div>
      <div class="cm-cell" style="background:rgba(34,197,94,0.12)"><div class="num green">{TN}</div><div class="lbl">✅ 实际没错判对</div></div>
      <div class="cm-cell" style="background:rgba(239,68,68,0.15)"><div class="num red">{FP}</div><div class="lbl">⛔ 实际没错却判错</div></div>
      <div class="cm-cell" style="background:rgba(245,158,11,0.15)"><div class="num amber">{FN}</div><div class="lbl">⚠️ 实际有错却漏过</div></div>
      <div class="cm-cell" style="background:rgba(34,197,94,0.12)"><div class="num green">{TP}</div><div class="lbl">🎯 正确识别出错误</div></div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:14px;font-size:0.82rem;">
      <div style="color:var(--muted);">准确率</div><div style="text-align:right;font-weight:600;">{acc}%</div>
      <div style="color:var(--muted);">精确率（判错中真错比例）</div><div style="text-align:right;font-weight:600;">{prec}%</div>
      <div style="color:var(--muted);">召回率（有错中能发现比例）</div><div style="text-align:right;font-weight:600;">{rec}%</div>
      <div style="color:var(--muted);">F1 分数</div><div style="text-align:right;font-weight:600;color:#22c55e;">{f1}</div>
    </div>"""


def grade_bars(dm, vm, om):
    """按年级三模式准确率对比"""
    all_grades = set()
    for m in [dm, vm, om]:
        if m: all_grades.update(m.get("by_grade", {}).keys())
    if not all_grades: return '--'
    grades = sorted(all_grades)
    rows = ['<div style="font-size:0.75rem;color:var(--muted);margin-bottom:6px;">🔵 Vision  🟣 OCR+LLM  🟢 V+DS</div>']
    for g in grades:
        dv = dm.get("by_grade",{}).get(g,{}) if dm else {}
        vv = vm.get("by_grade",{}).get(g,{}) if vm else {}
        ov = om.get("by_grade",{}).get(g,{}) if om else {}
        da, va, oa = dv.get("accuracy",0), vv.get("accuracy",0), ov.get("accuracy",0)
        vt = dv.get("total", vv.get("total", 0))
        rows.append(f"""<div class="bar-row"><div class="bar-label">{g.upper()}</div>
          <div style="display:flex;gap:1px;flex:1;height:22px;border-radius:4px;overflow:hidden;">
            <div style="width:{va}%;background:#3b82f6;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:600;color:#fff;min-width:0;">{va}%</div>
            <div style="width:{oa}%;background:#a855f7;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:600;color:#fff;min-width:0;">{oa}%</div>
            <div style="width:{da}%;background:#10b981;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:600;color:#fff;min-width:0;">{da}%</div>
            <div style="flex:1;background:rgba(255,255,255,.02);"></div>
          </div><div class="bar-val">{vt}题</div></div>""")
    return '\n'.join(rows)


def error_summary(vr, orr, vdr, vm, om, dm):
    """错误类型汇总表格（三种模式）"""
    def g(m, k, default="--"):
        if not m: return default
        return str(m.get(k, default))
    has_v = bool(vm); has_o = bool(om); has_d = bool(dm)
    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.88rem;margin-top:12px;">
    <thead>
      <tr style="color:var(--muted);text-align:left;border-bottom:1px solid var(--border);">
        <th style="padding:10px 8px;">类型</th><th style="padding:10px 8px;">说明</th>
        {('<th style="padding:10px 8px;color:#3b82f6;">Vision</th>' if has_v else '')}
        {('<th style="padding:10px 8px;color:#a855f7;">OCR+LLM</th>' if has_o else '')}
        {('<th style="padding:10px 8px;color:#10b981;">V+DS</th>' if has_d else '')}
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
        <td style="padding:10px 8px;">✅ 正确</td>
        <td style="padding:10px 8px;color:var(--muted);">判断与标准答案一致</td>
        {('<td style="padding:10px 8px;color:#22c55e;font-weight:700;">'+g(vm,'matched')+'</td>' if has_v else '')}
        {('<td style="padding:10px 8px;color:#22c55e;font-weight:700;">'+g(om,'matched')+'</td>' if has_o else '')}
        {('<td style="padding:10px 8px;color:#22c55e;font-weight:700;">'+g(dm,'matched')+'</td>' if has_d else '')}
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
        <td style="padding:10px 8px;">🔴 误判 FP</td>
        <td style="padding:10px 8px;color:var(--muted);">实际没错却判错</td>
        {('<td style="padding:10px 8px;color:#ef4444;font-weight:700;">'+g(vm,'fp')+'</td>' if has_v else '')}
        {('<td style="padding:10px 8px;color:#ef4444;font-weight:700;">'+g(om,'fp')+'</td>' if has_o else '')}
        {('<td style="padding:10px 8px;color:#ef4444;font-weight:700;">'+g(dm,'fp')+'</td>' if has_d else '')}
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
        <td style="padding:10px 8px;">🟡 漏判 FN</td>
        <td style="padding:10px 8px;color:var(--muted);">实际有错却没发现</td>
        {('<td style="padding:10px 8px;color:#f59e0b;font-weight:700;">'+g(vm,'fn')+'</td>' if has_v else '')}
        {('<td style="padding:10px 8px;color:#f59e0b;font-weight:700;">'+g(om,'fn')+'</td>' if has_o else '')}
        {('<td style="padding:10px 8px;color:#f59e0b;font-weight:700;">'+g(dm,'fn')+'</td>' if has_d else '')}
      </tr>
      <tr>
        <td style="padding:10px 8px;">⚠️ 失败</td>
        <td style="padding:10px 8px;color:var(--muted);">返回无法解析</td>
        {('<td style="padding:10px 8px;color:var(--muted);font-weight:700;">'+g(vm,'errors')+'</td>' if has_v else '')}
        {('<td style="padding:10px 8px;color:var(--muted);font-weight:700;">'+g(om,'errors')+'</td>' if has_o else '')}
        {('<td style="padding:10px 8px;color:var(--muted);font-weight:700;">'+g(dm,'errors')+'</td>' if has_d else '')}
      </tr>
    </tbody>
    </table>"""


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="FERMAT 评测结果可视化")
    parser.add_argument("--port", type=int, default=8001, help="HTTP 端口 (默认8001)")
    args = parser.parse_args()

    print("📊 加载评测数据...")
    vision, ocr, vd = load_data()
    vm = stats(vision) if vision else {}
    om = stats(ocr) if ocr else {}
    dm = stats(vd) if vd else {}

    for name, s in [("Vision", vm), ("OCR+LLM", om), ("Vision+DeepSeek", dm)]:
        if s:
            print(f"   {name}: {s['valid']}有效 {s['errors']}失败  准确率 {s['accuracy']}%")

    if not vm and not dm and not om:
        print("❌ 数据不足"); return

    def v(m, k, default="--"):
        return str(m.get(k, default)) if m else "--"

    page = HTML
    page = page.replace("SAMPLE_COUNT", str(max(len(vision or []), len(ocr or []), len(vd or []))))
    page = page.replace("CONCURRENCY", "3")
    page = page.replace("VISION_ACC", v(vm,"accuracy"))
    page = page.replace("OCR_ACC", v(om,"accuracy"))
    page = page.replace("VD_ACC", v(dm,"accuracy"))
    page = page.replace("VISION_TIME", v(vm,"avg_time")+"s/题")
    page = page.replace("OCR_TIME", v(om,"avg_time")+"s/题")
    page = page.replace("VD_TIME", v(dm,"avg_time")+"s/题")
    page = page.replace("VISION_GAUGE", gauge_svg(vm.get("accuracy",0), "#3b82f6") if vm else "--")
    page = page.replace("OCR_GAUGE", gauge_svg(om.get("accuracy",0), "#a855f7") if om else "--")
    page = page.replace("VD_GAUGE", gauge_svg(dm.get("accuracy",0), "#10b981") if dm else "--")
    page = page.replace("VISION_QUAD", quad_html(vm) if vm else "--")
    page = page.replace("OCR_QUAD", quad_html(om) if om else "--")
    page = page.replace("VD_QUAD", quad_html(dm) if dm else "--")
    page = page.replace("VISION_OK", v(vm,"matched")); page = page.replace("VISION_WRONG", v(vm,"wrong")); page = page.replace("VISION_ERR", v(vm,"errors"))
    page = page.replace("OCR_OK", v(om,"matched")); page = page.replace("OCR_WRONG", v(om,"wrong")); page = page.replace("OCR_ERR", v(om,"errors"))
    page = page.replace("VD_OK", v(dm,"matched")); page = page.replace("VD_WRONG", v(dm,"wrong")); page = page.replace("VD_ERR", v(dm,"errors"))
    page = page.replace("GRADE_BARS", grade_bars(dm, vm, om))
    page = page.replace("VISION_CONFUSION", confusion_html(vm, vision) if vm else "--")
    page = page.replace("OCR_CONFUSION", confusion_html(om, ocr) if om else "--")
    page = page.replace("VD_CONFUSION", confusion_html(dm, vd) if dm else "--")
    page = page.replace("ERROR_SUMMARY", error_summary(vision, ocr, vd, vm, om, dm))

    # 保存 HTML
    report_path = PROJECT_ROOT / "eval" / "fermat_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(page)

    print(f"\n✅ 可视化报告: {report_path}")
    print(f"   用浏览器打开: file:///{report_path.as_posix()}")

    # 也提供 HTTP 服务
    try:
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import os
        os.chdir(str(PROJECT_ROOT / "eval"))
        server = HTTPServer(("0.0.0.0", args.port), SimpleHTTPRequestHandler)
        print(f"\n🌐 或访问: http://localhost:{args.port}/fermat_report.html")
        print(f"   Ctrl+C 关闭")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 已关闭")
    except Exception as e:
        print(f"\n⚠️ HTTP 服务启动失败: {e}")
        print(f"   直接用浏览器打开上面的 file:// 链接即可")


if __name__ == "__main__":
    main()
