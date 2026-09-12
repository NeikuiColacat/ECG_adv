"""Canonical portable-report payload, rendered by the installed report skill."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess

PLUGIN = Path("/home/linbinhao/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.10-13ceeea1f599")


def build_report(output: Path, metrics: dict, bootstrap: dict, overview: list[dict], quality: list[dict],
                 support: list[dict], provenance: dict, config: dict, *, diagnostics: list[dict] | None = None,
                 cohort_mode: str = "balanced_pilot", sensitivity: dict | None = None) -> None:
    if cohort_mode not in {"balanced_pilot", "full_pulse"} or (cohort_mode == "full_pulse" and sensitivity is None):
        raise ValueError("full report requires explicit cohort mode and sensitivity evidence")
    full = cohort_mode == "full_pulse"
    title = "图片 ECG 大模型：PN2021 配对鲁棒性评估"
    stamp = datetime.now(timezone.utc).isoformat()
    source_id = "paired-evaluation"
    models = list(metrics["models"])
    center_counts = {r["center"]: r["n_records"] for r in support}
    records = sum(center_counts.values())
    cohort_text = ("四中心合规全量 " + f"{records:,} 条 ECG（" + "、".join(f"{c}: {n:,}" for c, n in center_counts.items()) + "）"
                   if full else f"四中心各 {support[0]['n_records']} 条 ECG，共 {records} 条")
    totals = [row for row in overview if row["center"] == "center_equal"]
    conclusions = []
    for row in totals:
        ci = bootstrap["models"][row["model"]]["drop_pp_ci95"]
        uncertainty = "区间跨越零，不能确定下降方向" if ci[0] <= 0 <= ci[1] else "本测试集合存在下降证据" if ci[0] > 0 else "本测试集合未显示下降"
        conclusions.append(f"- {row['model']}：Clean Macro-F1 **{100*row['clean_macro_f1']:.2f}%** → 腐蚀均值 **{100*row['corrupted_macro_f1']:.2f}%**；下降 **{row['drop_pp']:.2f} 个百分点**，95% CI [{ci[0]:.2f}, {ci[1]:.2f}]。{uncertainty}。")
        if row["model"] in bootstrap["paired_vs_first_model"]:
            reference = next(r for r in totals if r["model"] == "PULSE-7B")
            delta = 100 * (row["corrupted_macro_f1"] - reference["corrupted_macro_f1"])
            delta_ci = bootstrap["paired_vs_first_model"][row["model"]]["corrupted_delta_pp_ci95"]
            conclusions[-1] += f" 相对 PULSE，腐蚀 Macro-F1 差为 {delta:+.2f} pp，配对 95% CI [{delta_ci[0]:+.2f}, {delta_ci[1]:+.2f}]。"
    shared = (cohort_text + "；每条包含 Clean 与固定 20 个 severity-5、depth-2/3 腐蚀条件。"
              "所有模型使用相同记录、标签和腐蚀种子。PULSE 复用已完成的同记录逐例预测，并按本表的等中心、等视图口径重新聚合。")
    center_note = ("中心样本量不同，主表仍为四中心等权，不按中心大小加权。" if full else "中心样本数量相同。")
    center_note += "未按标签分布配平；完整类别支持数在后表中。"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}\n\n冻结推理开发测试 · 500 Hz 原始波形 · 硬标签 Super5 · 无训练"},
        {"id": "summary", "type": "markdown", "body": "## 核心结论\n\n" + shared + "\n\n" + "\n".join(conclusions)},
        {"id": "definitions", "type": "markdown", "body": "## 指标与比较口径\n\n"
         "主要指标：每个条件先算固定五类 Macro-F1，20 个腐蚀条件等权，再对四中心等权。类别顺序为 CD、HYP、MI、NORM、STTC；分母为零时按 sklearn zero_division=0。"
         "Micro-F1、完整标签集合 exact-match accuracy、Hamming loss 同时导出；Hamming 越低越好，表中的 Clean−Corrupt 差值对它不应解释为性能下降。"
         "降幅小还须结合 Clean 与腐蚀后的绝对成绩解读，不能把低基线的地板效应视为更强鲁棒性。"
         "\n\n置信区间按中心分层、按原始 ECG 记录重采样，所有模型及 21 个视图一起抽样。它不是患者聚类区间，也不覆盖模型随机种子、腐蚀参数选择或总体采样框的不确定性。"
         "\n\n本次只生成类别标签，没有定义连续类别评分协议，因此不报告 AUROC/AUPRC。主结论是此测试协议下的腐蚀鲁棒性，不能单凭它推断相对于作者域内测试的跨中心泛化下降。"},
        {"id": "main-table", "type": "table", "tableId": "overview", "layout": "full"},
        {"id": "center-intro", "type": "markdown", "body": "## 四中心差异\n\n以下每张图对同一模型展示 Clean 和 20 个腐蚀视图的平均 Macro-F1。" + center_note},
    ]
    charts, datasets = [], {"overview": overview, "quality": quality, "support": support,
                            "classes": metrics["class_rows"], "conditions": metrics["condition_rows"],
                            "class_f1": [row for row in metrics["class_rows"] if row["metric"] == "f1"]}
    for index, model in enumerate(models):
        key = f"centers-{index}"
        center_rows = [row for row in overview if row["model"] == model and row["center"] != "center_equal"]
        datasets[key] = [{**row, "n_records": center_counts[row["center"]], "corruption_conditions": 20,
                          "condition": label, "macro_f1": row[field]}
                         for row in center_rows
                         for label, field in (("Clean", "clean_macro_f1"), ("Corrupted mean", "corrupted_macro_f1"))]
        charts.append({"id": key, "title": model, "subtitle": "固定五类；" + ("各中心合规全量" if full else "四中心等量抽样") + "；腐蚀为等视图均值", "type": "bar",
                       "dataset": key, "sourceId": source_id, "layout": "full", "valueFormat": "percent",
                       "showDescription": True, "yAxisTitle": "Macro-F1",
                       "palette": {"kind": "categorical"},
                       "encodings": {"x": {"field": "center", "type": "ordinal", "label": "中心"},
                                     "y": {"field": "macro_f1", "type": "quantitative", "label": "Macro-F1", "format": "percent"},
                                     "color": {"field": "condition", "type": "nominal", "label": "条件"}},
                       "labels": {"values": "all"}})
        largest = max(center_rows, key=lambda row: row["drop_pp"])
        smallest = min(center_rows, key=lambda row: row["drop_pp"])
        blocks.append({"id": f"chart-reading-{index}", "type": "markdown", "body":
                       f"### {model}：逐中心降幅 {smallest['drop_pp']:.2f}–{largest['drop_pp']:.2f} pp\n\n"
                       f"每组依次为 Clean 和腐蚀平均 Macro-F1，数值越高越好。{largest['center']} 的降幅最大，"
                       f"为 {largest['drop_pp']:.2f} pp；{smallest['center']} 为 {smallest['drop_pp']:.2f} pp。"
                       "同时比较两根柱的绝对高度；不同中心病例组成不同，不能把这一区别解释为纯中心效应。"})
        blocks.append({"id": f"chart-{index}", "type": "chart", "chartId": key, "layout": "full"})
    text_column = lambda field, label: {"field": field, "label": label, "type": "text"}
    numeric_column = lambda field, label, fmt="number": {"field": field, "label": label, "format": fmt}
    tables = [
        {"id": "overview", "title": "主要结果与逐中心结果", "dataset": "overview", "sourceId": source_id,
         "columns": [text_column("model", "模型"), text_column("center", "中心/聚合"),
                     numeric_column("clean_macro_f1", "Clean Macro-F1", "percent"),
                     numeric_column("corrupted_macro_f1", "Corrupted Macro-F1", "percent"),
                     numeric_column("drop_pp", "下降 (pp)"),
                     numeric_column("clean_micro_f1", "Clean Micro-F1", "percent"),
                     numeric_column("corrupted_micro_f1", "Corrupted Micro-F1", "percent"),
                     numeric_column("clean_exact_match", "Clean exact match", "percent"),
                     numeric_column("corrupted_exact_match", "Corrupted exact match", "percent"),
                     numeric_column("clean_hamming_loss", "Clean Hamming", "percent"),
                     numeric_column("corrupted_hamming_loss", "Corrupted Hamming", "percent")]},
        {"id": "support", "title": "真实标签支持数（多标签，可重叠）", "dataset": "support", "sourceId": source_id,
         "columns": [text_column("center", "中心"), numeric_column("n_records", "记录数")]
         + [numeric_column(label, label) for label in ("CD", "HYP", "MI", "NORM", "STTC")]},
        {"id": "quality", "title": "解析与输出质量", "dataset": "quality", "sourceId": source_id,
         "columns": [text_column("model", "模型"), text_column("center", "中心"), numeric_column("n_predictions", "预测数"),
                     numeric_column("parse_failure_rate", "空/无标签率", "percent"),
                     numeric_column("non_list_outputs", "非代码列表输出"),
                     numeric_column("common_parser_label_changes", "统一解析会改变的输出"),
                     numeric_column("normal_abnormal_conflicts", "正常与异常同报"), numeric_column("token_limit_hits", "输出上限次数")]},
        {"id": "class-f1", "title": "逐中心、逐类别 F1", "dataset": "class_f1", "sourceId": source_id,
         "columns": [text_column("model", "模型"), text_column("center", "中心"), text_column("class", "类别"),
                     numeric_column("positive_count", "真实阳性数"), numeric_column("clean", "Clean F1", "percent"),
                     numeric_column("corruption_view_mean", "Corrupted F1", "percent"), numeric_column("drop_pp", "下降 (pp)")]},
    ]
    if sensitivity is not None:
        datasets["sensitivity"] = sensitivity["overview"]
        datasets["sensitivity-support"] = [{"center": c, **counts} for c, counts in sensitivity["center_counts"].items()]
        tables.extend([
            {"id": "sensitivity", "title": "排除此前已看过记录后的结果（同一固定协议）", "dataset": "sensitivity", "sourceId": source_id,
             "columns": [text_column("model", "模型"), text_column("center", "中心/聚合"),
                         numeric_column("clean_macro_f1", "Clean Macro-F1", "percent"),
                         numeric_column("corrupted_macro_f1", "Corrupted Macro-F1", "percent"),
                         numeric_column("drop_pp", "下降 (pp)"), numeric_column("drop_pp_ci95_low", "下降 95% CI 下界"),
                         numeric_column("drop_pp_ci95_high", "下降 95% CI 上界")]},
            {"id": "sensitivity-support", "title": "敏感性分析记录数", "dataset": "sensitivity-support", "sourceId": source_id,
             "columns": [text_column("center", "中心"), numeric_column("full_records", "全量记录"),
                         numeric_column("excluded_records", "排除记录"), numeric_column("retained_records", "保留记录")]},
        ])
        blocks[4:4] = [
            {"id": "sensitivity-intro", "type": "markdown", "body": "## 已观察记录的敏感性分析\n\n"
             f"预先按身份排除此前 pilot/smoke 的 {sensitivity['excluded_records']:,} 条，保留 {sensitivity['retained_records']:,} 条。"
             "所有模型同步排除这些记录的全部21个视图；不根据预测对错选样本，主结论仍以全量结果为主。"
             "该检查不能消除已依据先导结果决定研究方向的影响，不把剩余集合自动称作严格盲测或严格域外数据。"},
            {"id": "sensitivity-table", "type": "table", "tableId": "sensitivity", "layout": "full"},
            {"id": "sensitivity-support-table", "type": "table", "tableId": "sensitivity-support", "layout": "full"},
        ]
    if diagnostics:
        datasets["diagnostics"] = diagnostics
        tables.append({"id": "diagnostics", "title": "接入诊断，不是分类成绩", "dataset": "diagnostics", "sourceId": source_id,
                       "columns": [text_column("probe", "探测"), text_column("status", "执行状态"),
                                   numeric_column("predictions", "生成次数"), numeric_column("no_code_outputs", "无五类代码"),
                                   numeric_column("token_limit_hits", "到达输出上限"), text_column("note", "说明")]})
        blocks[2:2] = [
            {"id": "diagnostics-intro", "type": "markdown", "body": "## 模型接入与纳入边界\n\n以下是独立 smoke 的运行及输出格式证据，不是标签正确率。显存不足、只输出报告或代码解析失败不能伪装成分类准确率为零，也不自动证明模型在其他任务或适配方式下不可用。只有完成同一直接标签任务的正式结果进入后续配对指标。"},
            {"id": "diagnostics-table", "type": "table", "tableId": "diagnostics", "layout": "full"},
        ]
    timing = []
    for model, info in provenance.items():
        if "shards" in info:
            perf = [s["performance"] for s in info["shards"]]
            if info["shards"][0].get("scheduler") == "fixed_task_queue_v1":
                p = perf[0]
                timing.append(f"- {model}：弹性固定任务队列，记录 {p['worker_launches']} 次模型 worker 接入；已提交任务平均 {p['images_per_second']:.2f} 图片/单卡秒，"
                              f"已正常结束 worker 的累计时间 {p['gpu_hours']:.2f} GPU-hour；峰值 allocated 显存 {p['peak_allocated_bytes']/1024**3:.2f} GiB。"
                              "任务吞吐不含模型载入、接入复核和失败重试；GPU-hour 若存在无最终记录的异常退出则是下界，须同时核对 workers 的 launch/exit 日志。")
            else:
                timing.append(f"- {model}：{len(perf)} 个分片；各卡端到端 {min(p['images_per_second'] for p in perf):.2f}–{max(p['images_per_second'] for p in perf):.2f} 图片/秒；峰值 allocated 显存 {max(p['peak_allocated_bytes'] for p in perf)/1024**3:.2f} GiB。")
        else:
            timing.append("- PULSE：复用历史逐例预测。本轮没有重新测速，不将历史吞吐当作硬件/批量完全配对的速度比较。")
    blocks.extend([
        {"id": "support-section", "type": "markdown", "body": "## 标签分布与解析质量\n\nSuper5 是多标签任务，类计数之和可以大于记录数。解析失败保留为空集合并计入整体评分，绝不自动填为 NORM；截断和正常/异常冲突另行报告。逐类别 precision/recall/F1 以及每个腐蚀条件的指标保存在随附 CSV。"},
        {"id": "support-table", "type": "table", "tableId": "support", "layout": "full"},
        {"id": "class-f1-table", "type": "table", "tableId": "class-f1", "layout": "full"},
        {"id": "quality-table", "type": "table", "tableId": "quality", "layout": "full"},
        {"id": "runtime", "type": "markdown", "body": "## 推理效率\n\n" + "\n".join(timing) + "\n\n新增模型使用 Transformers + SDPA，不是 vLLM。图片在线 GPU 渲染，仅在有界内存中存在；" +
         ("每个固定小任务完成后，正式预测与审计证据原子持久化。新增卡领剩余任务，安全退出先完成手头任务；任务和 batch 分组不依赖显卡数量。" if full else "正式预测与审计证据在分片完成后落盘。")},
        {"id": "design", "type": "markdown", "body": "## 实验设计与可复核性\n\n"
         + ("包含 PULSE 全量合规测试集合的全部记录；按固定哈希顺序组成小任务，不根据标签成绩选择规模或子集。先导和 smoke 身份单独标记并用于敏感性分析。" if full else
            "样本按固定哈希排序等量选取，独立保留 smoke 样本；正式规模只根据接入/吞吐预算冻结，不根据正式标签成绩选择。") +
         "数据为原始物理 mV、native 500 Hz；保持 PULSE 的预处理、20 个腐蚀条件、随机种子及 GPU renderer 源码哈希。"
         "固定排除各中心 K500，去掉全零 Super5 标签；映射版本 v7_super5_sjr_rgq_review_20260528，哈希 555ec85d5b51。"
         "\n\n共同 GPU 页面后，新模型执行 RGB uint8 舍入和各自官方图像处理器，PULSE 使用浮点页面的 CLIP 处理；因此共同渲染不等于模型处理后的 tensor 相同。"
         "权重 revision/文件 SHA256、完整 YAML 闭包、Git dirty 内容身份、逐例预测及文件索引均单独留存。新增模型保存逐例输入波形 SHA256，多模型汇总时核对其一致性。"
         "记录 hash 是身份哈希，不等于原始 WFDB 字节哈希；PULSE 历史结果没有保存波形字节 SHA，因此历史输入逐字节不变未获独立证明。"},
        {"id": "limitations", "type": "markdown", "body": "## 限制与下一步\n\n" + "\n".join(f"- {line}" for line in config["limitations"]) +
         ("\n\n下一步应单独建立源中心同任务、同渲染和同指标的配对参照，并核验训练数据重叠，不能直接用作者不同任务的数字相减。" if full else
          "\n\n建议先复核低支持类别和输出失败，再按同一冻结协议扩大样本。") +
         "任何提示词、渲染器或阈值优化必须另开开发集，不用本次正式成绩继续选择参数。"},
        {"id": "sources", "type": "markdown", "body": "## 数据与证据\n\n"
         "本报告由同目录 metrics.json、bootstrap.json、overview.csv、condition_metrics.csv、class_metrics.csv、class_support.csv、parse_quality.csv 生成。"
         "逐例配对预测和 provenance.json 提供原始结果与哈希，comparison_result.json 与受管 run 文件索引记录最终产物身份。"
         "\n\n解析复核逐条确认保存的标签等于各自冻结解析器的结果，并统计改用统一、大小写不敏感的最终答案解析器会改变多少条。此项仅作审计，不重写主表预测；若非零，比较仍可能受解析口径差异影响。"},
    ])
    # The portable renderer requires actual query provenance for every widget.
    # Metrics remain NumPy-derived; SQLite only selects the frozen display rows.
    # Run the recorded SQL, rather than inventing a pretend warehouse query.
    (output / "report_source_rows.json").write_text(json.dumps(datasets, ensure_ascii=False, indent=2) + "\n")
    sources = []
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    for index, (dataset, rows) in enumerate(datasets.items()):
        table = f"frozen_display_rows_{index}"
        fields = list(rows[0])
        connection.execute(f'CREATE TABLE "{table}" (' + ",".join(f'"{field}"' for field in fields) + ")")
        connection.executemany(f'INSERT INTO "{table}" VALUES (' + ",".join("?" for _ in fields) + ")",
                               [[row.get(field) for field in fields] for row in rows])
        sql = f'SELECT * FROM "{table}" ORDER BY rowid'
        datasets[dataset] = [dict(row) for row in connection.execute(sql)]
        source = {"id": f"display-{dataset}", "label": f"Validated {dataset} display rows", "path": "report_source_rows.json",
                  "query": {"engine": "sqlite", "sql": sql, "tables_used": [table],
                            "description": f"In-memory table loaded from report_source_rows.json dataset {dataset}; source metrics are NumPy-derived from paired predictions."}}
        sources.append(source)
        for widget in charts + tables:
            if widget["dataset"] == dataset:
                widget["sourceId"] = source["id"]
    connection.close()
    for table in tables:
        field = "drop_pp" if table["id"] in {"overview", "sensitivity", "class-f1"} else table["columns"][0]["field"]
        table["defaultSort"] = {"field": field, "direction": "desc" if field == "drop_pp" else "asc"}
    notes = {"audience": "technical", "delivery": "portable_html_localhost_only",
             "required_structure": {"title": "title", "technical_summary": "summary",
                                    "key_findings": ["main-table", "center-intro", "chart-reading-0", "chart-reading-1"],
                                    "scope_and_definitions": "definitions", "methodology": "design",
                                    "robustness": "sensitivity-intro" if full else "definitions",
                                    "limitations_next_steps_and_open_questions": "limitations"},
             "ordering_note": "Definitions precede detailed findings; robustness follows the primary table. Further questions are merged with limitations and next steps.",
             "chart_map": [{"id": chart["id"], "question": "Within-model Clean versus corrupted Macro-F1 across centers",
                            "family": "comparison_grouped_bar", "rows": 8,
                            "fields": ["center", "clean_macro_f1", "corrupted_macro_f1", "n_records", "drop_pp"],
                            "palette": "native_categorical_two_series_only", "non_color": "fixed_series_order_labels_and_semantic_table",
                            "delivery": "report.html", "claim": "Descriptive center-specific corruption gaps; not pure center effects"}
                           for chart in charts],
             "repeated_chart_reason": "Same paired comparison shown once per model; tables provide exact multi-metric and audit lookup.",
             "ci_scope": "1000 paired record bootstrap repeats for Macro-F1 and its deltas; other metrics have point estimates only.",
             "source_transform": "ecg_image_metrics.py computes metrics from paired prediction files; SQLite selects frozen display rows only.",
             "qa_scope": "See report_build.log; structural_only is not browser visual or interaction verification."}
    (output / "report_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n")
    artifact = {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title,
                "description": shared, "generatedAt": stamp, "cards": [], "charts": charts, "tables": tables,
                "sources": [{k: s[k] for k in ("id", "label", "path")} for s in sources], "blocks": blocks},
                "snapshot": {"version": 1, "generatedAt": stamp, "status": "ready", "datasets": datasets},
                "sources": sources}
    (output / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    process = subprocess.run(["node", str(PLUGIN / "skills/build-report/scripts/deliver_portable_artifact.mjs"),
                              "--input", str(output / "artifact.json"), "--output", str(output / "report.html")],
                             check=False, text=True, capture_output=True)
    (output / "report_build.log").write_text(process.stdout + "\n" + process.stderr)
    if process.returncode or not (output / "report.html").is_file():
        raise RuntimeError("portable report build failed; inspect report_build.log")
