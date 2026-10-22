#!/usr/bin/env python
"""compare_results.py: side-by-side scores of the run_comparison_set.sh runs.

Run from the repo root in an Inspect venv:
  python compare_results.py               # full runs (<task>/logs/)
  python compare_results.py logs/smoke    # smoke runs

Scores are over all samples x epochs; errored samples count as 0 (AGENTS.md section 3).
Reads the newest .eval in each log dir, so keep only the final run there.
"""
import glob, math, statistics, sys
from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.scorer import value_to_float

MODELS = ["Qwen_Qwen3.8-27B", "RedHatAI_Qwen3.8-27B-MXFP4", "RedHatAI_Qwen3.8-27B-MXFP4xFP8"]
TASKS = ["knowledge-and-reasoning/gpqa-diamond", "knowledge-and-reasoning/aime-2026",
         "knowledge-and-reasoning/mmlu-pro", "agentic-and-coding/terminal-bench-2.1"]
sub = sys.argv[1] if len(sys.argv) > 1 else "logs"
to_float = value_to_float()

print(f"{'task':<22} {'model':<32} {'score':>7} {'stderr':>7} {'n':>6} {'err':>4} {'trunc':>5} {'limit':>5} {'out tok mean':>12}")
for t in TASKS:
    for m in MODELS:
        files = sorted(glob.glob(f"{m}/{t}/{sub}/*.eval"))
        if not files:
            print(f"{t.split('/')[-1]:<22} {m:<32} {'no log':>7}")
            continue
        f = files[-1]
        status = read_eval_log(f, header_only=True).status
        vals, out_tok = [], []
        err = trunc = limit = 0
        for s in read_eval_log_samples(f, all_samples_required=False):
            if s.error:
                err += 1
                vals.append(0.0)
                continue
            limit += bool(s.limit)
            trunc += s.output.stop_reason == "max_tokens"
            if s.output.usage:
                out_tok.append(s.output.usage.output_tokens)
            sc = next(iter(s.scores.values()), None) if s.scores else None
            vals.append(to_float(sc.value) if sc is not None else 0.0)
        n = len(vals)
        mean = sum(vals) / n if n else float("nan")
        se = statistics.stdev(vals) / math.sqrt(n) if n > 1 else float("nan")
        tok = f"{statistics.mean(out_tok):.0f}" if out_tok else "-"
        flag = "" if status == "success" else f"  [{status}]"
        print(f"{t.split('/')[-1]:<22} {m:<32} {mean:7.3f} {se:7.3f} {n:6d} {err:4d} {trunc:5d} {limit:5d} {tok:>12}{flag}")
    print()
