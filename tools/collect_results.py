#!/usr/bin/env python3
"""Collect the README Results numbers from the final .eval log of task dirs. Run it with the client venv's python.

    python tools/collect_results.py [--smoke] [--detail] DIR...     DIR: task dirs, or model / category dirs above them

    python tools/collect_results.py Qwen_Qwen3.8-27B RedHatAI_Qwen3.8-27B-MXFP4 RedHatAI_Qwen3.8-27B-MXFP4xFP8

Finds every task dir (a dir with run_eval.sh) under the arguments, reads the newest .eval in its logs/ (logs/smoke/
with --smoke), and prints one summary table grouped by task. --detail adds, per dir, the README Results block: the
generate config line, the metrics table and the token table.

Scores are over all samples (AGENTS.md section 3): an errored or unscored sample counts as 0 for that epoch, and a
sample missing an epoch counts 0 for it. With N epochs, each sample's score is its mean over the N epochs (avg@N), and
stderr is the standard error of those per-sample means. Inspect's own figure (errors excluded) is shown as `inspect`
so the two can be compared. Tokens are per sample and epoch, summed over all model calls of the sample (agentic
samples make many), from sample.model_usage; samples that errored before any model call are left out of them.
"""
import argparse
import glob
import math
import os
import statistics
import sys
from collections import defaultdict

from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.scorer import value_to_float

to_float = value_to_float()


def task_dirs(roots):
    found = set()
    for r in roots:
        if os.path.isfile(os.path.join(r, "run_eval.sh")):
            found.add(os.path.normpath(r))
        found.update(os.path.dirname(os.path.normpath(p))
                     for p in glob.glob(os.path.join(r, "**", "run_eval.sh"), recursive=True))
    return sorted(found)


def newest_log(d, smoke):
    logs = glob.glob(os.path.join(d, "logs", "smoke" if smoke else "", "*.eval"))
    return (max(logs, key=os.path.getmtime), len(logs)) if logs else (None, 0)


def score_value(sample, scorer):
    if sample.error or not sample.scores:
        return 0.0
    sc = sample.scores.get(scorer) or next(iter(sample.scores.values()))
    try:
        return to_float(sc.value)
    except Exception:
        return 0.0


def stats3(xs):
    return (statistics.mean(xs), statistics.median(xs), max(xs)) if xs else (0, 0, 0)


def collect(path):
    head = read_eval_log(path, header_only=True)
    scorer = head.results.scores[0].name if head.results and head.results.scores else None
    epochs = head.eval.config.epochs or 1
    per_id = defaultdict(dict)  # id -> epoch -> score
    n_err = n_trunc = n_unscored = 0
    limits = defaultdict(int)
    tok = {"input": [], "output": [], "reasoning": []}
    for s in read_eval_log_samples(path, all_samples_required=False):
        per_id[s.id][s.epoch] = score_value(s, scorer)
        n_err += bool(s.error)
        n_unscored += not s.error and not s.scores
        n_trunc += bool(s.output and s.output.choices and s.output.stop_reason == "max_tokens")
        if s.limit:
            limits[s.limit.type] += 1
        usage = (s.model_usage or {}).values()
        if not usage:  # errored before any model call
            continue
        tok["input"].append(sum(u.input_tokens or 0 for u in usage))
        tok["output"].append(sum(u.output_tokens or 0 for u in usage))
        tok["reasoning"].append(sum(u.reasoning_tokens or 0 for u in usage))
    ids = list(head.eval.dataset.sample_ids or []) or list(per_id)
    sample_means = [sum(per_id.get(i, {}).get(e, 0.0) for e in range(1, epochs + 1)) / epochs for i in ids]
    n = len(sample_means)
    acc = sum(sample_means) / n if n else 0.0
    stderr = statistics.stdev(sample_means) / math.sqrt(n) if n > 1 else 0.0
    per_epoch = [sum(per_id.get(i, {}).get(e, 0.0) for i in ids) / n for e in range(1, epochs + 1)] if n else []
    inspect_acc = None
    if head.results and head.results.scores:
        m = head.results.scores[0].metrics
        inspect_acc = (m.get("accuracy") or m.get("mean") or next(iter(m.values()), None))
        inspect_acc = inspect_acc.value if inspect_acc else None
    return dict(path=path, status=head.status, scorer=scorer, epochs=epochs, n=n,
                n_runs=sum(len(v) for v in per_id.values()), acc=acc, stderr=stderr,
                correct=sum(sample_means), per_epoch=per_epoch, inspect_acc=inspect_acc,
                n_err=n_err, n_unscored=n_unscored, n_trunc=n_trunc, limits=dict(limits), tok=tok,
                config=head.eval.model_generate_config, model=head.eval.model,
                task_args=head.eval.task_args, limit=head.eval.config.limit,
                time_limit=head.eval.config.time_limit, model_args=head.eval.model_args)


def fmt_limits(r):
    return ", ".join(f"{k} {v}" for k, v in r["limits"].items()) or "0"


def print_detail(d, r):
    cfg = r["config"].model_dump(exclude_none=True)
    extra = [f"epochs={r['epochs']}"]
    if r["time_limit"]:
        extra.append(f"time_limit={r['time_limit']}")
    if r["limit"]:
        extra.append(f"limit={r['limit']}")
    if "client_timeout" in (r["model_args"] or {}):
        extra.append(f"client_timeout={r['model_args']['client_timeout']}")
    print(f"\n### {d}\n{r['path']}  status={r['status']}  scorer={r['scorer']}\n")
    print("With model generation config:")
    print(f"`{cfg}, {', '.join(extra)}`")
    print("the following scores and token stats are obtained:\n")
    label = "accuracy (pass@1)" if r["epochs"] == 1 else f"accuracy (avg@{r['epochs']})"
    print("| metric | value |\n|---|---|")
    print(f"| {label} | {r['acc']:.3f} ({r['correct']:g} of {r['n']}) |")
    print(f"| stderr | {r['stderr']:.3f} |")
    if r["epochs"] > 1:
        print(f"| per-epoch | {' / '.join(f'{x:.3f}' for x in r['per_epoch'])} |")
    print(f"| samples | {r['n']} × {r['epochs']} epoch(s) = {r['n'] * r['epochs']} ({r['n_runs']} in log) |")
    print(f"| truncated (max_tokens) | {r['n_trunc']} |")
    print(f"| limit hits | {fmt_limits(r)} |")
    print(f"| unscored | {r['n_unscored']} |")
    print(f"| sample errors | {r['n_err']} |")
    print("\n| tokens / sample | input | output | reasoning |\n|---|---|---|---|")
    st = {k: stats3(v) for k, v in r["tok"].items()}
    for i, name in enumerate(("mean", "median", "max")):
        print(f"| {name} | {st['input'][i]:,.0f} | {st['output'][i]:,.0f} | {st['reasoning'][i]:,.0f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--smoke", action="store_true", help="read logs/smoke/ instead of logs/")
    ap.add_argument("--detail", action="store_true", help="also print the README Results block per dir")
    a = ap.parse_args()

    rows = []
    for d in task_dirs(a.dirs):
        path, n_logs = newest_log(d, a.smoke)
        if not path:
            rows.append((d, None, "no .eval log"))
            continue
        try:
            rows.append((d, collect(path), f"{n_logs} .eval logs, newest used" if n_logs > 1 else ""))
        except Exception as e:
            rows.append((d, None, f"failed to read {path}: {e}"))
    if not rows:
        sys.exit("no task dirs (dirs with run_eval.sh) found under the arguments")

    # group by task (category/task), models side by side
    by_task = defaultdict(list)
    for d, r, note in rows:
        parts = d.split(os.sep)
        by_task[os.sep.join(parts[-2:])].append((parts[-3] if len(parts) >= 3 else d, r, note))
    hdr = "| task | model | accuracy | stderr | inspect | samples × ep | errors | max_tokens | limits | status |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for task, items in by_task.items():
        for model, r, note in items:
            if r is None:
                print(f"| {task} | {model} | – | – | – | – | – | – | – | {note} |")
                continue
            ia = f"{r['inspect_acc']:.3f}" if isinstance(r["inspect_acc"], (int, float)) else "–"
            status = r["status"] + (f"; {note}" if note else "") + (f"; limit={r['limit']}" if r["limit"] else "")
            print(f"| {task} | {model} | {r['acc']:.3f} ({r['correct']:g}/{r['n']}) | {r['stderr']:.3f} | {ia} "
                  f"| {r['n']} × {r['epochs']} | {r['n_err']} | {r['n_trunc']} | {fmt_limits(r)} | {status} |")
    if a.detail:
        for d, r, _ in rows:
            if r:
                print_detail(d, r)


if __name__ == "__main__":
    main()
