#!/usr/bin/env python3
"""Collate and analyse speculative-decoding metrics from Inspect `.eval` logs.

Two stages, usable separately or in one shot:

    ./analyze_spec_decode.py collate LOG_DIR [-o calls.csv]   # needs `inspect_ai` (run from the client venv)
    ./analyze_spec_decode.py analyze calls.csv [-o OUT_DIR]   # stdlib only
    ./analyze_spec_decode.py LOG_DIR [-o OUT_DIR]             # both

`collate` reads every `*.eval` under LOG_DIR with `read_eval_log(..., resolve_attachments=True)`
(the `--log-model-api` raw payloads are large and stored as attachments), and writes one row per
model API call. It relies on two things this task dir sets up:

  * `serve.sh` passes `--per-request-spec-decode-metrics detailed`, so every chat-completion
    response carries `metrics.speculative_decoding`;
  * `run_eval.sh` passes `--log-model-api`, so every call's raw request/response is kept in the
    log (by default Inspect keeps only the first few calls per model plus errors).

Calls without spec metrics still get a row with `has_spec=0`, so coverage is visible instead of
silently shrinkwrapping the denominators.

`analyze` computes acceptance statistics from that CSV. Metric definitions:

  * MAL (mean acceptance length) = 1 + accepted_draft_tokens / num_spec_steps. The leading 1 is
    the always-accepted bonus token, so a draft that is never accepted gives MAL 1.0.
  * micro MAL pools the accepted-token and verify-step counts before dividing, so every verify step
    carries the same weight and long responses dominate. Macro MAL is the unweighted mean of the
    per-call values, so every request carries the same weight no matter how many verify steps it ran.
    The two differ whenever a call's step count correlates with its acceptance, which it does (short
    tool-call turns accept more than long reasoning turns). Micro is the headline; macro is reported
    next to it.
  * per-position marginal acceptance = hist[j] / num_spec_steps: the fraction of verify steps that
    accepted *exactly* j draft tokens (the "exactly j" probability mass function).
  * per-position overall acceptance = sum_{i>j} hist[i] / num_spec_steps: the probability that the draft
    token at position j is accepted. Draft tokens are accepted in prefix order, so this equals the
    probability that a verify step accepted more than j drafts. It is the tail of the histogram and the
    model-card per-position table. Summing it over j and adding 1 reproduces MAL.
  * per-position conditional acceptance = overall[j] / overall[j-1]: the probability that position j is
    accepted given that position j-1 was. Position 0 has no predecessor, so its conditional rate equals
    its overall rate. This is the per-position hazard rate; a flat column means acceptance is geometric
    in position, and overall[j] is the running product of the conditionals. The two are therefore
    interchangeable: conditional from overall by division, overall from conditional by cumulative product.
  * thinking vs response: the completion is one contiguous token sequence, and each verify step emits its
    accepted draft tokens followed by one always-accepted bonus token, so step i covers tokens
    [o_i, o_i + a_i + 1) from o_0 = 0 (verified: sum(a_i + 1) == completion_tokens). The reasoning block is
    the prefix of the completion, so its length in tokens, usage.reasoning_tokens, is the boundary
    (verified equal to the token count of the reasoning text). MAL, draft acceptance rate and both
    per-position rates are reported per region with no tokenizer and no extra vLLM endpoint.

Outputs in OUT_DIR: `report.md` (human), `report.json` (machine), and one CSV per slice family.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPEC_FIELDS = [
    "mean_acceptance_length",
    "acceptance_histogram",
    "num_spec_steps",
    "num_accepted_draft_tokens",
    "num_draft_tokens",
    "num_spec_tokens",
    "per_step_accepted",
    "per_step_drafted",
]

CSV_COLUMNS = [
    "log_file",
    "sample_id",
    "epoch",
    "turn",
    "model",
    "role",
    "error",
    "stop_reason",
    "input_tokens",
    "input_cache_read",
    "input_cache_write",
    "output_tokens",
    "reasoning_tokens",
    "has_spec",
    "mal",
    "num_spec_steps",
    "num_accepted_draft_tokens",
    "num_draft_tokens",
    "num_spec_tokens",
    "acceptance_histogram",
    "per_step_accepted",
    "per_step_drafted",
    "sample_score",
    "sample_error",
    "sample_limit",
]

# --------------------------------------------------------------------------- collate


def _get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _jsonify(v: Any) -> str:
    return "" if v is None else json.dumps(v, separators=(",", ":"))


def collate(log_dir: Path, out_csv: Path) -> int:
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:  # pragma: no cover - guidance for the wrong venv
        sys.exit(
            "collate needs inspect_ai: run it from the Inspect client venv "
            "(the one local_orchestrator.sh activates), or run `analyze` on an existing CSV."
        )

    logs = sorted(log_dir.rglob("*.eval"))
    if not logs:
        sys.exit(f"no *.eval logs under {log_dir}")

    rows: list[dict[str, Any]] = []
    n_calls = n_spec = 0
    for path in logs:
        log = read_eval_log(str(path), resolve_attachments=True)
        log_name = str(path.relative_to(log_dir)) if path.is_relative_to(log_dir) else str(path)
        for sample in log.samples or []:
            score = None
            for sc in (sample.scores or {}).values():
                score = _num(_get(sc, "value"))
                break
            turn = 0
            for ev in sample.events:
                if _get(ev, "event") != "model":
                    continue
                turn += 1
                n_calls += 1
                call = _get(ev, "call")
                resp = _get(call, "response") if call is not None else None
                metrics = _get(resp, "metrics") if isinstance(resp, dict) else None
                spec = _get(metrics, "speculative_decoding") if metrics else None

                usage = _get(ev, "output")
                usage = _get(usage, "usage") if usage is not None else None
                output = _get(ev, "output")

                row: dict[str, Any] = {
                    "log_file": log_name,
                    "sample_id": _get(sample, "id"),
                    "epoch": _get(sample, "epoch"),
                    "turn": turn,
                    "model": _get(ev, "model"),
                    "role": _get(ev, "role") or "",
                    "error": "" if not _get(call, "error") else "1",
                    "stop_reason": _get(output, "stop_reason"),
                    "input_tokens": _get(usage, "input_tokens"),
                    "input_cache_read": _get(usage, "input_tokens_cache_read"),
                    "input_cache_write": _get(usage, "input_tokens_cache_write"),
                    "output_tokens": _get(usage, "output_tokens"),
                    "reasoning_tokens": _get(usage, "reasoning_tokens"),
                    "has_spec": 1 if spec else 0,
                    "sample_score": score,
                    "sample_error": _get(sample, "error") or "",
                    "sample_limit": _get(sample, "limit") or "",
                }
                if spec:
                    n_spec += 1
                    for f in SPEC_FIELDS:
                        row[f] = spec.get(f)
                    row["mal"] = spec.get("mean_acceptance_length")
                    for f in ("acceptance_histogram", "per_step_accepted", "per_step_drafted"):
                        row[f] = _jsonify(spec.get(f))
                else:
                    for f in ("mal", *SPEC_FIELDS):
                        row.setdefault(f, "")
                rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})

    print(f"collated {n_calls} model calls from {len(logs)} log(s); {n_spec} carried spec metrics")
    print(f"wrote {out_csv}")
    return 0


# --------------------------------------------------------------------------- analyze


TURN_BINS: list[tuple[str, int, int | None]] = [
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
    ("41+", 41, None),
]
CTX_BINS: list[tuple[str, int, int | None]] = [
    ("<4k", 0, 3_999),
    ("4-16k", 4_000, 15_999),
    ("16-32k", 16_000, 31_999),
    ("32-64k", 32_000, 63_999),
    ("64-128k", 64_000, 127_999),
    ("128k+", 128_000, None),
]
OUT_BINS: list[tuple[str, int, int | None]] = [
    ("<1k", 0, 999),
    ("1-4k", 1_000, 3_999),
    ("4-16k", 4_000, 15_999),
    ("16-64k", 16_000, 63_999),
    ("64k+", 64_000, None),
]


def _bucket(value: float | None, bins: list[tuple[str, int, int | None]]) -> str | None:
    if value is None:
        return None
    for label, lo, hi in bins:
        if value >= lo and (hi is None or value <= hi):
            return label
    return None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _position_stats(hist: Counter, steps: int) -> list[dict[str, Any]]:
    """Per-position overall and conditional acceptance from a verify-step histogram.

    hist[j] = number of verify steps that accepted exactly j draft tokens; steps = total verify steps.
    overall[j] = P(step accepted more than j drafts) = P(position j accepted) (prefix-ordered acceptance).
    conditional[j] = overall[j] / overall[j-1] is the hazard rate at position j. Only j in 0..k-1 are real
    draft positions; index k is the "accepted all k drafts" boundary.
    """
    if not steps:
        return []
    kmax = max(hist) if hist else 0
    marginal = [hist.get(j, 0) / steps for j in range(kmax + 1)]
    overall = [
        sum(hist.get(i, 0) for i in range(j + 1, kmax + 1)) / steps for j in range(kmax + 1)
    ]
    positions = []
    for j in range(kmax):
        if j == 0:
            cond: float | None = overall[0]
        else:
            prev = overall[j - 1]
            cond = overall[j] / prev if prev else None
        positions.append(
            {"position": j, "marginal": marginal[j], "overall": overall[j], "conditional": cond}
        )
    return positions


class Agg:
    """Accumulates spec-decode counts for one slice."""

    __slots__ = ("calls", "steps", "accepted", "mal", "hist", "per_step")

    def __init__(self) -> None:
        self.calls = 0
        self.steps = 0
        self.accepted = 0
        self.mal: list[float] = []
        self.hist: Counter[int] = Counter()
        self.per_step: list[int] = []

    def add(self, r: dict[str, Any]) -> None:
        steps = r["num_spec_steps"]
        if not steps:
            return
        self.calls += 1
        self.steps += steps
        self.accepted += r["num_accepted_draft_tokens"] or 0
        if r["mal"] is not None:
            self.mal.append(r["mal"])
        for j, c in enumerate(r["acceptance_histogram"]):
            self.hist[j] += c
        self.per_step.extend(r["per_step_accepted"])

    def result(self) -> dict[str, Any]:
        if not self.calls:
            return {"calls": 0}
        steps = self.steps
        # Pad responses that hit max_tokens: they report fewer steps than k.
        kmax = max(self.hist) if self.hist else 0
        positions = _position_stats(self.hist, steps)
        return {
            "calls": self.calls,
            "num_spec_steps": steps,
            "micro_mal": 1 + self.accepted / steps,
            "macro_mal": statistics.fmean(self.mal) if self.mal else None,
            "positions": positions,
            "marginal_acceptance": [p["marginal"] for p in positions]
            + ([self.hist.get(kmax, 0) / steps] if kmax else []),
            "survival_acceptance": [p["overall"] for p in positions]
            + ([0.0] if kmax else []),
        }


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def analyze(csv_path: Path, out_dir: Path) -> int:
    with open(csv_path, newline="") as fh:
        raw = list(csv.DictReader(fh))
    if not raw:
        sys.exit(f"{csv_path} is empty")

    rows: list[dict[str, Any]] = []
    for r in raw:
        row = dict(r)
        for k in (
            "epoch",
            "turn",
            "input_tokens",
            "input_cache_read",
            "input_cache_write",
            "output_tokens",
            "reasoning_tokens",
            "num_spec_steps",
            "num_accepted_draft_tokens",
            "num_draft_tokens",
            "num_spec_tokens",
        ):
            row[k] = int(float(r[k])) if r.get(k) not in (None, "") else None
        for k in ("mal", "sample_score"):
            row[k] = _num(r.get(k))
        for k in ("acceptance_histogram", "per_step_accepted", "per_step_drafted"):
            v = r.get(k)
            row[k] = json.loads(v) if v else None
        row["has_spec"] = r.get("has_spec") == "1"
        row["passed"] = None if row["sample_score"] is None else bool(row["sample_score"])
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = [r for r in rows if r["has_spec"] and r["num_spec_steps"]]
    report: dict[str, Any] = {
        "csv": str(csv_path),
        "coverage": {
            "model_calls": len(rows),
            "calls_with_spec": len(spec),
            "calls_without_spec": len(rows) - len(spec),
            "logs": len({r["log_file"] for r in rows}),
            "samples": len({(r["log_file"], r["sample_id"], r["epoch"]) for r in rows}),
            "tasks": len({r["sample_id"] for r in rows}),
        },
    }
    if not spec:
        out_dir.joinpath("report.json").write_text(json.dumps(report, indent=2))
        out_dir.joinpath("report.md").write_text(
            "# Speculative-decoding analysis\n\nNo call in these logs carried "
            "`metrics.speculative_decoding`. Check that the server ran with "
            "`--per-request-spec-decode-metrics detailed`, that speculative decoding was actually "
            "active, and that `run_eval.sh` passed `--log-model-api`.\n"
        )
        print("no spec metrics found; wrote a stub report")
        return 0

    # ---------- overall
    overall = Agg()
    for r in spec:
        overall.add(r)
    o = overall.result()
    report["overall"] = o

    mal_values = [r["mal"] for r in spec if r["mal"] is not None]
    report["mal_distribution"] = {
        "n": len(mal_values),
        "mean": statistics.fmean(mal_values) if mal_values else None,
        "median": statistics.median(mal_values) if mal_values else None,
        "p05": _pct(mal_values, 0.05),
        "p25": _pct(mal_values, 0.25),
        "p75": _pct(mal_values, 0.75),
        "p95": _pct(mal_values, 0.95),
    }

    # ---------- histogram consistency
    bad = [
        r
        for r in spec
        if r["acceptance_histogram"] and sum(r["acceptance_histogram"]) != r["num_spec_steps"]
    ]
    report["histogram_consistency"] = {"calls_checked": len(spec), "mismatches": len(bad)}

    # ---------- slice builders
    def slice_by(key_fn, out_name: str, title: str, order: list[str] | None = None) -> list[dict[str, Any]]:
        groups: dict[str, Agg] = defaultdict(Agg)
        for r in spec:
            k = key_fn(r)
            if k is not None:
                groups[str(k)].add(r)
        table = [{"slice": k, **v.result()} for k, v in groups.items()]
        if order:
            rank = {label: i for i, label in enumerate(order)}
            table.sort(key=lambda t: rank.get(t["slice"], len(rank)))
        else:
            table.sort(key=lambda t: _sort_key(t["slice"]))
        cols = [
            "slice",
            "calls",
            "num_spec_steps",
            "micro_mal",
            "macro_mal",
        ]
        _write_csv(out_dir / out_name, table, cols)
        report.setdefault("slices", {})[title] = table
        return table

    turn_table = slice_by(
        lambda r: _bucket(r["turn"], TURN_BINS),
        "mal_by_turn.csv",
        "mal_by_turn",
        [b for b, _, _ in TURN_BINS],
    )
    ctx_table = slice_by(
        lambda r: _bucket(
            None if r["input_tokens"] is None else r["input_tokens"] + (r["input_cache_read"] or 0),
            CTX_BINS,
        ),
        "mal_by_context.csv",
        "mal_by_context",
        [b for b, _, _ in CTX_BINS],
    )
    out_table = slice_by(
        lambda r: _bucket(r["output_tokens"], OUT_BINS),
        "mal_by_output.csv",
        "mal_by_output",
        [b for b, _, _ in OUT_BINS],
    )
    slice_by(lambda r: r["epoch"], "mal_by_epoch.csv", "mal_by_epoch")
    slice_by(lambda r: "pass" if r["passed"] else "fail", "mal_by_pass.csv", "mal_by_pass")
    slice_by(lambda r: r["stop_reason"] or "unknown", "mal_by_stop_reason.csv", "mal_by_stop_reason")

    # per-position acceptance, sliced
    pos_slices: dict[str, dict[str, list[float]]] = {}
    for name, keyf in [
        ("by_turn", lambda r: _bucket(r["turn"], TURN_BINS)),
        (
            "by_context",
            lambda r: _bucket(
                None if r["input_tokens"] is None else r["input_tokens"] + (r["input_cache_read"] or 0),
                CTX_BINS,
            ),
        ),
        ("by_output", lambda r: _bucket(r["output_tokens"], OUT_BINS)),
        ("by_pass", lambda r: "pass" if r["passed"] else "fail"),
        ("by_stop_reason", lambda r: r["stop_reason"] or "unknown"),
    ]:
        groups: dict[str, Agg] = defaultdict(Agg)
        for r in spec:
            k = keyf(r)
            if k is not None:
                groups[str(k)].add(r)
        pos_slices[name] = {k: v.result()["positions"] for k, v in groups.items()}

    # per-position CSV: overall and conditional acceptance per slice
    pos_rows: list[dict[str, Any]] = []
    for slice_name, groups in pos_slices.items():
        for label, positions in sorted(groups.items()):
            for p in positions:
                pos_rows.append(
                    {
                        "slice_family": slice_name,
                        "slice": label,
                        "position": p["position"],
                        "overall": p["overall"],
                        "conditional": p["conditional"],
                        "marginal": p["marginal"],
                    }
                )
    _write_csv(
        out_dir / "acceptance_by_position.csv",
        pos_rows,
        ["slice_family", "slice", "position", "overall", "conditional", "marginal"],
    )
    report["acceptance_by_position"] = pos_slices

    # per-position acceptance overall
    report["per_position_overall"] = o["positions"]

    conds = [p["conditional"] for p in o["positions"] if p["conditional"] is not None]
    report["headline"] = {
        "micro_mal": o["micro_mal"],
        "macro_mal": o["macro_mal"],
        "median_mal": report["mal_distribution"]["median"],
        "mean_conditional_acceptance": statistics.fmean(conds) if conds else None,
        "first_token_acceptance": o["positions"][0]["overall"] if o["positions"] else None,
    }

    # ---------- thinking vs response split
    # The completion is one contiguous token sequence. Each verify step i emits its a_i accepted draft
    # tokens followed by one always-accepted bonus token, so step i covers tokens [o_i, o_i + a_i + 1)
    # with o_0 = 0 (sum(a_i + 1) == completion_tokens, verified on real logs). The reasoning block is the
    # prefix of the completion, so its length in tokens -- usage.reasoning_tokens -- is the boundary.
    # No tokenizer and no extra vLLM /tokenize or /render call is needed: reasoning_tokens was verified
    # equal to len(tokenize(reasoning text)).
    regions: dict[str, dict[str, Any]] = {
        name: {"steps": 0, "accepted": 0, "hist": Counter()}
        for name in ("thinking", "response", "unknown")
    }
    exact_tokens = {"thinking": 0, "response": 0}
    straddling = 0
    for r in spec:
        ps = r["per_step_accepted"] or []
        total = r["output_tokens"] or 0
        b_raw = r["reasoning_tokens"]
        b = None if b_raw is None else max(0, min(int(b_raw), total))
        o_pos = 0
        for i, a in enumerate(ps):
            acc_start, acc_end = o_pos, o_pos + a
            # Attribute the step by the midpoint of its accepted tokens (its bonus token if a == 0).
            mid = acc_start + a / 2 if a else o_pos + a + 0.5
            region = "unknown" if b is None else ("thinking" if mid < b else "response")
            reg = regions[region]
            reg["steps"] += 1
            reg["hist"][a] += 1
            reg["accepted"] += a
            if b is not None:
                if acc_start < b < acc_end:
                    straddling += 1
                acc_think = max(0, min(acc_end, b) - acc_start)
                exact_tokens["thinking"] += acc_think
                exact_tokens["response"] += a - acc_think
            o_pos += a + 1

    by_region: dict[str, Any] = {}
    for name, reg in regions.items():
        if not reg["steps"]:
            continue
        by_region[name] = {
            "steps": reg["steps"],
            "accepted_draft_tokens": reg["accepted"],
            "micro_mal": 1 + reg["accepted"] / reg["steps"],
            "share_of_steps": reg["steps"] / o["num_spec_steps"] if o["num_spec_steps"] else None,
            "positions": _position_stats(reg["hist"], reg["steps"]),
        }
    report["by_region"] = by_region
    report["by_region_straddling_steps"] = straddling
    report["by_region_exact_accepted_tokens"] = exact_tokens

    # ---------- per task
    tasks: dict[str, Agg] = defaultdict(Agg)
    for r in spec:
        tasks[str(r["sample_id"])].add(r)
    task_table = [{"task": k, **v.result()} for k, v in tasks.items()]
    task_table.sort(key=lambda t: (t["micro_mal"] or 0))
    _write_csv(
        out_dir / "mal_by_task.csv",
        task_table,
        ["task", "calls", "num_spec_steps", "micro_mal", "macro_mal"],
    )
    report["per_task"] = task_table

    # ---------- outliers, two ways
    def filter_stats(subset: list[dict[str, Any]], label: str) -> dict[str, Any]:
        a = Agg()
        for r in subset:
            a.add(r)
        res = a.result()
        return {"variant": label, "calls": res.get("calls", 0), **{k: v for k, v in res.items() if k != "calls"}}

    out_vals = [r["output_tokens"] for r in spec if r["output_tokens"]]
    p95_calls = _pct(out_vals, 0.95)
    kept_calls = [r for r in spec if (r["output_tokens"] or 0) <= (p95_calls or 0)]

    sample_out: dict[tuple, int] = defaultdict(int)
    for r in spec:
        sample_out[(r["log_file"], r["sample_id"], r["epoch"])] += r["output_tokens"] or 0
    p95_samples = _pct(list(sample_out.values()), 0.95)
    kept_samples = [r for r in spec if sample_out[(r["log_file"], r["sample_id"], r["epoch"])] <= (p95_samples or 0)]

    report["outliers"] = {
        "thresholds": {"call_output_tokens_p95": p95_calls, "sample_output_tokens_p95": p95_samples},
        "variants": [
            filter_stats(spec, "all calls"),
            filter_stats(kept_calls, "drop calls with output_tokens > p95"),
            filter_stats(kept_samples, "drop whole samples with total output_tokens > p95"),
        ],
    }

    # ---------- within-response drift
    deciles: dict[int, Agg] = defaultdict(Agg)
    for r in spec:
        ps = r["per_step_accepted"]
        if not ps:
            continue
        n = len(ps)
        for i, acc in enumerate(ps):
            d = min(9, int(10 * i / n))
            deciles[d].steps += 1
            deciles[d].accepted += acc
    report["within_response_drift"] = [
        {
            "decile": d,
            "verify_steps": deciles[d].steps,
            "mal": 1 + deciles[d].accepted / deciles[d].steps if deciles[d].steps else None,
        }
        for d in sorted(deciles)
    ]

    # ---------- pass/fail
    pf: dict[str, Agg] = defaultdict(Agg)
    for r in spec:
        pf["pass" if r["passed"] else "fail"].add(r)
    report["pass_vs_fail"] = {
        k: {
            "calls": v.result().get("calls", 0),
            "micro_mal": v.result().get("micro_mal"),
        }
        for k, v in sorted(pf.items())
    }

    out_dir.joinpath("report.json").write_text(json.dumps(report, indent=2))
    out_dir.joinpath("report.md").write_text(_render(report))
    print(f"wrote {out_dir}/report.md, report.json and slice CSVs")
    return 0


def _sort_key(label: str) -> tuple:
    """Natural sort so '2' < '10' and numeric prefixes order before text."""
    import re as _re

    return tuple(int(p) if p.isdigit() else p.lower() for p in _re.split(r"(\d+)", label))


def _write_csv(path: Path, table: list[dict[str, Any]], cols: list[str]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in table:
            w.writerow(
                {
                    k: (json.dumps(row[k], separators=(",", ":")) if isinstance(row.get(k), list) else row.get(k, ""))
                    for k in cols
                }
            )


def _render(rep: dict[str, Any]) -> str:
    cov = rep["coverage"]
    o = rep["overall"]
    d = rep.get("mal_distribution", {})
    L: list[str] = []
    L.append("# Speculative-decoding analysis\n")
    L.append("## Coverage\n")
    L.append(f"- Logs: {cov['logs']}, samples: {cov['samples']}, tasks: {cov['tasks']}")
    L.append(f"- Model calls: {cov['model_calls']}, with spec metrics: {cov['calls_with_spec']}, "
             f"without: {cov['calls_without_spec']}")
    hc = rep.get("histogram_consistency") or {}
    if hc:
        L.append(f"- Histogram/step-count consistency: {hc['mismatches']} mismatch(es) over "
                 f"{hc['calls_checked']} calls")
    L.append("")
    L.append("## Headline\n")
    L.append("| metric | value |")
    L.append("|---|---|")
    h = rep.get("headline") or {}
    L.append(f"| micro MAL (1 + accepted/steps) | {_fmt(h.get('micro_mal'))} |")
    L.append(f"| mean first-token acceptance rate | {_fmt(h.get('first_token_acceptance'))} |")
    L.append(f"| macro MAL (unweighted mean of per-call values) | {_fmt(h.get('macro_mal'))} |")
    L.append(f"| median per-call MAL | {_fmt(h.get('median_mal'))} |")
    L.append(f"| mean conditional acceptance rate | {_fmt(h.get('mean_conditional_acceptance'))} |")
    L.append("")
    L.append("Micro MAL weights every verify step equally, so long responses dominate it. Macro MAL is the "
             "unweighted mean of the per-call values, so every request carries the same weight. The two "
             "differ when call length correlates with acceptance.\n")
    L.append("## Distribution of per-call MAL\n")
    L.append("| n | mean | median | p05 | p25 | p75 | p95 |")
    L.append("|---|---|---|---|---|---|---|")
    L.append(
        f"| {d.get('n')} | {_fmt(d.get('mean'))} | {_fmt(d.get('median'))} | {_fmt(d.get('p05'))} | "
        f"{_fmt(d.get('p25'))} | {_fmt(d.get('p75'))} | {_fmt(d.get('p95'))} |"
    )
    L.append("")
    L.append("## Per-position acceptance (overall)\n")
    L.append("`overall` = P(the draft token at this position is accepted) = P(verify step accepted more than "
             "j drafts); `conditional` = P(position j accepted | position j-1 accepted) = overall[j] / "
             "overall[j-1], with position 0 conditioned on nothing. The conditional column is the per-position "
             "hazard rate: a flat column means acceptance is geometric in position, and overall[j] is the running "
             "product of the conditionals. The 'exactly j' marginal probabilities are in `report.json` and "
             "`acceptance_by_position.csv`.\n")
    L.append("| position | overall acceptance | conditional acceptance |")
    L.append("|---|---|---|")
    for p in rep.get("per_position_overall", []):
        L.append(f"| {p['position']} | {_fmt(p['overall'])} | {_fmt(p['conditional'])} |")
    sums = sum(p["overall"] for p in rep.get("per_position_overall", []))
    L.append(f"| **Σ overall + 1 = MAL** | {_fmt(1 + sums)} | |")
    L.append("")

    def table(title: str, table: list[dict[str, Any]], label: str) -> None:
        if not table:
            return
        L.append(f"## {title}\n")
        L.append(f"| {label} | calls | steps | micro MAL | macro MAL |")
        L.append("|---|---|---|---|---|")
        for t in table:
            L.append(
                f"| {t['slice']} | {t['calls']} | {t['num_spec_steps']} | {_fmt(t['micro_mal'])} | "
                f"{_fmt(t['macro_mal'])} |"
            )
        L.append("")

    L.append("## Acceptance by region: thinking vs response\n")
    L.append(
        "The reasoning block is the prefix of the completion, so its length in completion tokens "
        "(`usage.reasoning_tokens`, verified equal to the token count of the reasoning text) is the boundary. "
        "Each verify step's token span follows from its accepted-draft count. A step is attributed to the "
        "region that holds the midpoint of its accepted draft tokens. "
        f"{rep.get('by_region_straddling_steps', 0)} step(s) straddled the boundary. No tokenizer is needed, "
        "because the boundary and the step spans are both in the logged response.\n"
    )
    br = rep.get("by_region") or {}
    if br:
        L.append("| region | verify steps | share of steps | MAL | accepted drafts |")
        L.append("|---|---|---|---|---|")
        for name in ("thinking", "response", "unknown"):
            v = br.get(name)
            if not v:
                continue
            L.append(
                f"| {name} | {v['steps']} | {_fmt(v.get('share_of_steps'), 4)} | {_fmt(v['micro_mal'])} | "
                f"{v['accepted_draft_tokens']} |"
            )
        L.append("")
        ex = rep.get("by_region_exact_accepted_tokens") or {}
        if ex:
            L.append(
                f"Accepted draft tokens split exactly by token position: thinking {ex.get('thinking', 0)}, "
                f"response {ex.get('response', 0)}.\n"
            )
        think = br.get("thinking", {}).get("positions") or []
        resp = br.get("response", {}).get("positions") or []
        if think or resp:
            L.append("| position | thinking overall | thinking conditional | response overall | response conditional |")
            L.append("|---|---|---|---|---|")
            for j in range(max(len(think), len(resp))):
                t = think[j] if j < len(think) else None
                rr = resp[j] if j < len(resp) else None
                L.append(
                    f"| {j} | {_fmt(t['overall']) if t else 'n/a'} | {_fmt(t['conditional']) if t else 'n/a'} | "
                    f"{_fmt(rr['overall']) if rr else 'n/a'} | {_fmt(rr['conditional']) if rr else 'n/a'} |"
                )
            L.append("")
            if think:
                L.append(
                    f"thinking: Σ overall + 1 = {_fmt(1 + sum(p['overall'] for p in think))} "
                    f"(MAL {_fmt(br['thinking']['micro_mal'])})"
                )
            if resp:
                L.append(
                    f"response: Σ overall + 1 = {_fmt(1 + sum(p['overall'] for p in resp))} "
                    f"(MAL {_fmt(br['response']['micro_mal'])})"
                )
            L.append("")

    table("MAL by turn number", rep.get("slices", {}).get("mal_by_turn", []), "turns")
    table("MAL by context length (prompt tokens)", rep.get("slices", {}).get("mal_by_context", []), "tokens")
    table("MAL by output length", rep.get("slices", {}).get("mal_by_output", []), "tokens")
    table("MAL by epoch", rep.get("slices", {}).get("mal_by_epoch", []), "epoch")
    table("MAL pass vs fail", rep.get("slices", {}).get("mal_by_pass", []), "verdict")
    table("MAL by stop reason", rep.get("slices", {}).get("mal_by_stop_reason", []), "stop_reason")

    out = rep.get("outliers")
    if out:
        L.append("## Outlier removal\n")
        L.append(
            f"Thresholds: call output tokens p95 = {_fmt(out['thresholds']['call_output_tokens_p95'], 0)}, "
            f"sample total output tokens p95 = {_fmt(out['thresholds']['sample_output_tokens_p95'], 0)}.\n"
        )
        L.append("| variant | calls | steps | micro MAL | macro MAL |")
        L.append("|---|---|---|---|---|")
        for v in out["variants"]:
            L.append(
                f"| {v['variant']} | {v['calls']} | {v.get('num_spec_steps')} | {_fmt(v.get('micro_mal'))} | "
                f"{_fmt(v.get('macro_mal'))} |"
            )
        L.append("")

    drift = rep.get("within_response_drift") or []
    if drift:
        L.append("## Within-response drift\n")
        L.append("MAL by decile of position within the response (MAL = 1 + accepted drafts / verify steps).\n")
        L.append("| decile | verify steps | MAL |")
        L.append("|---|---|---|")
        for d_ in drift:
            L.append(f"| {d_['decile']} | {d_['verify_steps']} | {_fmt(d_['mal'])} |")
        L.append("")

    ps = rep.get("pass_vs_fail") or {}
    if ps:
        L.append("## Pass vs fail\n")
        L.append("| verdict | calls | micro MAL |")
        L.append("|---|---|---|")
        for k, v in ps.items():
            L.append(f"| {k} | {v['calls']} | {_fmt(v.get('micro_mal'))} |")
        L.append("")

    pt = rep.get("per_task") or []
    if pt:
        L.append("## Per-task MAL (20 lowest)\n")
        L.append("| task | calls | micro MAL | macro MAL |")
        L.append("|---|---|---|---|")
        for t in pt[:20]:
            L.append(
                f"| {t['task']} | {t['calls']} | {_fmt(t['micro_mal'])} | {_fmt(t['macro_mal'])} |"
            )
        L.append("")
        L.append(f"Full table: `mal_by_task.csv`. Positions per slice: `acceptance_by_position.csv`.\n")

    return "\n".join(L)


# --------------------------------------------------------------------------- main


def main() -> int:
    argv = sys.argv[1:]

    if argv and argv[0] in ("collate", "analyze"):
        if argv[0] == "collate":
            p = argparse.ArgumentParser(
                prog="analyze_spec_decode.py collate",
                description="Read every *.eval under LOG_DIR and write one CSV row per model API call.",
            )
            p.add_argument("log_dir", type=Path)
            p.add_argument("-o", "--out", type=Path, default=Path("calls.csv"))
            a = p.parse_args(argv[1:])
            return collate(a.log_dir, a.out)
        p = argparse.ArgumentParser(
            prog="analyze_spec_decode.py analyze",
            description="Analyse a collated CSV into report.md/report.json and slice CSVs.",
        )
        p.add_argument("csv", type=Path)
        p.add_argument("-o", "--out-dir", type=Path, default=Path("spec_decode_report"))
        a = p.parse_args(argv[1:])
        return analyze(a.csv, a.out_dir)

    # No subcommand: LOG_DIR in, report out (collate then analyze).
    p = argparse.ArgumentParser(
        prog="analyze_spec_decode.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("log_dir", type=Path, nargs="?", help="directory to search recursively for *.eval")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("spec_decode_report"))
    a = p.parse_args(argv)
    if not a.log_dir:
        p.print_help()
        return 2
    tmp = a.out_dir / "calls.csv"
    rc = collate(a.log_dir, tmp)
    if rc:
        return rc
    return analyze(tmp, a.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
