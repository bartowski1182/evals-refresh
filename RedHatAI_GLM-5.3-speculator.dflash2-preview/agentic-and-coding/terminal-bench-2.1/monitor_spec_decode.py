#!/usr/bin/env python3
"""Detect the silent DFlash/DFlash2 acceptance-collapse bug (vLLM #63587) for this run.

Background
----------
On vLLM 0.32.0/0.33.0, a DFlash/DFlash2 draft served with the default FULL_AND_PIECEWISE CUDA graphs can
silently stop accepting draft tokens. Output text stays correct, so pass/fail scores are unaffected, but:

  * a call can be dead from the first verify step (MAL ~ 1.00), or accept for a while and then stop for the
    rest of the call (a trailing run of zero-accepted steps) with no recovery;
  * it concentrates in long or max_tokens-capped generations;
  * it costs ~3x wall clock, because the request falls back to one token per forward pass while still
    drafting N per step;
  * the pooled acceptance numbers move a lot: one collapsed call emits ~1 token per step against ~4.4 for a
    healthy one, so collapsed calls can dominate the pooled step count and drag micro MAL from ~3.5 to ~2.4
    while macro MAL (per-call mean) stays ~4.2.

The fix is PR #64059 (merged 2026-11-04, in no release tag as of 2026-11-11). `serve.sh` avoids the bug by
serving with `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`, which keeps the draft out of the in-graph
decode-metadata path. This script is the independent audit that the workaround held.

Subcommands
-----------
canary  Send periodic capped-length chat completions to the running server and watch each response's
        `metrics.speculative_decoding` payload. Writes one JSON line per probe, and a flag file as soon as a
        probe looks collapsed. Cheap enough to run beside the eval (one request per interval).
audit   Post-process the per-call CSV produced by `analyze_spec_decode.py collate` and quantify collapsed
        calls: counts, step share, micro/macro MAL with and without them, per-position rates, top offenders,
        and `collapsed_calls.csv`. Use this to decide whether the reported acceptance numbers are trustworthy
        or need the collapsed calls excluded/flagged.

Both subcommands define a call as collapsed when either
  * MAL <= 1.02 with at least 4 verify steps (dead from step 0 or nearly), or
  * a trailing run of >= 8 zero-accepted steps that is also >= 30% of the call, on a call of >= 20
    verify steps and with the pre-trailing part of the call accepting above 10% (it was working, then died).
The thresholds are calibrated against healthy runs: across 52 calls of a healthy GLM-5.3 + dflash2 run the
trailing zero-accept run was at most 5 steps and at most 14% of a call, so a run of 8 covering 30% is far
outside normal end-of-call noise. A short or cap-truncated call cannot trigger either rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

COLLAPSE_MAL = 1.02          # MAL at/below this is "no draft token ever accepted"
COLLAPSE_MIN_STEPS = 4       # ... but only if the call actually ran a few verify steps
TRAILING_ZERO_RUN = 8        # a trailing run this long is the mid-call death signature
TRAILING_MIN_STEPS = 20      # ... only on calls long enough for "the rest of the call" to mean something
TRAILING_MIN_FRACTION = 0.30 # ... and only if it is this much of the call
TRAILING_PRE_ACCEPT = 0.10   # ... and the call was accepting before it died

CANARY_PROMPT = (
    "Write a complete, production-quality Python implementation of an AVL tree with insert, delete, search "
    "and range-query, followed by an exhaustive pytest suite covering every rotation case. Output code only, "
    "with no prose, comments or explanation."
)

# --------------------------------------------------------------------------- shared


def trailing_zeros(per_step_accepted: list[int] | None) -> int:
    """Number of consecutive zero-accepted verify steps at the end of a call."""
    if not per_step_accepted:
        return 0
    n = 0
    for v in reversed(per_step_accepted):
        if v:
            break
        n += 1
    return n


def classify(mal: float | None, steps: int, per_step: list[int] | None) -> str:
    """Return "", "collapsed" or "trailing_zero_run" for one call."""
    if mal is not None and steps >= COLLAPSE_MIN_STEPS and mal <= COLLAPSE_MAL:
        return "collapsed"
    tz = trailing_zeros(per_step)
    if (
        per_step
        and steps >= TRAILING_MIN_STEPS
        and tz >= TRAILING_ZERO_RUN
        and tz >= TRAILING_MIN_FRACTION * steps
    ):
        head = per_step[: steps - tz]
        # The call must have been working before it died: enough of the pre-trailing steps accepted something.
        if head and sum(1 for v in head if v) / len(head) >= TRAILING_PRE_ACCEPT:
            return "trailing_zero_run"
    return ""


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _json_list(v: Any) -> list[int] | None:
    if isinstance(v, list):
        return [int(x) for x in v]
    if isinstance(v, str) and v:
        try:
            out = json.loads(v)
        except json.JSONDecodeError:
            return None
        return [int(x) for x in out] if isinstance(out, list) else None
    return None


# --------------------------------------------------------------------------- canary


def _post_chat(base_url: str, model: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": CANARY_PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "reasoning_effort": "max",
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def canary(args: argparse.Namespace) -> int:
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path = Path(args.flag)
    stop_path = Path(args.stop_file) if args.stop_file else None
    healthy: list[float] = []
    n = 0
    print(
        f"canary: probing {args.base_url} model={args.model} every {args.interval}s, "
        f"max_tokens={args.max_tokens}; log={log_path}",
        flush=True,
    )
    while True:
        if stop_path is not None and stop_path.exists():
            print("canary: stop file present, exiting", flush=True)
            return 0
        if args.max_requests and n >= args.max_requests:
            print(f"canary: reached {n} requests, exiting", flush=True)
            return 0
        n += 1
        t0 = time.time()
        rec: dict[str, Any] = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "probe": n}
        try:
            resp = _post_chat(args.base_url, args.model, args.max_tokens, args.timeout)
            elapsed = time.time() - t0
            choice = (resp.get("choices") or [{}])[0]
            usage = resp.get("usage") or {}
            spec = ((resp.get("metrics") or {}).get("speculative_decoding")) or {}
            per_step = _json_list(spec.get("per_step_accepted"))
            mal = _num(spec.get("mean_acceptance_length"))
            steps = int(spec.get("num_spec_steps") or (len(per_step) if per_step else 0))
            ctoks = usage.get("completion_tokens")
            rec.update(
                {
                    "ok": True,
                    "finish_reason": choice.get("finish_reason"),
                    "latency_s": round(elapsed, 2),
                    "completion_tokens": ctoks,
                    "output_tps": round(ctoks / elapsed, 2) if ctoks and elapsed > 0 else None,
                    "mal": mal,
                    "num_spec_steps": steps,
                    "num_accepted_draft_tokens": spec.get("num_accepted_draft_tokens"),
                    "num_draft_tokens": spec.get("num_draft_tokens"),
                    "per_step_accepted": per_step,
                    "trailing_zero_run": trailing_zeros(per_step),
                }
            )
            kind = classify(mal, steps, per_step)
            if kind:
                rec["collapse"] = kind
            elif mal is not None and len(healthy) >= 5:
                med = statistics.median(healthy)
                if mal < 0.6 * med:
                    rec["collapse"] = "mal_drop_vs_baseline"
                    rec["baseline_median_mal"] = round(med, 3)
            if not rec.get("collapse") and mal is not None and steps >= TRAILING_MIN_STEPS:
                healthy.append(mal)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        if rec.get("collapse"):
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(flag_path, "a") as fh:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
            print(
                "COLLAPSE DETECTED "
                f"({rec['collapse']}): mal={rec.get('mal')} steps={rec.get('num_spec_steps')} "
                f"trailing_zeros={rec.get('trailing_zero_run')} -> {flag_path}",
                flush=True,
            )
        else:
            print(
                f"canary {n}: ok={rec.get('ok')} mal={rec.get('mal')} steps={rec.get('num_spec_steps')} "
                f"tps={rec.get('output_tps')}",
                flush=True,
            )
        # Sleep in short slices so a stop file is honoured promptly.
        slept = 0.0
        while slept < args.interval:
            if stop_path is not None and stop_path.exists():
                break
            time.sleep(min(5.0, args.interval - slept))
            slept += 5.0


# --------------------------------------------------------------------------- audit


def _micro_mal(rows: list[dict[str, Any]]) -> float | None:
    steps = sum(int(_num(r.get("num_spec_steps")) or 0) for r in rows)
    acc = sum(int(_num(r.get("num_accepted_draft_tokens")) or 0) for r in rows)
    return 1.0 + acc / steps if steps else None


def _macro_mal(rows: list[dict[str, Any]]) -> float | None:
    mals = [_num(r.get("mal")) for r in rows]
    mals = [m for m in mals if m is not None]
    return statistics.fmean(mals) if mals else None


def _positions(rows: list[dict[str, Any]], n_pos: int) -> list[float] | None:
    """Overall per-position acceptance from pooled histograms."""
    steps = sum(int(_num(r.get("num_spec_steps")) or 0) for r in rows)
    if not steps:
        return None
    hist = [0] * (n_pos + 1)
    for r in rows:
        h = _json_list(r.get("acceptance_histogram"))
        if not h:
            continue
        for i, v in enumerate(h[: n_pos + 1]):
            hist[i] += int(v)
    # hist[i] = number of steps that accepted exactly i draft tokens
    out = []
    for j in range(n_pos):
        out.append(sum(hist[j + 1 :]) / steps)
    return out


def audit(args: argparse.Namespace) -> int:
    src = Path(args.calls_csv)
    if not src.exists():
        sys.exit(f"no such CSV: {src}")
    rows: list[dict[str, Any]] = []
    with open(src, newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("has_spec") or "0") != "1":
                continue
            per_step = _json_list(r.get("per_step_accepted"))
            steps = int(_num(r.get("num_spec_steps")) or (len(per_step) if per_step else 0))
            r["_steps"] = steps
            r["_trailing"] = trailing_zeros(per_step)
            r["_collapse"] = classify(_num(r.get("mal")), steps, per_step)
            rows.append(r)
    if not rows:
        sys.exit(f"no rows with spec metrics in {src}")

    bad = [r for r in rows if r["_collapse"]]
    good = [r for r in rows if not r["_collapse"]]
    total_steps = sum(r["_steps"] for r in rows)
    bad_steps = sum(r["_steps"] for r in bad)
    # Positions come from the draft width (num_spec_tokens / drafted-per-step), NOT from the number of steps.
    n_pos = 0
    for r in rows:
        n_pos = max(n_pos, int(_num(r.get("num_spec_tokens")) or 0))
        hist = _json_list(r.get("acceptance_histogram"))
        if hist:
            n_pos = max(n_pos, len(hist) - 1)
        drafted = _json_list(r.get("per_step_drafted"))
        if drafted:
            n_pos = max(n_pos, max(drafted))
    n_pos = min(n_pos, 32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-call rows for the collapsed calls, for eyeballing.
    if bad:
        cols = [
            "log_file", "sample_id", "epoch", "turn", "stop_reason", "output_tokens",
            "reasoning_tokens", "mal", "num_spec_steps", "num_accepted_draft_tokens",
            "num_draft_tokens",
        ]
        with open(out_dir / "collapsed_calls.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols + ["trailing_zero_run", "kind"], extrasaction="ignore")
            w.writeheader()
            for r in bad:
                d = {k: r.get(k) for k in cols}
                d["trailing_zero_run"] = r["_trailing"]
                d["kind"] = r["_collapse"]
                w.writerow(d)

    def fmt(v: float | None, nd: int = 3) -> str:
        return "n/a" if v is None else f"{v:.{nd}f}"

    lines = []
    lines.append("# DFlash2 acceptance-collapse audit (vLLM #63587)\n")
    lines.append(f"Source: `{src}` — {len(rows)} calls carrying `metrics.speculative_decoding`.\n")
    verdict = (
        "**No collapsed calls found.** The acceptance numbers in `report.md` are trustworthy as reported; "
        "the build in `serve.sh` (vLLM main, which contains the #64059 fix) held."
        if not bad
        else (
            f"**{len(bad)} collapsed call(s) found** ({100 * len(bad) / len(rows):.2f}% of calls, "
            f"{100 * bad_steps / total_steps:.2f}% of verify steps). Pooled/micro numbers below are reported "
            "with and without them; the run's headline acceptance should be the *excluded* column, with the "
            "bug stated explicitly, unless the run was re-done with the fix."
        )
    )
    lines.append(verdict + "\n")
    lines.append("| metric | all calls | excluding collapsed |")
    lines.append("|---|---|---|")
    lines.append(f"| calls | {len(rows)} | {len(good)} |")
    lines.append(f"| verify steps | {total_steps} | {total_steps - bad_steps} |")
    lines.append(f"| micro MAL (1 + accepted/steps) | {fmt(_micro_mal(rows))} | {fmt(_micro_mal(good))} |")
    lines.append(f"| macro MAL (mean of per-call) | {fmt(_macro_mal(rows))} | {fmt(_macro_mal(good))} |")
    lines.append(f"| collapsed calls | {len(bad)} | 0 |")
    lines.append(f"| calls with trailing zero run >= {TRAILING_ZERO_RUN} | "
                 f"{sum(1 for r in rows if r['_trailing'] >= TRAILING_ZERO_RUN and r['_steps'] >= TRAILING_MIN_STEPS)} | 0 |")
    lines.append("")

    if n_pos:
        p_all = _positions(rows, n_pos)
        p_good = _positions(good, n_pos) if good else None
        if p_all:
            lines.append(f"| position | overall acceptance (all) | overall acceptance (excluding collapsed) |")
            lines.append("|---|---|---|")
            for j in range(n_pos):
                lines.append(
                    f"| {j} | {p_all[j]:.4f} | "
                    f"{('n/a' if p_good is None else f'{p_good[j]:.4f}')} |"
                )
            lines.append("")

    if bad:
        lines.append("## Largest collapsed calls by verify steps\n")
        lines.append("| sample | turn | steps | accepted | MAL | trailing zeros | kind |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in sorted(bad, key=lambda r: -r["_steps"])[:15]:
            lines.append(
                f"| {r.get('sample_id')} | {r.get('turn')} | {r['_steps']} | "
                f"{r.get('num_accepted_draft_tokens')} | {fmt(_num(r.get('mal')))} | "
                f"{r['_trailing']} | {r['_collapse']} |"
            )
        lines.append("")
        called_out = sorted({r.get("sample_id") for r in bad})
        lines.append(f"Affected samples ({len(called_out)}): " + ", ".join(f"`{s}`" for s in called_out) + "\n")

    (out_dir / "collapse_audit.md").write_text("\n".join(lines))
    (out_dir / "collapse_audit.json").write_text(
        json.dumps(
            {
                "source": str(src),
                "calls": len(rows),
                "steps": total_steps,
                "collapsed_calls": len(bad),
                "collapsed_steps": bad_steps,
                "micro_mal_all": _micro_mal(rows),
                "micro_mal_excluding_collapsed": _micro_mal(good),
                "macro_mal_all": _macro_mal(rows),
                "macro_mal_excluding_collapsed": _macro_mal(good),
                "position_acceptance_all": _positions(rows, n_pos),
                "position_acceptance_excluding_collapsed": _positions(good, n_pos) if good else None,
            },
            indent=2,
        )
    )
    print("\n".join(lines))
    print(f"wrote {out_dir/'collapse_audit.md'}, {out_dir/'collapse_audit.json'}"
          + (f", {out_dir/'collapsed_calls.csv'}" if bad else ""))
    return 0


# --------------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("canary", help="probe a running server for the acceptance-collapse bug")
    c.add_argument("--base-url", default="http://localhost:8000/v1")
    c.add_argument("--model", required=True, help="served model name (serve.sh's --served-model-name)")
    c.add_argument("--interval", type=float, default=120.0)
    c.add_argument("--max-tokens", type=int, default=1024)
    c.add_argument("--timeout", type=float, default=1800.0)
    c.add_argument("--log", default="logs/canary.jsonl")
    c.add_argument("--flag", default="logs/collapse_detected.jsonl")
    c.add_argument("--stop-file", default="logs/canary.stop")
    c.add_argument("--max-requests", type=int, default=0, help="0 = run until the stop file appears")

    a = sub.add_parser("audit", help="quantify collapsed calls in an analyze_spec_decode.py calls.csv")
    a.add_argument("calls_csv", help="CSV from `analyze_spec_decode.py collate`")
    a.add_argument("-o", "--out", default="spec_decode_report")

    args = ap.parse_args()
    return canary(args) if args.cmd == "canary" else audit(args)


if __name__ == "__main__":
    sys.exit(main())
