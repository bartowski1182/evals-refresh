# RedHatAI/GLM-5.3-MXFP4xFP8 — DeepSWE 1.1

| | |
|---|---|
| Task | `inspect_harbor/datacurve_deep_swe_1_1` (113 tasks, Harbor hub package digest `sha256:5affcd53…`, loaded through `assets/harbor_registry.json`, which pins the tasks to the fork commit carrying the verifier fix) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per task (public.ecr.aws, 2 CPUs / 8 GB requested), bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3 (base model; its Hopper section is identical to this checkpoint's model-card command — the recipes site has no MXFP4xFP8 entry) |
| Task source | https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1 |

**Protocol note.** The zai-org/GLM-5.3 card reports 66.9 on DeepSWE v1.1 with the mini-swe-agent harness
(temp 0.95, top_p 1.0, 6 h timeout, 400K context), measured on the FP8 `zai-org/GLM-5.3` checkpoint. This run uses
Inspect's generic ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context
compaction; no message limit), the checkpoint's own sampling defaults (temp 1.0, top_p 0.95), reasoning effort
`max`, and a single `--time-limit 14400` wall-clock guard with `override_cpus=0`, one epoch (pass@1), so the numbers
are not comparable to the card. The container runs without network, as every DeepSWE task declares for its agent and
verifier (stock inspect_harbor 1.0 ignored that and ran with internet; see Patches). Checkpoint: mixed MXFP4 (routed
experts and dense MLPs, W4A8 with per-group dynamic FP8 activations) / FP8 (attention, shared experts, MTP layer),
compressed-tensors format (LLM Compressor), ~4.5 bits per parameter, revision `13c63cd`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, KV-cache dtype,
  MTP speculative decoding, reasoning/tool parsers), taken from the model card's serving command (= the base
  recipe's single-node Hopper section), with one documented departure: `--max-model-len 262144` instead of the
  checkpoint's 1M default, because the DSA sparse-indexer warm-up OOMs on H200 at 1M (see the comment in the file).
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and dataset pin, epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
  Concurrency is 10 in flight: the model's ~1.19M-token KV pool holds about ten 120k-token DeepSWE histories
  without evicting prefix-cached histories (see the comment in the file).
- `assets/harbor_registry.json` — Harbor registry entry with every task pinned to the fork commit carrying the
  dataset fixes (see Patches).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs
  (all 8, port 8000, one TP=8 replica so each sample's prefix cache stays on one replica), the HF cache and the
  Docker socket, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below (Python 3.12,
host details in their headers).

```bash
# server: vLLM 0.32.0 (the first release that serves this MXFP4/FP8 checkpoint on Hopper, per the model card)
python3.12 -m venv /data/playground/dsikka/venvs/vllm-glm53 && source /data/playground/dsikka/venvs/vllm-glm53/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.32.0
# client: Inspect + the inspect_harbor fork that enforces per-phase network policy (see Patches)
python3.12 -m venv /data/playground/dsikka/venvs/inspect-harbor-fix && source /data/playground/dsikka/venvs/inspect-harbor-fix/bin/activate
pip install -r client_requirements.txt          # or:
# pip install httpx2 "inspect_harbor @ git+https://github.com/eldarkurtic/inspect_harbor@cb6192598173009fd637e12cf5ec603c3c341359" && pip install "openai==3.14.0"
```

Model weights are pre-downloaded into the HF cache (`HF_HOME=/data/playground/dsikka/hf` in
`local_orchestrator.sh`; about 415 GB) and served with `HF_HUB_OFFLINE=1`. Nothing here is gated; export `HF_TOKEN`
in your shell before downloading (never in the repo) for higher Hub rate limits. The Harbor task files are fetched
from GitHub at the pinned commit on first use.

Docker with the compose plugin (v2.x or later) and buildx >= 0.17 is required. The results below were produced with
a rootless Docker daemon (containerd image store on `/data/playground/dsikka`, 512 network pools, runc with
`NoNewKeyring`); `local_orchestrator.sh` points `DOCKER_HOST` at its socket. Inspect builds a task's image before
its first sample if it is missing (serially, before any sample starts). If you pre-built the `hb__<hash>` images,
pass `--sandbox-prebuilt` to skip that step. The 113 task images (about 122 GB) come from ECR public
(`public.ecr.aws/d3j8x8q7/swe-bench-202605`), which throttles bursts; pull them ahead of time, one at a time.

### Run the eval

```bash
./local_orchestrator.sh                  # add --sandbox-prebuilt if the task images are pre-built
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

With model generation config:
`max_connections=10, max_tokens=65536, top_p=0.95, temperature=1.0, reasoning_effort='max'`
(one epoch = pass@1, `--time-limit 14400`, `-M client_timeout=7200`, 10 concurrent samples)
the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (pass@1) | **0.673 (76 of 113)**, all-samples denominator |
| stderr | 0.044 |
| sample errors | 0 |
| time-limit hits (14400 s) | 0 |
| per-turn `max_tokens` truncations | 1 (no final-turn truncations) |
| wall time | 8 h 45 m at 10 concurrent |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 15,900,664 | 142,451 | 100,697 |
| median | 13,322,346 | 132,696 | 93,758 |
| max | 74,704,285 | 325,612 | 255,679 |

Run totals: 1,796,775,002 input / 16,097,005 output / 11,378,795 reasoning tokens. Input is
cumulative over the whole session (the ReAct loop resends the full history every turn), not
peak context; histories stayed inside the 262k window via automatic compaction. Reasoning is
counted separately from output.

Leak audit (repo §11): all 113 trajectories scanned tool call by tool call. 35 outbound network
attempts across the run, every one failed with DNS resolution errors (containers had no route
out — the per-phase network fix held for the whole run); no task files, upstream history or
gold tests were accessible. No leaked passes; the score is clean.

Failed tasks (37 of 113): actionlint-action-pinning-lint, arktype-json-schema-refs-dependencies,
awilix-async-container-initialization, boa-hierarchical-evaluation-cancellation,
clack-async-autocomplete-options, claude-code-by-agents-recursive-delegation,
csstree-shorthand-expansion-compression, dateutil-rfc5545-timezone-interop,
effect-sse-httpapi-streaming, etree-xml-diff-patch, fd-deterministic-multi-key-sorting,
go-git-worktree-merge-conflicts, happy-dom-deterministic-intersectionobserver,
httpx-streaming-json-iteration, igel-persist-feature-schema, ink-grid-box-layout,
katex-multicolumn-array-spans, kea-atomic-signal-selectors, kombu-virtual-queue-dead-lettering,
koota-entity-snapshot-rollback, kysely-window-grouping-helpers,
meriyah-explicit-resource-declarations, obsidian-linter-auto-table-of-contents,
obsidian-linter-link-format-conversion, onedump-dump-encryption-pipeline,
oxvg-structural-selector-preservation, pebble-durability-wait-apis,
pest-character-class-coalescing, scc-bounded-memory-spilling, sqlfmt-create-table-ddl-formatting,
superjson-error-stack-serialization, termenv-preserve-ansi-resets, ts-pattern-match-each,
updo-policy-alerting, vulture-persistent-analysis-cache, wasmi-trap-coredumps,
wazero-multi-module-snapshots (all `datacurve/`).

## Patches for upstream

- **The task's own network policy was not enforced.** Every DeepSWE `task.toml` declares `[agent]` and `[verifier]`
  `network_mode = "no-network"`, but inspect_harbor 1.0 read network policy only from `[environment]`, so agents had
  public internet. Fixed on branch [`fix/enforce-agent-network-mode`](https://github.com/eldarkurtic/inspect_harbor/tree/fix/enforce-agent-network-mode)
  of https://github.com/eldarkurtic/inspect_harbor (commit `cb61925`, on top of v1.0.0), which applies `[agent]` and
  `[verifier]` `network_mode` per phase; with both denied, the container runs with `network_mode: none` throughout.
  The client venv installs inspect_harbor from that branch; the sample containers were checked to have no route out.
  Upstream PR: [meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200).
- **Git history is clean.** The v1.1 task images (`public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`) are built from
  the tasks' `environment/Dockerfile`, which already keeps only the base commit's history. All 113 were checked: every
  ref reachable from `HEAD`, no unreachable commits. The v1.0 images do keep future `origin/*` refs, but these tasks
  do not use them.
- Carried over from the Qwen3.8-27B DeepSWE dir and consumed here: the `[[verifier.collect]]` hook (inspect_harbor
  0.7.6, meridianlabs-ai/inspect_harbor#155), the vitest worker-pool OOM fixed with `taskset` in the pinned fork
  commit (datacurve-ai/deep-swe#100), and the verifier's non-login shell (inspect_harbor 1.0).
