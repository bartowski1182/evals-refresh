# RedHatAI/Qwen3.8-Flash-Next-NVFP4 — DeepSWE 1.1

| | |
|---|---|
| Task | `inspect_harbor/datacurve_deep_swe_1_1` (113 tasks, Harbor hub package digest `sha256:5affcd53…`, loaded through `assets/harbor_registry.json`, which pins the tasks to the fork commit carrying the verifier fix) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per task (public.ecr.aws, 2 CPUs / 8 GB requested), bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |
| Task source | https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1 |

**Protocol note.** The Qwen/Qwen3.8-Flash-Next card reports 58.7 (best of the Claude Code and mini-SWE-agent
harnesses, temp 1.0, top_p 0.95, 256K context). `inspect_harbor` solves every task with Inspect's generic ReAct
agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction; no message
limit), not those harnesses; a single `--time-limit 14400` wall-clock guard is used, `override_cpus=0`, one epoch
(pass@1). The container runs without network, as every DeepSWE task declares for its agent and verifier (stock
inspect_harbor 1.0 ignored that and ran with internet; see Patches), so the numbers are not directly comparable to
the Qwen3.8-27B DeepSWE dir of this repo. Checkpoint: NVFP4 (W4A4) MoE experts, compressed-tensors format (LLM
Compressor), revision `863f54a`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtypes, reasoning/tool parsers, text-only mode), taken from the official recipe (Blackwell, features
  tool_calling, reasoning, text_only).
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and dataset pin, epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
- `assets/harbor_registry.json` — Harbor registry entry with every task pinned to the fork commit carrying the
  dataset fixes (see Patches).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs
  (0-3, port 8000), one TP=4 replica (so each sample's prefix cache stays on one replica), the local
  checkpoint path and the Docker socket, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

`local_orchestrator.sh` activates `~/.venvs/vllm-nightly-ple-fix` for the server and `~/.venvs/inspect-harbor-fix`
for the Inspect client; create them as below (edit the paths in `local_orchestrator.sh` if you keep them
elsewhere). `server_requirements.txt` and `client_requirements.txt` are the exact `pip freeze` of the environments
that produced the results below (Python 3.12, host details in their headers).

```bash
# server: the same vLLM nightly (0.30.1rc1.dev418+g50319916a) binaries plus the Qwen4Exp compressed-tensors
# fix from the fork branch (see Patches for upstream)
uv venv ~/.venvs/vllm-nightly-ple-fix --python 3.12 && source ~/.venvs/vllm-nightly-ple-fix/bin/activate
uv pip install -r server_requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/50319916a3c72ec1358f9e29c7c282d316d86f78 \
  --extra-index-url https://download.pytorch.org/whl/cu130
git clone -b fix/qwen4-exp-ple-compressed-tensors https://github.com/eldarkurtic/vllm.git && cd vllm
git checkout 77e1e136bf884d7e126af5a7561c7ce39f37cd7c   # the fix commit that produced the results; the branch has since merged newer main
uv pip install "cmake>=3.26.1" ninja "setuptools>=77.0.3,<81" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=50319916a3c72ec1358f9e29c7c282d316d86f78 \
  uv pip install --no-deps --no-build-isolation -e .
# client: Inspect + inspect_harbor
uv venv ~/.venvs/inspect-harbor-fix --python 3.12 && source ~/.venvs/inspect-harbor-fix/bin/activate
uv pip install -r client_requirements.txt   # or: uv pip install httpx2 "inspect-harbor @ git+https://github.com/eldarkurtic/inspect_harbor@fix/enforce-agent-network-mode" && uv pip install "openai==3.14.0"
```

Docker with the compose plugin (v2.x or later) and buildx >= 0.17 is required. The results below were produced with
a rootless Docker 29 daemon (containerd image store on local NVMe, 512 network pools, runc with `NoNewKeyring`);
`local_orchestrator.sh` points `DOCKER_HOST` at its socket. Inspect builds a task's image before its first sample
if it is missing (serially, before any sample starts). If you pre-built the `hb__<hash>` images, pass
`--sandbox-prebuilt` to skip that step. The 113 task images (about 122 GB) come from ECR public
(`public.ecr.aws/d3j8x8q7/swe-bench-202605`), which throttles bursts; pull them ahead of time, one at a time.

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`); unset it to let vLLM fetch
`RedHatAI/Qwen3.8-Flash-Next-NVFP4` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running
(never in the repo); the Harbor task files are fetched from GitHub at the pinned commit.

### Run the eval

```bash
./local_orchestrator.sh                  # add --sandbox-prebuilt if the task images are pre-built
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=48, max_sandboxes=48, time_limit=14400s, client_timeout=7200, override_cpus=0, TP=4 DP=1, epochs=1; errored samples retried once with `inspect eval-retry` (completed samples kept)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.623 (66 of 106 scored tasks) |
| stderr (binomial) | 0.0473 |
| resolved rate counting sample errors as failures | 0.584 (66 of 113) |
| samples | 113 (113 tasks x 1 epoch) |
| time-limit hits (14400 s) | 1 |
| turns truncated at max_tokens | 0 |
| tool calls that hit the tool timeout | 45 of 17987 (0.25%) |
| failed verifications with EMFILE (host inotify instance limit, scored 0) | 1 (testem-per-launcher-reports) |
| context compactions | 75 |
| sample errors (excluded from the score) | 7 (effect-sse-httpapi-streaming, koota-composite-trait-aspects, koota-pair-relation-tracking, opa-template-string-reconstruction, scriggo-method-declarations, tomlkit-toml-table-converters, wasmi-trap-coredumps) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 13,767,830 | 124,595 | 77,436 |
| median | 12,196,179 | 112,926 | 68,701 |
| max | 34,172,261 | 320,870 | 238,416 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 118,
mean 146, max 815. Wall time per sample: median 4,902 s, mean 5,692 s, max 14,433 s. Total run time about 9.6 h
including the first pass and the retry on 4x B200 (one TP=4 replica, 48 concurrent samples), with the other model's
runs on the other four GPUs of the host at the same time.

Failed tasks: awilix-async-container-initialization, bandit-structured-nosec-directives,
boa-hierarchical-evaluation-cancellation, cattrs-partial-structuring-recovery,
csstree-shorthand-expansion-compression, expr-try-catch-errors, fastapi-deprecation-response-headers,
fastapi-implicit-head-options, fd-deterministic-multi-key-sorting, geo-shapeindex-serialization,
go-genai-streamed-function-args, gql-incremental-graphql-delivery, httpx-deterministic-cookie-store,
httpx-multipart-response-parsing, httpx-streaming-json-iteration, igel-persist-feature-schema, ink-grid-box-layout,
kea-atomic-signal-selectors, kombu-virtual-queue-dead-lettering, koota-deferred-mutation-buffer,
meriyah-explicit-resource-declarations, obsidian-linter-auto-table-of-contents,
obsidian-linter-link-format-conversion, opa-rego-rule-profiling, oxvg-structural-selector-preservation,
pest-character-class-coalescing, prometheus-typed-label-sorting, python-statemachine-state-data-scoping,
quill-shared-toolbar-focus, scc-bounded-memory-spilling, sqlfmt-create-table-ddl-formatting,
superjson-error-stack-serialization, termenv-preserve-ansi-resets, testem-per-launcher-reports,
textual-kitty-key-phases, textual-richlog-follow-state, updo-policy-alerting, valibot-recursive-schema-composition,
vitest-duration-sharding, vulture-persistent-analysis-cache.

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

vLLM (nightly and `main` as of 2026-09-30) cannot load this checkpoint:
`Qwen4ExpPLEEmbeddingMethod.from_quant_config` raises `NotImplementedError: Qwen4Exp PLE embedding does not support
quantization config CompressedTensorsConfig` for any compressed-tensors checkpoint, even when no config group
targets the n-gram table. The fix (the table is loaded unquantized unless a compressed-tensors config group targets
it) is branch
[`fix/qwen4-exp-ple-compressed-tensors`](https://github.com/eldarkurtic/vllm/tree/fix/qwen4-exp-ple-compressed-tensors)
of https://github.com/eldarkurtic/vllm, based on the nightly commit `50319916a3` (fix commit `77e1e136bf`); the
server venv installs it editable on top of that nightly's precompiled binaries (see Setup). Fixed upstream since:
[vllm-project/vllm#59431](https://github.com/vllm-project/vllm/pull/59431) (merged 2026-09-30, after the nightly
used here) makes the same change; nightlies that include it should load this checkpoint without the fork (not
verified here).
