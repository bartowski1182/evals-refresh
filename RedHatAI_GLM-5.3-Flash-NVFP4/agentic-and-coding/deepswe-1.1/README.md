# RedHatAI/GLM-5.3-Flash-NVFP4 — DeepSWE 1.1

| | |
|---|---|
| Task | `inspect_harbor/harbor` (113 tasks of DeepSWE 1.1, loaded through `assets/harbor_registry.json`, which pins each task to the fork commit carrying the verifier fix) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per task (public.ecr.aws, 8 GB requested), bash/python tools, hidden test verifier, no network |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash |
| Task source | https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1 |

**Protocol note.** The GLM-5.3-Flash card reports DeepSWE = 63.4 with the mini-swe-agent harness (temperature 0.95,
top_p 1.0, 400K context, 6 h timeout). `inspect_harbor` instead solves every task with Inspect's generic ReAct agent
(`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction; no message limit),
temperature 1.0 / top_p 0.95, a single `--time-limit 14400` wall-clock guard, `override_cpus=0`, one epoch (pass@1),
so the number below is **not directly comparable** to that card figure. The container runs with no network, as every
DeepSWE task declares for its agent and verifier (stock inspect_harbor 1.0 ignored that and ran with internet; see
Patches). Checkpoint: NVFP4 (mixed precision — W4A4 on most MoE experts, FP8 on the last expert layer and the MTP
module), compressed-tensors format (LLM Compressor), base `zai-org/GLM-5.3-Flash`, revision `18d55bf`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtype, MTP speculative decoding, reasoning/tool parsers, text-only mode), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task and dataset pin,
  epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
- `assets/harbor_registry.json` — Harbor registry entry with every task pinned to the fork commit carrying the
  dataset fixes (see Patches).
- `server_requirements.txt` / `client_requirements.txt` — `uv pip freeze` of the vLLM and Inspect venvs that
  produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs (0-7,
  port 8000), one TP=8 replica (so each sample's prefix cache stays on one replica), the local checkpoint path,
  starts `serve.sh` in the background, runs `run_eval.sh`, tears down. **This is the only file you should modify**
  to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `uv pip freeze` of the environments that produced the results below
(Python 3.12, host details in their headers).

```bash
# server: a vLLM nightly that carries the glm5_next integration (0.32.0rc1.dev249+gfaf417b75, commit faf417b75).
# Stock nightly loads this checkpoint (no model-specific fork needed). flashinfer-python 0.7.1 provides the sparse
# MLA kernels; CUDA_HOME must point at a CUDA 13 toolkit so its JIT path works.
uv venv ~/.venvs/glm-vllm --python 3.12 && source ~/.venvs/glm-vllm/bin/activate
uv pip install -r server_requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/faf417b75 \
  --extra-index-url https://download.pytorch.org/whl/cu130
# (server_requirements.txt records vLLM as an editable checkout; installing the nightly wheel for commit faf417b75
#  gives an equivalent environment, since the glm5_next support is upstream.)

# client: Inspect + inspect_harbor from the network-mode fork (see Patches)
uv venv ~/.venvs/glm-inspect-harbor-fix --python 3.12 && source ~/.venvs/glm-inspect-harbor-fix/bin/activate
uv pip install -r client_requirements.txt
# or: uv pip install "inspect-ai==0.3.273" "inspect-harbor==1.0.0" httpx2 "openai==3.14.0" && \
#     uv pip install --no-deps "inspect_harbor @ git+https://github.com/eldarkurtic/inspect_harbor@cb61925"
```

Docker with the compose plugin (v2.x or later) is required. The results below were produced with the host's system
(rootful) Docker 25 daemon (overlay2 image store on local NVMe); the invoking user is in the `docker` group, so
`local_orchestrator.sh` leaves `DOCKER_HOST` at its default. The 113 task images (about 122 GB) come from ECR public
(`public.ecr.aws/d3j8x8q7/swe-bench-202605`), which throttles bursts; pull them ahead of time, one at a time. Use the
**v1.1** task images (the v1.0 images keep future `origin/*` refs).

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`); unset it to let vLLM fetch
`RedHatAI/GLM-5.3-Flash-NVFP4` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running (never in the
repo); the Harbor task files are fetched from GitHub at the pinned commit.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k unset, reasoning_effort=max, max_tokens=65536 (per turn), max_connections=48, max_sandboxes=48, time_limit=14400s, client_timeout=7200, override_cpus=0, TP=8 DP=1 with MTP (num_speculative_tokens=5), kv_cache_dtype=fp8, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch, sample errors as failures) | **0.602 (68 of 113)** |
| stderr | 0.046 |
| resolved rate counting only scored samples | 0.630 (68 of 108) |
| samples | 113 (113 tasks × 1 epoch) |
| time-limit hits (14400 s) | 1 (wasmi-trap-coredumps) |
| turns truncated at max_tokens | 0 |
| sample errors (`RuntimeError: Compaction insufficient`) | 5 (dynamodb-toolbox-conditional-attribute-requirements, dynamodb-toolbox-lazy-recursive-schemas, opa-template-string-reconstruction, scriggo-method-declarations, valibot-recursive-schema-composition) |
| samples that reached the network (fix check) | 0 of 855 outbound attempts (all failed: DNS/connection errors) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 23,173,915 | 139,182 | 93,650 |
| median | 14,096,458 | 126,588 | 87,022 |
| max | 373,682,269 | 598,164 | 523,954 |

Input tokens are cumulative over agent turns (history is resent each turn). Assistant turns per sample: median 127,
mean 189, max 2,200. Total run time 5:04:12 on 8× B200 (one TP=8 replica, 48 concurrent samples; prefix-cache hit
rate held at ~97% throughout, so the single large-TP replica avoided the per-replica KV thrash that a DP topology
causes on these ~120k-token histories). No solution leaks: every DeepSWE task container ran with `network_mode:
none`, and every one of the 855 outbound attempts the agents made (pip install, git clone, curl/npm) failed with
DNS/connection errors.

The 5 sample errors are `RuntimeError: Compaction insufficient` — the agent's messages still exceed the compaction
threshold after a compaction pass, so the sample is not graded (counted as a failure above). Two occur at the
~116k-token threshold (Inspect's 128k fallback window) and three at the ~245k threshold (the true 262k window, once
Inspect has learned it from the server); see the note in `../terminal-bench-2.1/README.md`.

Failed tasks (scored 0): anko-default-function-arguments, awilix-async-container-initialization,
boa-hierarchical-evaluation-cancellation, claude-code-by-agents-recursive-delegation,
csstree-shorthand-expansion-compression, expr-try-catch-errors, fd-deterministic-multi-key-sorting,
geo-shapeindex-serialization, happy-dom-deterministic-intersectionobserver, httpx-streaming-json-iteration,
igel-persist-feature-schema, ink-grid-box-layout, kea-atomic-signal-selectors, kombu-single-active-consumer-priority,
kombu-virtual-queue-dead-lettering, koota-composite-trait-aspects, koota-deferred-mutation-buffer,
langchain-request-coalescing, meriyah-explicit-resource-declarations, obsidian-linter-link-format-conversion,
obsidian-linter-scoped-ignore-markers, onedump-dump-encryption-pipeline, opa-rego-rule-profiling,
optique-conditional-option-dependencies, oxvg-structural-selector-preservation, participle-grammar-conflict-analysis,
pest-character-class-coalescing, prometheus-transactional-reload-status, prometheus-typed-label-sorting,
psd-tools-blend-range-api, returns-validated-error-accumulation, sqlfmt-create-table-ddl-formatting,
superjson-error-stack-serialization, termenv-preserve-ansi-resets, testem-bail-on-test-failure,
testem-per-launcher-reports, ts-pattern-match-each, updo-policy-alerting, vulture-persistent-analysis-cache,
wasmi-trap-coredumps.

## Patches for upstream

- **The task's own network policy was not enforced.** Every DeepSWE `task.toml` declares `[agent]` and `[verifier]`
  `network_mode = "no-network"`, but inspect_harbor 1.0 read network policy only from `[environment]`, so agents had
  public internet. Fixed on branch [`fix/enforce-agent-network-mode`](https://github.com/eldarkurtic/inspect_harbor/tree/fix/enforce-agent-network-mode)
  of https://github.com/eldarkurtic/inspect_harbor (commit `cb61925`, on top of v1.0.0), which applies `[agent]` and
  `[verifier]` `network_mode` per phase; with both denied, the container runs with `network_mode: none` throughout.
  The client venv installs inspect_harbor from that branch; the sample containers were verified to have no route out
  (0 of 855 outbound attempts succeeded). Upstream PR:
  [meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200).
- **Git history is clean.** The v1.1 task images (`public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`) are built from
  the tasks' `environment/Dockerfile`, which keeps only the base commit's history. The v1.0 images keep future
  `origin/*` refs; use v1.1.
- Carried over in the pinned fork commit and consumed here: the `[[verifier.collect]]` hook, the vitest worker-pool
  OOM fixed with `taskset` (datacurve-ai/deep-swe#100), and the verifier's non-login shell (inspect_harbor 1.0).

No model-specific vLLM fork is needed: the stock nightly (commit faf417b75) loads and serves this checkpoint (the
Qwen4Exp PLE fix that the Qwen3.8-Flash-Next dirs required does not apply to `glm5_next`).
