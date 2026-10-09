# Inferact/Qwen3.8-Flash-Next-NVFP4 — DeepSWE 1.1

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
the Qwen3.8-27B DeepSWE dir of this repo. Checkpoint: NVFP4 (W4A4) MoE experts, ModelOpt format, revision `103a760`
(the recipe's default variant).

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
  (4-7, port 8001), one TP=4 replica (so each sample's prefix cache stays on one replica), the local
  checkpoint path and the Docker socket, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below (Python 3.12,
host details in their headers).

```bash
# server: vLLM nightly 0.30.1rc1.dev418+g50319916a (the recipe requires a nightly build; CUDA 13 wheels)
uv venv ~/.venvs/vllm-nightly --python 3.12 && source ~/.venvs/vllm-nightly/bin/activate
uv pip install -r server_requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/50319916a3c72ec1358f9e29c7c282d316d86f78 \
  --extra-index-url https://download.pytorch.org/whl/cu130
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
`Inferact/Qwen3.8-Flash-Next-NVFP4` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running
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
| resolved rate (pass@1, 1 epoch) | 0.581 (61 of 105 scored tasks) |
| stderr (binomial) | 0.0484 |
| resolved rate counting sample errors as failures | 0.540 (61 of 113) |
| samples | 113 (113 tasks x 1 epoch) |
| time-limit hits (14400 s) | 2 |
| turns truncated at max_tokens | 0 |
| tool calls that hit the tool timeout | 26 of 16918 (0.15%) |
| failed verifications with EMFILE (host inotify instance limit, scored 0) | 1 (testem-per-launcher-reports) |
| context compactions | 78 |
| sample errors (excluded from the score) | 8 (dynamodb-toolbox-lazy-recursive-schemas, kea-atomic-signal-selectors, koota-composite-trait-aspects, koota-pair-relation-tracking, opa-template-string-reconstruction, pwntools-tube-multiplexing, python-statemachine-state-data-scoping, scriggo-method-declarations) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 13,581,899 | 120,695 | 73,651 |
| median | 12,730,404 | 111,141 | 65,641 |
| max | 36,919,494 | 287,971 | 194,968 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 118,
mean 137, max 473. Wall time per sample: median 4,575 s, mean 5,298 s, max 14,539 s. Total run time about 9.9 h
including the first pass and the retry on 4x B200 (one TP=4 replica, 48 concurrent samples), with the other model's
runs on the other four GPUs of the host at the same time.

Failed tasks: arcane-drift-detection-baselines, awilix-async-container-initialization,
bandit-structured-nosec-directives, boa-hierarchical-evaluation-cancellation, clack-async-autocomplete-options,
claude-code-by-agents-recursive-delegation, cliffy-config-file-parsing, effect-sse-httpapi-streaming,
fastapi-deprecation-response-headers, fastapi-implicit-head-options, fd-deterministic-multi-key-sorting,
go-critic-doc-link-checker, gql-incremental-graphql-delivery, happy-dom-deterministic-intersectionobserver,
helm-array-merge-strategies, httpx-multipart-response-parsing, ink-grid-box-layout, katex-multicolumn-array-spans,
kgateway-consistent-hash-policy, kombu-virtual-queue-dead-lettering, koota-deferred-mutation-buffer,
koota-entity-snapshot-rollback, koota-query-predicates, langchain-request-coalescing,
meriyah-explicit-resource-declarations, narwhals-rolling-window-suite, obsidian-linter-auto-table-of-contents,
obsidian-linter-link-format-conversion, oxvg-structural-selector-preservation, pebble-durability-wait-apis,
pest-character-class-coalescing, prometheus-transactional-reload-status, prometheus-typed-label-sorting,
psd-tools-blend-range-api, quill-shared-toolbar-focus, superjson-error-stack-serialization,
termenv-preserve-ansi-resets, testem-bail-on-test-failure, testem-per-launcher-reports, textual-kitty-key-phases,
textual-richlog-follow-state, updo-policy-alerting, wasmi-trap-coredumps, yaegi-go-embed-directives.

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
