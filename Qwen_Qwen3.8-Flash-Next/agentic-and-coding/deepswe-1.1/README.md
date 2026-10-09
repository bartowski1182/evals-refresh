# Qwen/Qwen3.8-Flash-Next — DeepSWE 1.1

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
the Qwen3.8-27B DeepSWE dir of this repo. Checkpoint: the original BF16 weights, revision `de4b8e4`.

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
  (0-7, port 8003), one TP=8 replica (so each sample's prefix cache stays on one replica), the local
  checkpoint path and the Docker socket, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

`local_orchestrator.sh` activates `~/.venvs/vllm-nightly` for the server and `~/.venvs/inspect-harbor-fix` for the
Inspect client; create them as below (edit the paths in `local_orchestrator.sh` if you keep them elsewhere).
`server_requirements.txt` and `client_requirements.txt` are the exact `pip freeze` of the environments that
produced the results below (Python 3.12, host details in their headers).

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
`Qwen/Qwen3.8-Flash-Next` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running (never in the
repo); the Harbor task files are fetched from GitHub at the pinned commit.

### Run the eval

```bash
./local_orchestrator.sh                  # add --sandbox-prebuilt if the task images are pre-built
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=48, max_sandboxes=48, time_limit=14400s, client_timeout=7200, override_cpus=0, TP=8 DP=1, epochs=1; errored samples retried once with `inspect eval-retry` (completed samples kept)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.660 (70 of 106 scored tasks) |
| stderr (binomial) | 0.0462 |
| resolved rate counting sample errors as failures | 0.619 (70 of 113) |
| samples | 113 (113 tasks x 1 epoch) |
| time-limit hits (14400 s) | 0 |
| turns truncated at max_tokens | 0 |
| tool calls that hit the tool timeout | 24 of 16946 (0.14%) |
| failed verifications with EMFILE (host inotify instance limit, scored 0) | 1 (scc-bounded-memory-spilling) |
| context compactions | 44 |
| sample errors (excluded from the score) | 7 (dynamodb-toolbox-lazy-recursive-schemas, opa-template-string-reconstruction, optique-conditional-option-dependencies, scriggo-method-declarations, tomlkit-toml-table-converters, wasmi-trap-coredumps, yjs-map-conflict-detection) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 13,261,658 | 114,720 | 70,036 |
| median | 12,300,497 | 105,311 | 63,875 |
| max | 35,736,067 | 212,337 | 156,645 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 113,
mean 135, max 772. Wall time per sample: median 2,461 s, mean 2,698 s, max 6,195 s. Total run time about 4.1 h
including the first pass and the retry on 8x B200 (one TP=8 replica, 48 concurrent samples), with no other model on
the host.

Failed tasks: actionlint-action-pinning-lint, adaptix-name-mapping-aliases, awilix-async-container-initialization,
bandit-structured-nosec-directives, boa-hierarchical-evaluation-cancellation, dasel-html-document-format,
effect-sse-httpapi-streaming, fastapi-implicit-head-options, go-critic-doc-link-checker,
go-genai-streamed-function-args, gql-incremental-graphql-delivery, happy-dom-deterministic-intersectionobserver,
httpx-multipart-response-parsing, ink-grid-box-layout, ipython-session-bundle-replay,
kombu-virtual-queue-dead-lettering, koota-deferred-mutation-buffer, koota-pair-relation-tracking,
mashumaro-flattened-dataclass-fields, meriyah-explicit-resource-declarations,
obsidian-linter-auto-table-of-contents, obsidian-linter-link-format-conversion, onedump-dump-encryption-pipeline,
opa-rego-rule-profiling, oxvg-structural-selector-preservation, participle-grammar-conflict-analysis,
pebble-durability-wait-apis, pest-character-class-coalescing, prometheus-transactional-reload-status,
prometheus-typed-label-sorting, quill-shared-toolbar-focus, scc-bounded-memory-spilling,
termenv-preserve-ansi-resets, testem-bail-on-test-failure, textual-richlog-follow-state, updo-policy-alerting.

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
