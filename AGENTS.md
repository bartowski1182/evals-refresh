# evals-refresh: guide for agents

This repo holds copy-pasteable, reproducible evaluations of open-weight models served with
**vLLM** and evaluated with **Inspect** (`inspect-ai`, plus `inspect_evals` and `inspect_harbor`
task packages). Read this file before running or adding anything. Everything below is a rule
unless it says "hint".

## 0. Local environment notes

`local_env_notes.md`, next to this file, holds environment requirements and pitfalls that apply to any host
(serving, caches, Docker setup, image pulls, venvs). Read it before running anything; it is imported below so
that Claude Code loads it with this file. It is committed and used on other servers, so keep it general. Facts
about one machine (paths, user names, installed versions, other tenants) go in `local_env_notes.local.md`,
which git ignores; read it too if it exists.

@local_env_notes.md

## 1. Ground truth, in priority order

1. **Serving flags come from the model's official vLLM recipe** at `https://recipes.vllm.ai/<Org>/<model>`
   (path is case-sensitive; the source is `github.com/vllm-project/recipes`). Record the URL in the
   task README. Do not invent flags; do not copy flags from another model.
2. **Sampling params come from the model's `generation_config.json`** on the HF hub (temperature,
   top_p, top_k). Never run greedy: every model here is a reasoning model whose vendor warns
   against `temperature=0`.
3. **Thinking is on, at the highest effort the model supports**, set on both sides for safety:
   server-side via `--default-chat-template-kwargs` in `serve.sh`, and per request in
   `run_eval.sh` (`--reasoning-effort xhigh` for Qwen; `extra_body.chat_template_kwargs` via
   `assets/generate_config.json` for Gemma, which has no effort levels). Adapt as needed for
   each model family.
4. **Task definitions come from `inspect_evals` or `inspect_harbor` unchanged.** No task wrappers
   in this repo. If a task is broken, fix it upstream (PR from a fork under github.com/neuralmagic) and
   consume the fix from the fork branch (pip install from git, or a Harbor registry pin to the fork
   commit); document it under the README's "Patches for upstream" section. No local patch scripts.
   When the fix merges upstream, point the dir at the upstream commit instead of the fork, and check that the
   upstream commit applied to the dir's pinned base gives the source that produced the results (`git apply` the
   upstream commit's diff onto the base checkout and compare with the fork commit).
5. **The model card's protocol decides comparability.** Before adding a task, read what the card
   reports (subset, single-shot vs agentic, judge, epochs) and say in the README whether our number
   is comparable. `eval_coverage_matrix.md` at the repo root maps benchmarks to packages,
   categories, and execution tiers (`none` / `sandbox` / `agent`).

## 2. Layout

```
<org>_<model>/<category>/<task>/
  serve.sh                 vLLM server: model, TP/DP, context, parsers, thinking default. Nothing else apart from vLLM server specific arguments.
  run_eval.sh              inspect eval: task, epochs, sampling, reasoning, limits, concurrency. Nothing else apart from vLLM server specific arguments.
  local_orchestrator.sh    the ONLY env-specific file: venvs, GPUs, secrets, start server, wait, eval, teardown.
  README.md                task table, Files, Run, Results (config + scores + token stats), Patches.
  server_requirements.txt  `pip freeze` of the vLLM venv that produced the results, host info in the header.
  client_requirements.txt  `pip freeze` of the Inspect venv that produced the results.
  assets/                  only what the scripts reference: chat templates, generate_config.json, registry pins.
  logs/                    gitignored. Keep only the final run's .eval, serve.log, full.log.
```

- `<org>_<model>` is the HF id with `/` replaced by `_` (`Qwen_Qwen3.8-27B`, `google_gemma-4-31B-it`).
- `<category>` is the `Evaluates` column of `eval_coverage_matrix.md`: `knowledge-and-reasoning`,
  `agentic-and-coding`, `general-tool-usage`, `long-context`, `multimodal`.
- Every task directory is standalone. No shared includes, nothing to source. Duplication across
  models is deliberate; drift between copies is expected and is the point (each copy documents
  what ran). One exception: a `local_orchestrator.sh` may source `tools/vllm_server.sh`, which starts, health-checks
  and stops vLLM servers from a venv (`VLLM_VENV`) or a podman image (`VLLM_IMAGE`; setting both is an error) and stops them
  on any exit. It holds only environment glue; what is measured stays in the dir's `serve.sh` and `run_eval.sh`. The
  orchestrator sets `DEFAULT_VLLM_IMAGE`, used when neither is set, to the image that produced the dir's results, and `logs/serve.log` records the one a run
  used. The Muse-Spark-Open and GLM-5.3-Flash pair dirs use it; other dirs still inline their venv or podman commands. Move a dir
  over when you touch its orchestrator, not as a sweep.
- `serve.sh` and `run_eval.sh` are plain, readable bash with env-var overrides
  (`MODEL`, `PORT`, `TP`, `DP`, `EPOCHS`, `LOG_DIR`, `MAX_TOKENS`, `MAX_CONNECTIONS`, ...). No
  defensive scaffolding, no logging frameworks, no argument parsers.


## 3. Rules of engagement

- **Never `git commit` or `git push`.** Leave changes in the working tree and say what changed.
  Committing is the repo owner's job.
- **Pull before you start, and read what came in.** Several hosts push to this repo. Fixes and conventions found on
  one host (e.g. the leak fixes of section 11) reach you only through a pull, so before copying a task dir, check
  the newest copies of that task under every model. If a pull brings a task dir that you also have locally, the two
  are probably different runs: compare them, ask the user which one is final, and move the other out of the repo
  with its logs (a README must match the logs next to it).
- **Reflect after every eval and every meaningful piece of work, and write the general lessons into the notes.**
  This is mandatory; section 12 gives the procedure.
- **Keep every agent instruction in this file.** `CLAUDE.md` only imports it (`@AGENTS.md`), so Claude Code and
  other agent harnesses read the same rules. Do not add content to `CLAUDE.md`, including what `/init` proposes;
  put conventions here, environment notes that apply to any host in `local_env_notes.md`, and facts about one
  machine in `local_env_notes.local.md`.
- **Never refer to a local venv as if it exists.** Venvs are local to one user and server, and other users won't
  have yours. In this file, `local_env_notes.md` and READMEs, describe an environment by its packages (the dir's
  requirements files or pins) and give the commands to create it. Real paths live only in `local_orchestrator.sh`
  and `local_env_notes.local.md`. A README's Setup commands may create the environments at the paths the
  orchestrator activates, as long as the README says to change both together.
- **Never put secrets in the repo.** `HF_TOKEN`, API keys and other secrets should be local on
  the user's machine. 
- **Ask before deciding anything that changes what is measured**: protocol, subset, judge model,
  time limits, epochs. Do not silently narrow or widen a task. Infrastructure choices (TP/DP,
  concurrency, disk) are yours to make and to document.
- **Always smoke-test first** (`EPOCHS=1 ./local_orchestrator.sh --limit 5`, or `--limit 2` for
  agentic tasks), read the samples, and only then launch the full run. Verify in the log that
  reasoning was parsed into a separate `reasoning` content part, that `stop_reason` is `stop`
  (not `max_tokens`), and that the config you intended is what `eval.model_generate_config` recorded.
- **Audit every run of a task with tools for cheating before you report it** (exec tier `sandbox` or `agent`).
  Some models fetch a task's solution from the internet or from git history instead of solving it. Before the run,
  close every leak the task allows; after the run, scan the trajectories. Section 11 lists the fixes and the audit.
- **One epoch is the default** (`--epochs 1`, pass@1) for every benchmark, new task and new model. Run more
  epochs (`EPOCHS=N`, `--epochs-reducer mean`, avg@N) only when the user asks for them, and say so in the README.
  When you copy a task dir whose `run_eval.sh` defaults to more epochs, set its `EPOCHS` default back to 1. The
  defaults in existing dirs record what produced their results (section 10).
- **Results go in the README as data, not narrative.** Fill the Results template (below). Do not
  describe failed attempts, ablations, or how you got there in the README. README only contains
  final results, everything else should be discussed with the user in chat.
- **Headline scores over all samples.** Sample errors and unscored samples count as failures (0; in avg@N, 0 for
  that epoch), so every model is scored with the same denominator, e.g. "0.336 (38 of 113)". List the errors
  separately and compute stderr the same way. Never headline a scored-only rate: errored samples differ between
  models, and they are often the longest, unfinished trajectories. Some older READMEs headline errors-excluded
  numbers; recompute their all-samples figure before comparing with them.
- **One task directory, one server.** Do not add multi-model orchestration, judges on split GPUs,
  or background daemons. If a task needs a judge, run judging as a separate phase
  (`inspect eval --no-score`, then `inspect score --model-role grader=...`) and document it.

## 4. Python environments (adapt in `local_orchestrator.sh`)

Three environments, deliberately separate. They are local to each user and server, so create your own, name and place
them as you like, and point `local_orchestrator.sh` at them; it is the only file that knows their paths.

| environment | used by | create with |
|---|---|---|
| vLLM server | `serve.sh` | `pip install vllm` |
| Inspect client for `inspect_evals` tasks | `run_eval.sh` | `pip install git+https://github.com/neuralmagic/inspect_evals.git openai` |
| Inspect client for `inspect_harbor` tasks | `run_eval.sh` | `pip install inspect-harbor httpx2 "openai==3.14.0"` |

Always install `inspect_evals` from `github.com/neuralmagic/inspect_evals` (not PyPI). That fork is this repo's
source of truth for task definitions and carries tasks and fixes that are not on a PyPI release yet (e.g. MRCR).
Pin a commit or branch in the task README and `client_requirements.txt` when a dir needs a specific revision;
otherwise `main` is fine for a new env. These installs are the starting point for a new task. To reproduce a
dir's results, create the environments from its `server_requirements.txt` and `client_requirements.txt` instead,
as its README Setup shows.

`inspect-harbor` depends on `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible
providers need `openai>=3`; installing `openai==3.14.0` last overrides the pin and the resulting
pip warning is harmless (Inspect's agent path never imports litellm). `httpx2` is imported by
recent `inspect-ai` but not declared.

## 5. How to run an existing task

```bash
cd <org>_<model>/<category>/<task>
EPOCHS=1 ./local_orchestrator.sh --limit 5     # smoke test; inspect the samples first
./local_orchestrator.sh                        # full run (EPOCHS=N for more epochs, if the user asks)
inspect view --log-dir ./logs                  # browse samples (port 7575)
```

Monitor a running eval from another shell with `inspect ctl task list` (from the same venv).
During Docker image builds it reports no running eval; watch the eval's stdout instead.

Read results programmatically (this is how README numbers are produced):

```python
from inspect_ai.log import read_eval_log
log = read_eval_log("logs/<file>.eval")
log.eval.model_generate_config      # the config line for the README
log.results.scores                  # accuracy, stderr
log.stats.model_usage               # run totals: input / output / reasoning tokens
for s in log.samples:
    s.output.usage                  # per-sample tokens (reasoning counted separately)
    s.output.stop_reason            # "stop" or "max_tokens"
    s.scores, s.error, s.limit      # verdict, infra error, time/message limit hit
```

If a run is interrupted, `inspect eval-retry logs/<file>.eval` reruns only unfinished samples and
keeps completed ones. It reuses the log's task and args; env vars still apply.

To add epochs to a finished run (when the user asks for avg@N), keep the existing log as epoch 1, run
`EPOCHS=<N-1>` into its own log dir with the same concurrency (in agentic tasks the number of samples in flight
changes the time-limit hits), and combine with a per-task mean over all N epochs. Every log numbers its epochs
from 1: map them to overall epochs before you report per-epoch numbers or point to a sample.

## 6. README template (fill exactly this, nothing more)

```
# <org>/<model> — <Task name>

| | |
|---|---|
| Task | `<package>/<task>` (<size, subset>) |
| Category | <category> (subclass: <from the matrix>) |
| Exec tier | `none` / `sandbox` / `agent` — <what infra it needs> |
| vLLM recipe | <url> |

<one-paragraph protocol note if the number is not comparable to the model card>

## Files            (one line per file, what lives in it; local_orchestrator.sh is "the only file you should modify")
## Run              (### Setup: how to create both venvs from the *_requirements.txt files or the pinned
                     packages, HF_TOKEN / gated datasets, Docker expectations; ### Run the eval: the three
                     commands above. A newcomer must be able to reproduce the results from the README alone.)
## Results
With model generation config:
`<eval.model_generate_config, plus epochs, limits, client_timeout>`
the following scores and token stats are obtained:
| metric | value |                       accuracy (pass@1, or avg@N if asked for), stderr, per-epoch, samples,
                                          truncated / time-limit hits, unparsed / no-solution, sample errors
| tokens / sample | input | output | reasoning |   mean / median / max
<one line on how to read the token numbers if non-obvious; list of always-failing tasks for agentic evals>
## Patches for upstream   (every upstream defect this task needed fixed: the PR/issue link, and how this dir
                            consumes the fix from the fork - a pip install from the fork branch or a Harbor registry
                            pin to the fork commit in assets/ - never a local patch script; "None." otherwise)
```

No log filenames in the README (logs are gitignored). No environment details in the README
(they live in the two requirements files). Regenerate the requirements files after the final
run with `pip freeze` from each venv, with a header comment giving date, GPUs, driver, CUDA,
Python.

## 7. Adding a new model

1. Create `<org>_<model>/<category>/<task>/` by copying the closest existing task dir. For a task listed in
   section 11, copy the dir named there, which carries the leak fixes.
2. `serve.sh`: rewrite from the model's vLLM recipe. Keep only what the recipe says plus
   `--tensor-parallel-size`, `--data-parallel-size`, `--served-model-name`, `--host/--port`.
   Confirm the reasoning parser and tool parser names exist in the installed vLLM
   (`vllm/reasoning/__init__.py`, `vllm/tool_parsers/__init__.py`) and that the architecture in
   the model's `config.json` is in `vllm/model_executor/models/registry.py`. If the recipe uses a
   custom chat template, copy it into `assets/` byte-identical and point `--chat-template` at it.
   For text-only tasks on multimodal models pass `--limit-mm-per-prompt '{"image": 0, "audio": 0}'`.
3. `run_eval.sh`: sampling from `generation_config.json`; thinking on per request; keep
   `-M client_timeout=7200` (Inspect's default HTTP timeout is 600 s and silently retries long
   generations from scratch); `--max-tokens` strictly below `--max-model-len`.
4. Check how the model's card reports the benchmark and note comparability in the README.
5. Smoke test, read samples, full run, fill Results, regenerate requirements files.

## 8. Adding a new task

1. Find it in `eval_coverage_matrix.md`. `inspect_evals` tasks run in the `inspect_evals` client environment;
   `inspect_harbor` tasks (`Where` = `inspect_harbor`) need the `inspect_harbor` one (section 4) and Docker.
   `PARTIAL` / `MISSING` rows are not the benchmark the cards report; say so if asked.
2. Read the task's README in the package and its task signature (`inspect list tasks`, or import
   it). Identify: dataset and pinned revision, gating (HF token / terms), subset flags,
   sandbox, judge model, default metrics, and whether the protocol matches the model cards.
3. Decide with the user anything from rule 3 (subset, judge, epochs, limits). Also discuss with 
   user the potential need to extend models context length (e.g. with YaRN) if needed for the task
   and if the model supports it.
4. Category directory = the matrix's `Evaluates` column. Task directory name = the benchmark's
   common name in lowercase with hyphens (`gpqa-diamond`, `livecodebench`).
5. Sandboxed / agentic tasks: check disk (per-task images), set `--max-sandboxes` equal to
   `--max-connections` (Inspect holds a container per in-flight sample), set `--time-limit` as a
   runaway guard and report hits, and use `--fail-on-error 0.1` so one infra error cannot abort
   a multi-hour run while a systematic one still does. Per-turn `--max-tokens` must leave room
   for multi-turn history inside the context window (65536 was used with a 262k window). Find out how
   the task could leak its solution (network policy in each `task.toml`, git history in the images, public
   copies of the task files), close what you can, and add the task to section 11.
6. Judged tasks: no external APIs are assumed. Plan a local judge as a separate scoring phase and
   keep it identical across models.

## 9. Gotchas already paid for

- **`inspect ctl` "http_retries" climbing with no errors** = client-side timeouts re-running
  long generations forever. Fix: `-M client_timeout=<seconds>` (or `INSPECT_HTTP_REQUEST_TIMEOUT`).
  `--timeout` is a different, outer budget and does not fix it.
- **`extra_body` (e.g. `chat_template_kwargs`) has no CLI flag.** `-M` args go to the provider
  constructor, not the request. Use `--generate-config assets/generate_config.json`; it merges
  with the individual sampling flags.
- **`--max-tokens >= --max-model-len`** makes vLLM reject every request while Inspect reports
  success with accuracy 0.000 in seconds.
- **Harbor tasks build one image per task, serially, before any sample starts**, and Inspect's
  control server is invisible during that phase. A Dockerfile that copies per-task files before
  `apt-get` defeats layer sharing (LiveCodeBench: ~750 MB and ~1 min per image ×100). Pre-build in
  parallel if needed; check disk first.
- **A Harbor verifier that exits without writing `reward.txt`** turns a failed task into a sample
  *error*, excluded from accuracy, and aborts the eval under default fail-on-error. Fixed upstream
  for LiveCodeBench; pin via `assets/harbor_registry.json` + `inspect_harbor/harbor -T registry_path=...`.
- **Models sometimes emit pseudo tool calls as plain text** (Gemma: `<div class="tool_call">`);
  the ReAct loop then re-prompts until the time limit. Count these in the README as "no solution".
- **GitHub's API is rate-limited without a login**; use `git ls-remote` / shallow clones instead,
  or log in with `gh auth login`.
- **vLLM prints an `EngineDeadError` traceback at shutdown** when the orchestrator kills it.
  It is noise if it appears at the moment the eval finished.
- **Inspect's multiple-choice parser** requires `ANSWER: <LETTER>` on the last line; a model that
  writes `ANSWER: $A` can never be parsed. Check the unparsed count.
- **DSA/MLA models with a huge native context can OOM vLLM's post-graph warm-up on Hopper.** GLM-5.3's
  sparse-indexer JIT warm-up allocates ~24 GiB per GPU (warm-up batch × 64 heads × `max_model_len` × fp32) with
  the checkpoint's default 1M window, after CUDA graphs captured fine. Cap `--max-model-len` (262144 worked and
  matches the repo's other multi-turn agent dirs) unless the recipe explicitly fits the full window on your GPUs.
- **Size agentic concurrency to the KV pool, not the GPU count.** MLA-style KV costs ~50 KB/token: GLM-5.3 on
  8xH200 holds only ~1.19M KV tokens (a hybrid-attention model of similar size holds ~20M), so DeepSWE ran 10
  concurrent ~120k-token histories and Terminal-Bench 32. Read the "GPU KV cache size" line in serve.log before
  choosing `--max-connections`; oversubscribing evicts prefix-cached histories and re-prefills them every turn.
- **Inspect's vLLM context-window discovery loses a race with compaction.** The `vllm` provider learns the
  served `max_model_len` from `/v1/models` during the first `generate()`, but the ReAct agent's compaction
  resolves its threshold (`0.9 × window`) on the first `compact_fn`, which can run before that call. The race
  goes both ways in practice: some samples use Inspect's 128,000 default and a threshold of 115,200, others use
  the real window and 235,929, and Inspect caches whichever it resolved for the whole run. `CompactionEdit` only
  clears old tool results and thinking blocks, keeping the last 3 tool-use/result pairs and all conversation
  text, so the compacted size of a long trajectory rises toward the threshold. Once the remaining material
  cannot be reduced, the sample dies with `RuntimeError: Compaction insufficient`. A correct window is not a
  cure: one sample failed at 237,577 tokens against 235,929. Inspect records such a death as a *sample error*,
  so the sample drops out of the scored denominator without saying so. Count it as a failure, per section 3. The
  problem is not model-specific. It has hit deepswe-1.1 with 5 to 15 samples per model, swe-bench-pro and
  terminal-bench-2.1 across every family, and 12 such errors in one deepswe run tripped `--fail-on-error 0.1`
  and forced an `inspect eval-retry`. No CLI or env override exists. Pinning the window needs `set_model_info()`
  in Python, which is a wrapper, so only with the user's agreement. The proper fix is upstream: inspect-ai should
  re-resolve the threshold after registration, and inspect_harbor, which hardcodes `CompactionEdit()`, could
  expose the threshold as a task arg. The 128,000 is only the compaction trigger and never a hard cap, so
  contexts above it still occur.
- **vLLM 0.32 returns reasoning in the `reasoning` message field** (OpenAI style), not `reasoning_content`.
  inspect-ai 0.3.273 parses it into a separate reasoning content part; check the field name in a raw
  `/v1/chat/completions` response before concluding the reasoning parser is broken.
- **Killing a wrapper shell orphans its grandchildren.** A harness timeout that kills `bash -c` does not reach a
  vLLM server or eval started inside it; a vLLM server whose workers died mid-startup then hangs forever as an
  unhealthy process with a zombie EngineCore. Kill servers by exact PID, confirm `nvidia-smi` shows 0 MiB before
  restarting, and never `pkill -f` a pattern that appears in your own command string (use the bracket trick:
  `pgrep -f "vllm [s]erve"`).
- **vLLM ignores SIGTERM, and a reservation can be released while its workers still hold GPU memory.** A
  `kill $SERVER_PID` reaches only the launcher: EngineCore and the TP workers survive and keep their memory, and
  with `canhazgpu run` the reservation is released the moment the wrapped command exits, so the GPUs are handed
  out again while orphaned workers still occupy them. Start the server with `setsid` so it gets its own process
  group, tear down with `kill -9 -- -$PGID`, and block until `nvidia-smi` reports 0 MiB on *your* GPUs before
  exiting (`CUDA_VISIBLE_DEVICES` ids only: summing all GPUs never reaches zero on a shared host because other
  tenants hold the rest). An orchestrator that merely `kill`s and exits caused a real double-allocation race here.
- **Completion watchers must not race the eval start.** A `while pgrep inspect-eval` loop started during server
  startup exits immediately (the eval process does not exist yet) and reports a false "finished". Watch the
  orchestrator script's process instead, or start the watcher after samples are in flight.
- **The card and `generation_config.json` can disagree on a sampling value** (Muse Spark Open: the card says
  "top_k = disabled", `generation_config.json` sets 64). Rule 2 follows `generation_config.json`, which is also what
  vLLM applies as the server default when Inspect drops `--top-k`; say so in the README's protocol note.
- **A compressed-tensors checkpoint with an activation scheme vLLM does not dispatch loads without error** and runs
  with the wrong activation precision (an MXFP4-weight / FP8-activation MoE loaded as W4A16 before
  vllm-project/vllm#59815). Check the quantization method vLLM logs at start-up against the checkpoint's
  `quantization_config` before the smoke test.
- **MRCR's `max_model_len` HF auto-detect misses nested `text_config.max_position_embeddings`.** Qwen3.8
  multimodal configs put the window under `text_config`, so `infer_vllm_max_model_len`'s Hugging Face step returns
  nothing; the `/v1/models` fallback only helps if the server is already up and reachable during task build. Pass
  `-T max_model_len=<served window>` explicitly (e.g. 262144 for the native window).
- **Do not export `HF_HUB_OFFLINE=1` to the Inspect client for MRCR.** The server needs it so DP workers do not 429
  the Hub while loading a pre-cached checkpoint, but the client still loads `openai/mrcr` (and the model tokenizer)
  through Hugging Face. Scope offline mode to the server subprocess only. Also leave the client's `HF_HUB_CACHE` /
  `HF_HOME` alone when the server points at a shared host cache that is not writable by the eval user: the client
  still needs a writable hub home for `openai/mrcr` and tiktoken; pointing it at a read-only shared cache breaks
  dataset download. Symlink only the model weights into the client's hub cache if needed.
- **Unset HTTP(S)_PROXY for the MRCR client when the proxy 403s Hugging Face / tiktoken.** Corporate proxies that
  allow the Hub for the server path can still reject the client's tokenizer bootstrap. Clear proxy vars for the
  Inspect process (not necessarily for the whole shell).
- **MRCR concurrency follows the KV pool, same as agentic tasks.** On one B200 at `max_model_len=262144`, BF16
  Qwen3.8-27B holds ~1.83M KV tokens per DP replica (~7× at full window) while the NVFP4 sibling holds ~4.43M;
  this dir used `--max-connections 16` for BF16 vs 32 for NVFP4. Read "GPU KV cache size" in serve.log before
  copying the sibling's concurrency.

## 10. Conventions established in the task directories

Sections 1-9 set the defaults. The existing task dirs depart from some of them on purpose, and each README
says why. When you touch a task, follow the same task under the other model instead of the defaults, and do not
revert these departures:

- **No build, lint or unit tests.** The only test is the smoke run from inside a task dir
  (`EPOCHS=1 ./local_orchestrator.sh --limit 5`; use `--limit 2` for agentic tasks). It needs the GPUs, and Docker
  for sandboxed tasks.
- **Epoch defaults of existing dirs vary** (`EPOCHS` in `run_eval.sh`). They record what produced each README's
  results and predate the one-epoch default of section 3; new runs use 1 epoch unless the user asks for more:
  - 3: gpqa-diamond, livecodebench, terminal-bench-2.1, bfcl.
  - 8: aime-2026 (only 30 problems).
  - 1: mmlu-pro (12k samples), swe-bench-verified, swe-bench-pro, deepswe-1.1, tau2-bench, tau3-bench.
  - 1 for every task of the four Qwen3.8-Flash-Next dirs (RedHatAI, Inferact and nvidia NVFP4, Qwen BF16), at the
    user's request.
- **Two-phase generous-time rerun (Terminal-Bench 2.1, GLM-5.3-MXFP4xFP8 dirs).** At the user's request, the samples
  that hit the first pass's time limit (all failed) plus one leaked sample were rerun once at a higher
  `--time-limit` via `inspect eval --sample-id <comma list>`; each task counts once, the rerun result replacing the
  first-pass result for those samples. The README documents both phases. Not a default — ask the user before
  reusing this protocol on another task or model.
- **Sampling flags.** Inspect's `vllm` provider drops `--top-k` from the request. The value that applies is the
  server default vLLM takes from the checkpoint's `generation_config.json` (logged at vLLM startup).
- **Qwen3.8-Flash-Next dirs** (8x B200 host):
  - Serving: vLLM nightly, TP=1 × DP=4 set in `serve.sh`; the agentic dirs override it to TP=4 × DP=1 in
    `local_orchestrator.sh`. The RedHatAI dirs use GPUs 0-3 on port 8000, the Inferact dirs GPUs 4-7 on port 8001,
    the nvidia dirs GPUs 0-3 on port 8002, and the Qwen BF16 dirs GPUs 4-7 on port 8003.
  - Qwen BF16: `serve.sh` sets TP=4 × DP=1 (the recipe needs at least 423 GB of VRAM per replica). That replica has
    room for 9.5M KV-cache tokens against 20.4M for an NVFP4 TP=4 replica, and with about 100 agent sessions it
    re-prefilled every history each turn. So its swe-bench-pro and deepswe-1.1 dirs run one TP=8 replica on all 8
    GPUs (17.1M tokens), and terminal-bench-2.1 runs 45 concurrent samples instead of 89. The superseded runs are
    kept in `logs/superseded_kv_thrash/`.
  - Model loading: vLLM keeps the 51B n-gram table in pinned host memory. Start the two servers one after the
    other, and keep `VLLM_ENGINE_READY_TIMEOUT_S` raised.
  - RedHatAI's server venv carries the Qwen4Exp compressed-tensors PLE fix of vllm-project/vllm#59431 on top of a
    nightly that predates its merge (see its README). vLLM 0.32.0 and later releases include the fix.
- **Muse-Spark-Open dirs** (`RedHatAI_Muse-Spark-Open-816B-A42B-MXFP4xFP8`, 453 GB checkpoint): `serve.sh` sets
  TP=8 × DP=1 with `--enable-expert-parallel` (the recipe's single-node strategy) for every task, port 8000, 1M
  context, `muse` reasoning and tool parsers, `--language-model-only` with zero multimodal limits, the Python frontend
  (the recipe's Rust frontend needs the vllm-rs binary, which a source-built image may lack), and no DSpark speculative
  decoding (an opt-in recipe feature, default only for Meta's NVFP4 checkpoint). Reasoning effort is a top-level `reasoning_effort` request
  field (low/medium/high/xhigh, xhigh default) with no enable_thinking kwarg; `--reasoning-effort xhigh` in
  `run_eval.sh` and `--default-chat-template-kwargs '{"reasoning_effort": "xhigh"}'` in `serve.sh` set it on both
  sides. Sampling is temperature 1.0, top_p 1.0, top_k 64 (generation_config.json; the card says top_k disabled).
  Muse Spark support is not in a vLLM release, so `local_orchestrator.sh` runs `serve.sh` inside a podman image built
  from the vLLM branch (see `local_env_notes.md`) through `tools/vllm_server.sh`; the dirs were created on a host without the GPUs to run them and
  have no results or requirements files yet.
- **GLM-5.3-Flash pair** (`zai-org_GLM-5.3-Flash`, the native FP8 model, and `RedHatAI_GLM-5.3-Flash-MXFP4xFP8`, its
  MXFP4-weight / FP8-activation quantization): two copies of deepswe-1.1, swe-bench-verified and terminal-bench-2.1,
  written on 2026-10-27 for a like-for-like comparison on one 8x H200 host. Per task the four files are identical in
  the two dirs apart from the `MODEL` default and the checkpoint comments; keep them that way (an edit to one copy goes
  into the other). `serve.sh` takes the base recipe's Hopper flags (one TP=8 replica, MTP with 5 draft tokens, `glm47`
  parsers, `--no-enable-flashinfer-autotune`) with the KV cache pinned to BF16 (the recipe's `--kv-cache-dtype fp8` is
  Blackwell-only for this model), `--max-model-len 262144` and `--language-model-only`; served from the
  `docker.io/vllm/vllm-openai:v0.32.0` podman image (the first release whose registry carries `glm5_next`) through
  `tools/vllm_server.sh`, the Muse-Spark-Open way; concurrency defaults copied from the
  `RedHatAI_GLM-5.3-Flash-NVFP4` runs (48 / 48 / 89) with a note to check the KV pool in serve.log first, since BF16 KV
  halves it. The MXFP4xFP8 checkpoint was not public on 2026-10-27 (uploading under `inference-optimization/`), so
  the READMEs ask to verify its `quantization_config` and `generation_config.json` and to record the revision before
  the run. No results or requirements files yet. The `RedHatAI_GLM-5.3-Flash-NVFP4` dirs (B200, vLLM nightly, FP8 KV,
  TP=4 for swe-bench-verified) are a different run and not part of the pair.
- **Serving topology for multi-turn tasks.** swe-bench-*, deepswe-1.1, bfcl, tau2-bench and tau3-bench
  export `TP=8 DP=1` in `local_orchestrator.sh` instead of TP=2 × DP=4. vLLM's prefix cache is per replica, and
  DP routes each turn to a random replica, so agent histories would be prefilled again on every turn. `serve.sh`
  keeps the recipe defaults; the override lives only in the orchestrator.
- **User simulator (tau2/tau3-bench).** Qwen/Qwen3.8-27B plays the customer for every model. The Qwen dirs reuse
  the eval server. The Gemma dirs start a second server from `assets/serve_user_model.sh` on GPUs 4-7 and run
  the model under test at TP=4 on GPUs 0-3. This is the one accepted exception to "one task directory, one
  server".
- **Docker for agent tasks.** `local_orchestrator.sh` exports `DOCKER_HOST` pointing at the rootless socket
  (`/run/user/$UID/docker.sock`); the image store location is host-specific (see `local_env_notes.md`). Harbor tasks pass
  `-T override_cpus=0`, because rootless Docker rejects CPU caps. In tau3-bench, containers reach the host's
  vLLM through `CONTAINER_HOST_IP`, not localhost.
- **Per-model assets.** Every Gemma dir carries two assets:
  - `assets/tool_chat_template_gemma4.jinja`, the recipe's template, used by `serve.sh`.
  - `assets/generate_config.json`, which turns thinking on per request.

  Qwen dirs need neither, because `--reasoning-effort xhigh` does the job. The Gemma recipe URL is
  `https://recipes.vllm.ai/Google/gemma-4-31B-it`: capital `G`, although the HF org is `google`.
- **Each dir pins its own client packages.**
  - inspect_evals: always from `pip install git+https://github.com/neuralmagic/inspect_evals.git` (section 4).
    Some dirs pin a fix branch or commit of that fork (bfcl, older swe-bench-verified) instead of `main`.
  - inspect_harbor: 0.7.5, 0.7.6, 1.0.0, or a git commit (tau3-bench; the Flash-Next swe-bench-pro and
    deepswe-1.1 dirs).

  Before a run, create the client environment from the dir's `client_requirements.txt` and README Setup, or check
  that an environment you already have matches it.
- **Harbor fork pins.** livecodebench, swe-bench-pro and deepswe-1.1 load their tasks through
  `assets/harbor_registry.json`, which has one entry per task, each pinned to a commit of
  `github.com/eldarkurtic/harbor-datasets`. `run_eval.sh` points `inspect_harbor/harbor -T registry_path=...` at
  that file.
- **SWE-style tasks can leak the solution to the agent.** Only the Qwen3.8-Flash-Next dirs carry the fixes;
  the other dirs of deepswe-1.1, swe-bench-pro and swe-bench-verified ran without them. See section 11.
- **Every task exists once per model.** Do not mirror an edit into the other model's copy unless asked.
- **Coverage matrix.** When you add a task, set ✅ in the `Repo` column of `eval_coverage_matrix.md` in both
  places: the Benchmarks table and the per-subclass detail table.

## 11. Solution leaks: fixes to use and the audit after every run

An agent with tools can take a task's solution instead of solving it: fetch it from the internet, or read it from
the project's future git history inside the image. Models differ in how hard they try, so close every leak a task
allows before the run, and audit the trajectories after it.

**Fixes.** Copy deepswe-1.1, swe-bench-pro and swe-bench-verified from the Qwen3.8-Flash-Next dirs
(`Qwen_Qwen3.8-Flash-Next`, `RedHatAI_Qwen3.8-Flash-Next-NVFP4`, `Inferact_Qwen3.8-Flash-Next-NVFP4`,
`nvidia_Qwen3.8-Flash-Next-NVFP4`). The `Qwen_Qwen3.8-27B`, `google_gemma-4-31B-it`, `RedHatAI_Qwen3.8-27B-MXFP4`
and `amd_Qwen3.8-27B-Quark-AWQ-MXFP4` copies of these tasks ran before the fixes; don't reuse their client setup.

- **deepswe-1.1.** Every `task.toml` sets `network_mode = "no-network"` for `[agent]` and `[verifier]`, but stock
  inspect_harbor (1.0.0 and earlier) applies only `[environment].network_mode`, so agents silently get internet.
  - Fix: install inspect_harbor from the fork `eldarkurtic/inspect_harbor@cb61925` (branch
    `fix/enforce-agent-network-mode`), which applies the network mode per phase. Create the client environment
    from a Flash-Next deepswe-1.1 dir's `client_requirements.txt`, which pins that commit; `local_env_notes.md`
    gives the install caveats.
  - Upstream: meridianlabs-ai/inspect_harbor#200. Once a release carries it, switch back to the release.
  - Use the v1.1 task images. The v1.0 images keep future `origin/*` refs.
- **swe-bench-pro.** The published tasks give the agent internet, and their images keep the project's future git
  history, including the gold commit. Agents downloaded newer upstream releases, and one run was withdrawn.
  - Fix: the inspect_harbor fork above, plus `assets/harbor_registry.json` pinned to
    `eldarkurtic/harbor-datasets@d81053e87b`. That pin carries harbor-framework/harbor-datasets#263: the agent is
    offline, the history is removed, and the gold tests ship with the verifier.
  - The older pin `2494eda6ae` has only the verifier fixes (#258), not the leak fixes.
- **swe-bench-verified.** inspect_evals' default Epoch AI images keep every release tag of the project, so the
  reference fix is readable offline (`git show <later tag>:<file>`).
  - Fix: use the official images, `-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`
    (UKGovernmentBEIS/inspect_evals#2632). The agent has no internet by the task's default.
- **terminal-bench-2.1.** Every task sets `allow_internet = true`, and several verifiers `apt-get install` at test
  time, so the network can't be closed.
  - The task files, including `solution/solve.sh` and `tests/`, are public: `harbor-framework/terminal-bench-2-1`
    on GitHub, and mirrors on the HF Hub (find them with `huggingface.co/api/datasets?search=terminal-bench`).
  - Only the audit catches it when an agent downloads its own task's files.
  - GLM-5.3-MXFP4xFP8 (2026-10): one sample (`mteb-retrieve`) web-searched its instruction's distinctive phrases,
    found the HF mirror `harborframework/terminal-bench-2.1`, downloaded its own `solution/solve.sh` and submitted
    a verbatim copy. Blocking is not possible without breaking the task (its own reference solution needs
    `huggingface.co` for the embedding model — the same domain as the mirror), so the real fix is upstream:
    exclude `solution/` from the public mirror (issue drafted for harbor-framework/terminal-bench-2-1). Protocol
    chosen with the user for that run: rerun the leaked sample and every time-limit failure once at a higher
    `--time-limit` via `--sample-id`, audit the rerun, and count any repeat leak as a failure.
- **livecodebench.** The tasks set no network policy, so the agent has internet (the fork changes nothing). Audit
  for lookups of the problem's published solutions.
- **Any new task with tools.** Check each `task.toml` network policy against what the runner enforces (after a run,
  Harbor's task files are cached under `~/.cache/harbor/tasks/packages/` and `~/.cache/inspect_harbor/tasks/git/`;
  tasks loaded from a registry pin can be read from the pinned commit with a shallow fetch). Check the
  images for refs and tags newer than the base commit (`git log --all`, `git tag`, unreachable objects). Search for
  public copies of the task files. Then add the task to this list.
- Validate a dataset or verifier change with `--solver inspect_harbor/oracle` on every image family. Image
  families differ even within one repo; for example, the Alpine images' BusyBox `tar` lacks GNU options.

**Audit after every run, before you fill Results.** Scan every tool call of every sample in the final `.eval`
logs. Read `sample.events`, not `sample.messages`: context compaction drops early turns from the messages.

`tools/leak_audit.py` runs this scan (plus the canary check below) over `.eval` files or log dirs and prints every hit
with the sample's score; run it with the client venv's python, `--offline` for tasks whose agents must have no network,
`--out hits.jsonl` to keep the hits for hand review. It finds the hits; deciding which are leaks is still yours.

```python
import json, re
from inspect_ai.log import read_eval_log
NET = re.compile(r"pip3? (download|index)|git (fetch|clone|pull|ls-remote|remote add)|git log [^|;&\n]*--all"
                 r"|github\.com|githubusercontent|gitlab\.com|huggingface\.co|pypi\.org|registry\.npmjs|npm (view|pack)"
                 r"|proxy\.golang|go (get|install|mod download)|crates\.io|curl |wget |urlopen|requests\.get", re.I)
log = read_eval_log("logs/<file>.eval", resolve_attachments=True)
for s in log.samples:
    for ev in s.events:
        if ev.event == "tool" and NET.search(cmd := json.dumps(ev.arguments)):
            print(s.id, s.epoch, {k: v.value for k, v in (s.scores or {}).items()}, cmd[:300], str(ev.result)[:300])
```

- Networked tasks give hundreds of hits; review them by hand.
- Judge network attempts by their output, not by shell exit codes: `curl -sI URL | head -3; echo $?` reports
  `head`'s status (0) even when curl could not resolve the host (with `-s` curl prints nothing on failure). The
  reliable tells are empty output, `http_code 000`, exit 6, or a python `urlopen` name-resolution error.
- Scanning tool-call *results* for the benchmark's canary GUID (TB: `terminal-bench-canary GUID …`) finds task-file
  acquisition through channels your URL regex missed. Confirm the source before calling it a leak: files the task
  ships inside the agent's own container (working files, deliberately visible `environment/tests/`) are not leaks;
  network fetches of `solution/` or `tests/` are.
- A leak is any case where the agent obtained:
  - its own task's files (`solution/`, `tests/`);
  - the project's upstream work past the task's base commit: GitHub issues, PRs, commits or files, or a clone or
    fetch of the upstream repo;
  - a newer release of the project itself (`pip download`, `npm view`/`npm pack`, `go get ...@latest`);
  - the reference fix from git history in the image.
- Not leaks: installing tools (linters, formatters, test runners), dependency downloads, `git clone /app` (a local
  copy), URLs the agent writes into code or docs, and Go import paths.
- In a task whose agent should be offline, any successful outbound request means the fix is not active. Stop, and
  check the client environment against the dir's `client_requirements.txt`.
- For each leak, check whether the sample passed and whether it used what it fetched. Report every leak to the
  user in chat: sample id, epoch, what was fetched, the score, and the score with leaked samples counted as
  failures. The user decides whether and how the README reports it.

## 12. Reflection procedure (mandatory)

Run it when you finish an eval of a model (final run done, README filled) or any other meaningful piece of work: a
new task, an infrastructure problem solved, an upstream fix, a move to a new host or tool version. Do it before you
report back to the user. These notes are the knowledge base every later agent starts from, on any server, so each
lesson you skip will be paid for again.

1. **Collect.** List what went wrong or cost time, what surprised you, what you had to find out by experiment, and
   what you would do differently. Include what the user corrected or decided along the way.
2. **Keep what generalizes:** anything that would help another agent with another model on another server. Drop
   one-off noise.
3. **Write it where the next agent reads it:**
   - rules, eval protocol, scoring, Inspect / vLLM / Harbor behaviour, leaks and audits: this file (section 9 for
     gotchas, 10 for conventions, 11 for leaks, or a new section);
   - serving, caches, Docker, images, client environments: `local_env_notes.md`;
   - one task's upstream defects: that task's README, under Patches;
   - facts about this machine only: `local_env_notes.local.md`, which is not shared.
4. **Write it general.** Leave out paths, user names, venv locations, hostnames and other tenants. Name the tool,
   version and symptom, and give cause and fix with numbers where you have them, so others recognize the problem.
5. **Build on what is there.** Update or correct an existing note rather than adding a near-duplicate, and delete
   notes that turned out wrong. Keep READMEs to results (section 3): the lessons go in the notes, not the README.
6. **Report.** Tell the user in chat which notes you added or changed; committing them is their job.

The internet-access and cheating findings of section 11 show why. They were first recorded only in a few task READMEs,
so later runs of the same tasks on another host copied an older dir and ran without the fixes.

## 13. Speculative decoding benchmarks

A spec-decode dir measures draft-model acceptance on a real workload. Report the score next to the acceptance
numbers, as evidence that serving did not break the model, but the acceptance analysis is the deliverable.

- Layout and naming. Use `<org>_<draft-model>/<category>/<task>/`, for example
  `RedHatAI_Qwen3.8-27B-speculator.dspark/agentic-and-coding/terminal-bench-2.1/`. The directory is named after
  the speculator because the speculator is the artifact under test. Copy the dir of the same target model's
  non-spec run and change only serving and analysis, so that task, subset, scaffold, sampling, epochs and limits
  match. Record the run in the task README, and tick the `Spec` column of `eval_coverage_matrix.md` if that
  table has one.
- Serving. Serve the target with its recipe flags and add the draft model with
  `--speculative-config '{"method":"<method>","model":"<draft>","num_speculative_tokens":K}'`. Take the method
  and K from the recipe's spec-decode section or from the speculator card, and record the URL. Methods belong to
  the engine (`dspark`, `eagle3`, `ngram`, and others), so confirm the installed vLLM supports the one you need
  (`vllm/config/speculative.py`) and that the draft architecture is in the model registry. The draft's TP must
  be 1 or equal to the target's TP.
- Per-request metrics. Start the server with `--per-request-spec-decode-metrics detailed`, available from vLLM
  0.32.0 and marked experimental. Every chat completion then carries `metrics.speculative_decoding` with the
  mean acceptance length, a dense `acceptance_histogram` counting how many verify steps accepted exactly j
  draft tokens (length K+1, bonus token excluded), the step and token counts, and the
  `per_step_accepted` and `per_step_drafted` arrays that only `detailed` emits. Only single-sequence requests
  (`n == 1`) are populated, and Inspect always sends `n == 1`. The option is independent of
  `--disable-log-stats`, so leave vLLM's aggregate acceptance line enabled as a cross-check. Set
  `--served-model-name` to the speculator id, so the eval log names the deployment under test.
- Topology and concurrency. The draft adds weights and compute, which changes the KV pool. Re-read "GPU KV cache
  size" in serve.log and keep `concurrency * max_model_len` at or below about 1.2 times the pool (section 9). A
  small pool also limits how many long-context trajectories you can observe, which is a common reason to run a
  spec-decode study in the first place.
- Client. Add `--log-model-api` to `run_eval.sh`. Without it Inspect keeps only the first few model calls per
  model plus errors, which is not enough for acceptance analysis. With it, every raw request and response is
  kept, and `ModelCall.response` carries the metrics block, because the OpenAI-compatible provider stores
  `completion.model_dump()` and the non-standard top-level `metrics` field survives. Logs grow quickly, to
  hundreds of MB for a few hundred agent turns, so keep `--log-dir` on local disk.
- Analysis. Use the task's `analyze_spec_decode.py`. `collate` reads every `*.eval` with
  `read_eval_log(..., resolve_attachments=True)` and writes one row per model call holding the task, epoch,
  turn, usage and every acceptance field. `analyze` turns that CSV into `report.md`, `report.json` and slice
  CSVs. Read `sample.events` rather than `sample.messages`, because compaction drops early turns from the
  messages. Every metric below comes from the histogram and the counts already in the response.
  - MAL is `1 + accepted_draft_tokens / num_spec_steps`, where the leading 1 is the always-accepted bonus
    token. Report the pooled value over all steps (micro) as the headline, and the unweighted mean of the
    per-call values (macro) beside it. Micro weights every verify step equally; macro weights every request
    equally.
  - Per-position overall acceptance is `Σ_{i>j} hist[i] / steps`, the probability that position j is accepted.
    Draft tokens are accepted in prefix order, so this is a tail probability, and the sum over j plus 1
    reproduces MAL. That is the model-card convention.
  - Per-position conditional acceptance is `overall[j] / overall[j-1]`, the probability that position j is
    accepted given that j-1 was. It is the hazard rate. Either rate follows from the other, because the overall
    rate is the running product of the conditionals.
  - Thinking against response needs no tokenizer and no extra endpoint. The reasoning block is a prefix of the
    completion, and its token length is `usage.completion_tokens_details.reasoning_tokens`. Each verify step
    emits its accepted drafts plus one bonus token, so the step spans follow from `per_step_accepted`; check
    that the `Σ(a_i + 1)` sum equals `completion_tokens`. Report MAL and both per-position rates per
    region. The two regions can differ substantially.
- Report no timing or throughput metric. Tokens per second, wall time and speedup all depend on the host, the
  concurrency and the topology, so they describe only the run that produced them. Acceptance is the result that
  transfers to another host.
- Comparability. Speculative decoding changes what generation costs, not the task or the sampling. Report the
  score next to the non-spec run's score and state whether the two agree. A large drop means serving broke
  something, so check `stop_reason`, tool-call parsing, and that reasoning is parsed into a separate content
  part. Say that the numbers are specific to the engine version and to the draft checkpoint, and that
  `--per-request-spec-decode-metrics` is experimental.
