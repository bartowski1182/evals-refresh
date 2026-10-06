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
   in this repo. If a task is broken, fix it upstream (PR from a fork under github.com/eldarkurtic) and
   consume the fix from the fork branch (pip install from git, or a Harbor registry pin to the fork
   commit); document it under the README's "Patches for upstream" section. No local patch scripts.
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
  what ran).
- `serve.sh` and `run_eval.sh` are plain, readable bash with env-var overrides
  (`MODEL`, `PORT`, `TP`, `DP`, `EPOCHS`, `LOG_DIR`, `MAX_TOKENS`, `MAX_CONNECTIONS`, ...). No
  defensive scaffolding, no logging frameworks, no argument parsers.


## 3. Rules of engagement

- **Never `git commit` or `git push`.** Leave changes in the working tree and say what changed.
  Committing is the repo owner's job.
- **Keep every agent instruction in this file.** `CLAUDE.md` only imports it (`@AGENTS.md`), so Claude Code and
  other agent harnesses read the same rules. Do not add content to `CLAUDE.md`, including what `/init` proposes;
  put conventions here, environment notes that apply to any host in `local_env_notes.md`, and facts about one
  machine in `local_env_notes.local.md`.
- **Never put secrets in the repo.** `HF_TOKEN`, API keys and other secrets should be local on
  the user's machine. 
- **Ask before deciding anything that changes what is measured**: protocol, subset, judge model,
  time limits, epochs. Do not silently narrow or widen a task. Infrastructure choices (TP/DP,
  concurrency, disk) are yours to make and to document.
- **Always smoke-test first** (`EPOCHS=1 ./local_orchestrator.sh --limit 5`, or `--limit 2` for
  agentic tasks), read the samples, and only then launch the full run. Verify in the log that
  reasoning was parsed into a separate `reasoning` content part, that `stop_reason` is `stop`
  (not `max_tokens`), and that the config you intended is what `eval.model_generate_config` recorded.
- **avg@3 is the default** (`--epochs 3 --epochs-reducer mean`), matching the vendors' cards.
  Deviate only with the user's agreement and say so in the README.
- **Results go in the README as data, not narrative.** Fill the Results template (below). Do not
  describe failed attempts, ablations, or how you got there in the README. README only contains
  final results, everything else should be discussed with the user in chat.
- **One task directory, one server.** Do not add multi-model orchestration, judges on split GPUs,
  or background daemons. If a task needs a judge, run judging as a separate phase
  (`inspect eval --no-score`, then `inspect score --model-role grader=...`) and document it.

## 4. Environment on the reference host (adapt in `local_orchestrator.sh`)

Three venvs, deliberately separate:

| venv | used by | install |
|---|---|---|
| `~/.venvs/vllm` | `serve.sh` | `pip install vllm` |
| `~/.venvs/inspect` | `run_eval.sh` for `inspect_evals` tasks | `pip install inspect-evals openai` |
| `~/.venvs/inspect-harbor` | `run_eval.sh` for `inspect_harbor` tasks | `pip install inspect-harbor httpx2 "openai==3.14.0"` |

`inspect-harbor` depends on `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible
providers need `openai>=3`; installing `openai==3.14.0` last overrides the pin and the resulting
pip warning is harmless (Inspect's agent path never imports litellm). `httpx2` is imported by
recent `inspect-ai` but not declared.

## 5. How to run an existing task

```bash
cd <org>_<model>/<category>/<task>
EPOCHS=1 ./local_orchestrator.sh --limit 5     # smoke test; inspect the samples first
./local_orchestrator.sh                        # full avg@3
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
| metric | value |                       accuracy (avg@3), stderr, per-epoch, samples,
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

1. Create `<org>_<model>/<category>/<task>/` by copying the closest existing task dir.
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

1. Find it in `eval_coverage_matrix.md`. `inspect_evals` tasks run in `~/.venvs/inspect`;
   `inspect_harbor` tasks (`Where` = `inspect_harbor`) need `~/.venvs/inspect-harbor` and Docker.
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
   for multi-turn history inside the context window (65536 was used with a 262k window).
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

## 10. Conventions established in the task directories

Sections 1-9 set the defaults. The existing task dirs depart from some of them on purpose, and each README
says why. When you touch a task, follow the same task under the other model instead of the defaults, and do not
revert these departures:

- **No build, lint or unit tests.** The only test is the smoke run from inside a task dir
  (`EPOCHS=1 ./local_orchestrator.sh --limit 5`; use `--limit 2` for agentic tasks). It needs the GPUs, and Docker
  for sandboxed tasks.
- **Epoch defaults vary** (`EPOCHS` in `run_eval.sh`):
  - 3: gpqa-diamond, livecodebench, terminal-bench-2.1, bfcl.
  - 8: aime-2026 (only 30 problems).
  - 1: mmlu-pro (12k samples), swe-bench-verified, swe-bench-pro, deepswe-1.1, gdpval-aa, tau2-bench, tau3-bench.
  - 1 for every task of the four Qwen3.8-Flash-Next dirs (RedHatAI, Inferact and nvidia NVFP4, Qwen BF16), at the
    user's request.
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
  - RedHatAI's server venv carries a vLLM fix from the `eldarkurtic/vllm` fork (see its README).
- **Serving topology for multi-turn tasks.** swe-bench-*, deepswe-1.1, bfcl, gdpval-aa, tau2-bench and tau3-bench
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
- **Client venvs are shared across dirs, but each dir pins different packages.**
  - inspect_harbor: 0.7.5, 0.7.6, 1.0.0, or a git commit (tau3-bench; the Flash-Next swe-bench-pro and
    deepswe-1.1 dirs).
  - bfcl and the older swe-bench-verified dirs install inspect_evals from `eldarkurtic/inspect_evals` fork branches.

  Check the venv against the dir's `client_requirements.txt` and its README Setup before a run.
- **Harbor fork pins.** livecodebench, swe-bench-pro and deepswe-1.1 load their tasks through
  `assets/harbor_registry.json`, which has one entry per task, each pinned to a commit of
  `github.com/eldarkurtic/harbor-datasets`. `run_eval.sh` points `inspect_harbor/harbor -T registry_path=...` at
  that file.
- **SWE-style tasks can leak the solution to the agent: audit before trusting a score.**
  - Stock inspect_harbor 1.0 applies only `[environment].network_mode`. It ignores `[agent]` and `[verifier]`
    `network_mode`, so most containers have public internet. The Qwen3.8-Flash-Next `swe-bench-pro` and
    `deepswe-1.1` dirs need the `eldarkurtic/inspect_harbor` fork (`~/.venvs/inspect-harbor-fix`), which applies
    them per phase. With stock inspect_harbor, their agents silently get internet.
  - Images with future git history: the published SWE-bench Pro images, the DeepSWE v1.0 images, and
    inspect_evals' default Epoch AI images for SWE-bench Verified. Clean: the official `swebench/sweb.eval.*`
    images and the DeepSWE v1.1 images.
  - What agents did with this:
    - SWE-bench Pro: they downloaded newer upstream releases, and the run was withdrawn.
    - Terminal-Bench 2.1: one sample fetched its own task files from GitHub.
  - Scan trajectories for `pip download`, `git fetch/clone`, `curl github` and `git log --all`. The
    Qwen3.8-Flash-Next `swe-bench-pro`, `swe-bench-verified` and `deepswe-1.1` READMEs describe the fixes.
  - Validate a dataset or verifier change with `--solver inspect_harbor/oracle` on every image family. Image
    families differ even within one repo; for example, the Alpine images' BusyBox `tar` lacks GNU options.
- **Every task exists once per model.** Do not mirror an edit into the other model's copy unless asked.
- **Coverage matrix.** When you add a task, set ✅ in the `Repo` column of `eval_coverage_matrix.md` in both
  places: the Benchmarks table and the per-subclass detail table.
