# RedHatAI/Qwen3.8-Flash-Next-NVFP4 — SWE-bench Pro

| | |
|---|---|
| Task | `inspect_harbor/scale_ai_swe_bench_pro` (731 tasks, Harbor hub package digest `sha256:88411d32…`, loaded through `assets/harbor_registry.json`, which pins the tasks to a fork commit that closes the solution leaks, see Patches) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one Docker image per task built from `FROM jefzda/sweap-images:<instance>` (Docker Hub) plus a git reset, bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |
| Task source | https://hub.harborframework.com/datasets/scale-ai/swe-bench-pro |

**Protocol note.** The Qwen/Qwen3.8-Flash-Next card reports 62.5 (Claude Code harness, temp 1.0, top_p 0.95, 256K
context, on a corrected task set). `inspect_harbor` solves every task with Inspect's generic ReAct agent (`bash`,
`python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction; no message limit), not the
Claude Code harness; a single `--time-limit 14400` wall-clock guard is used. Containers run without Harbor's CPU
cap (`override_cpus=0`) and with `inspect_harbor`'s 6 GB memory floor. One epoch (pass@1). Unlike the other
SWE-bench Pro dirs of this repo, the tasks run from a dataset commit whose agent phase has no internet and no git
history past the base commit (the published tasks leak the reference solution, see Patches); the verifier keeps
internet, as in the upstream harness. The numbers are therefore not comparable to those dirs, nor directly to the
model card. Checkpoint: NVFP4 (W4A4) MoE experts, compressed-tensors format (LLM Compressor), revision `863f54a`.

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
`--sandbox-prebuilt` to skip that step. The 731 task images take about 3 TB, and each is built from a Docker Hub
base image (100 anonymous pulls per hour per IP), so pull the bases ahead of time at that rate, or through
`mirror.gcr.io` after checking that it serves the same manifest digest as Docker Hub.

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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=56, max_sandboxes=56, time_limit=14400s, client_timeout=7200, override_cpus=0, TP=4 DP=1, epochs=1; 731 tasks in 5 runs of 127 / 21 / 128 / 138 / 317 tasks; errored samples retried once with `inspect eval-retry` (completed samples kept)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.636 (462 of 726 scored tasks) |
| stderr (binomial) | 0.0179 |
| resolved rate counting sample errors as failures | 0.632 (462 of 731) |
| samples | 731 tasks, 1 epoch, in 5 runs: 127 + 21 + 128 + 138 + 317 |
| time-limit hits (14400 s) | 1 |
| turns truncated at max_tokens | 1 |
| tool calls that hit the tool timeout | 310 of 56838 (0.55%) |
| context compactions | 14 |
| sample errors (excluded from the score) | 5 (instance_gravitational__teleport-e6895d8934f6e48434103486990, instance_gravitational__teleport-2be514d3c33b0ae9188e11ac997, instance_gravitational__teleport-7744f72c6eb631791434b648ba4, instance_gravitational__teleport-87a593518b6ce94624f6c28516c, instance_gravitational__teleport-02d1efb8560a1aa1c72cfb1c08e) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 3,666,037 | 50,900 | 35,940 |
| median | 2,739,510 | 46,804 | 32,890 |
| max | 31,188,016 | 182,246 | 135,079 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 57, mean
62, max 261. Wall time per sample: median 2,428 s, mean 2,940 s, max 14,511 s. Summed run time about 17.7 h across
the runs on 4x B200 (one TP=4 replica, 56 concurrent samples), with the other model's runs on the other four GPUs
of the host at the same time.

Per repository (resolved / tasks): ansible 64/96, internetarchive 67/91, flipt-io 37/85, qutebrowser 56/79, gravitational 36/71, protonmail 44/65, future-architect 38/62, navidrome 32/57, element-hq 33/56, nodebb 35/44, tutao 20/20.

The reference solution (oracle) was run on 731 tasks and fails 7 of them (ansible__ansible-de5858f48dc9e1ce9117034,
element-hq__element-web-aec454dd6feeb930, future-architect__vuls-bff6b7552370b55ff,
gravitational__teleport-2b15263e49da5625, gravitational__teleport-87a593518b6ce946,
gravitational__teleport-baeb2697c4e4870c, nodebb__nodebb-00c70ce7b0541cfc94afe5679). Oracle-failing tasks that are
known failures of the published tasks were run and scored like any other task.

21 tasks (the 20 tutanota tasks and gravitational__teleport-c1b1c6a1541c478d) ran in a separate run with few
containers alive: the host's per-user inotify instance limit (128, shared by all rootless containers) made
tutanota's build-server tests fail with EMFILE under load, so `npm test` stopped before the suites the verifier
checks. Their oracle passes in that setting.

## Patches for upstream

The published tasks leak the reference solution to the agent in two ways:

- **Internet.** No task sets `[environment].network_mode`. Agents downloaded a newer release of the project under test
  (`pip download ansible-core==2.19.0`) or its current source (`curl raw.githubusercontent.com/<repo>/devel/...`), which
  already contain the fix.
- **Git history.** The task images keep the upstream clone: `origin/*` branches, later tags and the gold commit
  itself. The Dockerfile's `git reset` only moves `HEAD`.

A withdrawn first run with the published tasks found that 92-95% of samples used one of the two, at a 0.93-0.97
"resolved" rate.

The fix is branch [`swebenchpro/no-solution-leak`](https://github.com/eldarkurtic/harbor-datasets/tree/swebenchpro/no-solution-leak)
of https://github.com/eldarkurtic/harbor-datasets (commit `d81053e87b`, on top of `swebenchpro/verifier-env-fixes`,
i.e. harbor-framework/harbor-datasets#258). `assets/harbor_registry.json` pins every task to it. Upstream PR:
[harbor-framework/harbor-datasets#263](https://github.com/harbor-framework/harbor-datasets/pull/263). It ships the
gold test files as plain files under `tests/gold/` (that repo's `.gitignore` excludes archives), with the same
content as the `tests/gold_tests.tar` of the commit pinned here. Per task:

- `task.toml`: `[agent] network_mode = "no-network"`. The verifier keeps network, as in the upstream harness: 61
  reference fixes change dependency manifests (52 of them Go modules, mostly flipt and vuls) whose packages the tests
  download, and flipt's `Test_SourceSelfSignedSkipTLS` connects to `self-signed.badssl.com`. inspect_harbor 1.0 applies
  only `[environment].network_mode`, so the client venv installs branch
  [`fix/enforce-agent-network-mode`](https://github.com/eldarkurtic/inspect_harbor/tree/fix/enforce-agent-network-mode)
  of https://github.com/eldarkurtic/inspect_harbor (commit `cb61925`), which applies `[agent]` and `[verifier]`
  `network_mode` per phase: the container starts with `network_mode: none` and is moved onto Docker's default bridge
  after the agent has finished, before the verifier runs. Upstream PR:
  [meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200).
- `environment/Dockerfile`: after the reset, drop remotes, tags that are not ancestors of `HEAD`, other branches,
  stash/notes/replace refs and reflogs, then `gc --prune=now`. The build fails if any commit not reachable from `HEAD`
  or any unreachable commit remains.
- Gold tests: `tests/gold_tests.tar` holds exactly what the verifier's `git checkout <gold> -- <files>` produces. It
  was created by that same checkout in the base image, or from a blob-less fetch of the gold commit where the image
  was not available. `tests/test.sh` extracts it with a plain `tar -xf` (the tar holds file entries only; the Alpine
  images' BusyBox tar has no `--no-overwrite-dir`) instead of checking out the gold commit, which is no longer in the
  image. `/tests` enters the container only at scoring time.
- nodebb tasks whose `run_script.sh` runs `npm install` at test time: the same packages are installed at build time
  and `npm_config_offline=true` is set.

Every task was run with the reference solution (`--solver inspect_harbor/oracle`; same dataset commit, client and
network policy) before the models ran on it. Tasks the oracle fails that are not among the known oracle failures of
the Qwen3.8-27B SWE-bench Pro README were held back from the model runs and are listed under Results. Also consumed: the verifier fixes of #258 (ansible-test
`RLIMIT_NOFILE`, stale `/etc/pip.conf`) and inspect_harbor 1.0's non-login verifier shell (meridianlabs-ai/inspect_harbor#174).

Host-side: the 91 openlibrary base images contain files owned by UID 120012366, beyond rootless Docker's 65,536
subordinate UIDs. They were loaded with those owners remapped to root, with layers rewritten from the registry
(the processes in the task container run as root).

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
