# Qwen/Qwen3.8-27B — SWE-bench Pro

| | |
|---|---|
| Task | `inspect_harbor/scale_ai_swe_bench_pro` (731 tasks, Harbor hub package digest `sha256:88411d32…`, loaded through `assets/harbor_registry.json`, which pins the tasks to the fork commit carrying the verifier fixes, see Patches) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one Docker image per task built from `FROM jefzda/sweap-images:<instance>` (Docker Hub) plus a git reset, bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://hub.harborframework.com/datasets/scale-ai/swe-bench-pro |

**Protocol note.** The model card reports 61.7 (Claude Code harness, on a corrected task set). `inspect_harbor` solves every task with Inspect's generic ReAct agent
(`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction; no message limit),
not the Claude Code harness. Harbor's per-task agent budget is not applied by `inspect_harbor`; a single
`--time-limit 14400` wall-clock guard is used. Containers run without Harbor's CPU cap (`override_cpus=0`) and
with `inspect_harbor`'s 6 GB memory floor. One epoch (pass@1) instead of the repo's usual avg@3. All in-flight
samples share one vLLM server (one TP=8 replica so the prefix cache stays local; 56 samples in flight, what its
4M-token KV cache holds once histories average 45k tokens). The numbers are therefore not directly comparable to
the model cards.

## Files

- `assets/harbor_registry.json` — Harbor registry entry pinning the 731 tasks to the fork commit that carries the verifier fixes (see Patches).
- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and package digest, epochs, sampling params, per-turn max tokens, concurrency, sandbox count, time limit).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below. The client venv is the `inspect-harbor` one (see Run).
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs and TP/DP, points Docker at the rootless daemon, starts `serve.sh` in the background,
  runs `run_eval.sh`, tears down. **This is the only file you should modify** to match the
  environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and `client_requirements.txt`
are the exact `pip freeze` of the environments that produced the results below (Python 3.12, host details in their
headers): installing from them reproduces those environments; the commented alternative gives an equivalent fresh
install with the same pinned versions.

```bash
# server: vLLM 0.29.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.29.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install -r client_requirements.txt          # what produced the results (inspect-harbor 0.7.5); or the current setup:
# pip install "inspect-harbor>=1.0" httpx2 "openai==3.14.0"
```

`inspect-harbor` pulls in `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible providers need
`openai>=3`; installing `openai==3.14.0` last overrides the pin (pip prints a conflict warning that is harmless
here because Inspect's agent path never imports `litellm`). `httpx2` is imported by `inspect-ai` but not declared.

Model weights and datasets are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

Docker (with the compose plugin) must be reachable by your user. `local_orchestrator.sh` exports `DOCKER_HOST` for
a rootless daemon (`unix:///run/user/<uid>/docker.sock`, image store under `~/.local/share/docker`); with a
rootful daemon drop that export. Check the image-store disk space quoted below before the first run. Rootless
Docker needs a runc runtime with `NoNewKeyring` in `daemon.json` (one kernel key per container otherwise
exhausts a user's quota of 200) and cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

Requires Docker. **What to expect on the first run:** Harbor downloads the task definitions into
`~/.cache/harbor/tasks/packages/`, then Each task's image is built from a per-instance base image on Docker Hub (731 pulls at the anonymous limit of 100
per hour, 1 to 5 GB each). Inspect builds images serially before any sample starts, so pre-pull and pre-build them
(`docker build -t hb__scale-ai-<task> environment/`) and run in batches if the image store cannot hold all 731.
Later runs reuse the images.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), no message limit, time_limit=14400s, client_timeout=7200, tool_timeout=300s, TP=8 DP=1, epochs=1; 731 tasks in 5 runs of 101 / 79 / 307 / 214 / 33 tasks at 56 samples in flight (the 214-task run retuned live to 48)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.756 (551 of 729 scored tasks) |
| stderr (binomial) | 0.0159 |
| samples | 731 of 731 tasks, 1 epoch, in 5 run(s): 101 + 79 + 307 + 214 + 33 |
| time-limit hits (14400 s) | 0 |
| turns truncated at max_tokens | 0 |
| context compactions | 13 |
| sample errors (excluded from the score) | 2 (instance_gravitational__teleport-02d1efb8560a1aa1c72cfb1c08e, instance_gravitational__teleport-2b15263e49da562592258156983) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 3,459,141 | 43,183 | 29,351 |
| median | 2,393,603 | 38,124 | 25,653 |
| max | 24,733,351 | 213,719 | 156,133 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 56, mean 63, max 284. Wall time per sample: median 1,785 s,
mean 2,148 s, max 9,315 s. The agent ends when the model replies without a tool call (no `submit` tool is used);
the verifier then scores the container state. Total run time about 10 h 27 min on 8x H100 across the runs.

Per repository (resolved / tasks): ansible 71/96, internetarchive 75/91, flipt-io 47/85, qutebrowser 63/79, gravitational 48/74, protonmail 47/65, future-architect 53/62, navidrome 52/57, element-hq 40/56, nodebb 37/44, tutao 18/20.

The oracle solver (reference solutions, run on every batch before the model) cannot pass 8 of the 731 tasks;
they are scored like any other task (both models resolved 0 of them, except one teleport task Gemma resolved):
`ansible-de5858f4` (a required test id contains JSON and never matches the parser; 58/58 tests pass),
`element-web-aec454dd` (11 InviteDialog tests missing from the run output; 189/200 required pass),
`vuls-bff6b755` (7/7 tests pass but only 4 required ids match), `nodebb-00c70ce7` (the reference solution fails
one required test), and `teleport-87a59351`, `teleport-baeb2697`, `teleport-bb562408`, `teleport-eefac60a`
(the teleport test build exceeds Harbor's 3000 s verifier budget; timing-dependent). The 2 sample errors are
the same verifier timeout on two other teleport tasks and are excluded from the score.

`inspect_harbor`'s agent enables Inspect's context compaction; Inspect cannot read a vLLM model's context window
and falls back to a 128k-token threshold (the model's window is 262k), which compacted the history 13 times in
11 samples.

## Patches for upstream

**Known issue (results affected).** These results come from the published SWE-bench Pro tasks, which leak the
reference solution to the agent: agents have internet access, and the task images keep the project's future git
history, including the gold commit. The numbers are likely inflated and are not comparable to leak-free runs. Fixed
in [harbor-framework/harbor-datasets#263](https://github.com/harbor-framework/harbor-datasets/pull/263) (agent
offline, history removed, gold tests shipped with the verifier) together with
[meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200) (per-phase network
modes). The `*_Qwen3.8-Flash-Next*/agentic-and-coding/swe-bench-pro` dirs run with both. Rerun this task with that
setup before quoting these numbers.

Three environment defects in the published tasks make several task families ungradable as published (the reference
solutions score 0), all in the verifier scripts:

- ansible-test lowers its own soft `RLIMIT_NOFILE` to 1024 and runs `pytest -n auto`; with 384 visible CPUs xdist
  spawns 384 workers, pytest dies with `EMFILE` before running a test, and the verifier reports 0 passed tests even
  for the reference solution. Fixed by running every `ansible-test` call under `taskset -c 0-3` (bounds worker
  count only). Upstream: https://github.com/harbor-framework/harbor-datasets/pull/258.
- the images ship `/etc/pip.conf` pointing at a build-time mirror (`http://127.0.0.1:9876/`) that no longer exists,
  which breaks the tasks whose test run installs requirements. Fixed in the same PR (the file is removed first).
- `inspect_harbor` 0.7.x ran the verifier as `bash -l /tests/test.sh`; in the Go repositories' images `/etc/profile`
  resets `PATH`, `go` is not found and 0 tests run. Fixed upstream in inspect_harbor 1.0.0
  (https://github.com/meridianlabs-ai/inspect_harbor/pull/174, verifier no longer uses a login shell).

This directory consumes the dataset fix from the fork: `assets/harbor_registry.json` pins every task to commit
`2494eda6ae` of https://github.com/eldarkurtic/harbor-datasets (branch `swebenchpro/verifier-env-fixes`, the
content of PR #258), and `run_eval.sh` loads it with `inspect_harbor/harbor -T registry_path=... -T
dataset_name_version=swebenchpro@1.0`; Harbor fetches the task files from GitHub at that commit. It needs
`inspect-harbor>=1.0` for the verifier fix. The results below were produced with inspect_harbor 0.7.5 on a local copy
of the hub package with the same verifier changes plus a `PATH` restore that 0.7.5 needed; the pinned setup was
verified afterwards with `--solver inspect_harbor/oracle` on ansible, flipt-io and qutebrowser tasks. Once PR #258 is
merged and the hub package republished, switch back to `inspect_harbor/scale_ai_swe_bench_pro -T ref=<digest>` and
delete the registry file. Host-side measures that are not part of the task (pre-built images to stay under Docker
Hub's pull limit, UID rewriting for rootless Docker) are described in `local_env_notes.md`.
