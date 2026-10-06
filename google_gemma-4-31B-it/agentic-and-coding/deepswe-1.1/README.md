# google/gemma-4-31B-it — DeepSWE 1.1

| | |
|---|---|
| Task | `inspect_harbor/datacurve_deep_swe_1_1` (113 tasks, Harbor hub package digest `sha256:5affcd53…`, loaded through `assets/harbor_registry.json`, which pins the tasks to the fork commit carrying the verifier fix, see Patches) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per task (public.ecr.aws, 2 CPUs / 8 GB requested), bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/Google/gemma-4-31B-it |
| Task source | https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1 |

**Protocol note.** The model card does not report DeepSWE. `inspect_harbor` solves every task with Inspect's generic ReAct agent
(`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction; no message limit),
not the Claude Code harness. Harbor's per-task agent budget is not applied by `inspect_harbor`; a single
`--time-limit 14400` wall-clock guard is used. Containers run without Harbor's CPU cap (`override_cpus=0`) and
with `inspect_harbor`'s 6 GB memory floor. One epoch (pass@1) instead of the repo's usual avg@3. All in-flight
samples share one vLLM server (one TP=8 replica so the prefix cache stays local; 56 samples in flight, what its
4M-token KV cache holds once histories average 45k tokens). The numbers are therefore not directly comparable to
the model cards.

## Files

- `assets/harbor_registry.json` — Harbor registry entry pinning the 113 tasks to the fork commit that carries the verifier fix (see Patches).
- `assets/generate_config.json` — Inspect generate config that cannot be given as a CLI flag; here it
  carries the per-request `chat_template_kwargs` that enable thinking.
- `assets/tool_chat_template_gemma4.jinja` — vLLM's Gemma 4 chat template (from the recipe), used by `serve.sh`.
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

`local_orchestrator.sh` activates two virtualenvs, `~/.venvs/vllm` for the server and `~/.venvs/inspect-harbor` for the
Inspect client; create them as below (edit the paths in `local_orchestrator.sh` if you keep them elsewhere).
`server_requirements.txt` and `client_requirements.txt` are the exact `pip freeze` of the environments that
produced the results below (Python 3.12, host details in their headers): installing from them reproduces those
environments; the commented alternative gives an equivalent fresh install with the same pinned versions.

```bash
# server: vLLM 0.29.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.29.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install -r client_requirements.txt          # what produced the results (inspect-harbor 0.7.6); or the current setup:
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
`~/.cache/harbor/tasks/packages/`, then Inspect pulls one prebuilt image per task from public.ecr.aws as containers start (113 images, 3 to 4 GB each on disk
but sharing most layers; ECR rate-limits bursts, so pre-pull serially with `docker pull`).
Later runs reuse the images.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=64, chat_template_kwargs={enable_thinking: true}, max_tokens=65536 (per turn), no message limit, time_limit=14400s, client_timeout=7200, tool_timeout=300s, TP=8 DP=1, epochs=1; 113 tasks in one run at 56 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.018 (2 of 110 scored tasks) |
| stderr (binomial) | 0.0127 |
| mean partial score (fraction of required tests passing, grader's `partial`) | 0.539 |
| samples | 113 of 113 tasks, 1 epoch |
| time-limit hits (14400 s) | 0 |
| samples whose agent never committed (no `model.patch`, graded as the base state) | 19 |
| turns truncated at max_tokens | 0 |
| context compactions | 0 |
| sample errors (excluded from the score) | 3 (dynamodb-toolbox-lazy-recursive-schemas (OSError), httpx-multipart-response-parsing (SandboxTimeoutError), pwntools-tube-multiplexing (SandboxTimeoutError)) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 2,174,117 | 25,772 | 8,281 |
| median | 1,357,848 | 23,558 | 7,538 |
| max | 19,555,603 | 86,194 | 19,483 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 41, mean 55, max 203. Wall time per sample: median 1,317 s,
mean 1,412 s, max 3,460 s. The agent ends when the model replies without a tool call; the verifier
then collects `git diff <base> HEAD` as `model.patch` and grades it, so only committed work counts. Total run time
1 h 09 min on 8x H100.

Per language (resolved / tasks): go 0/34, typescript 1/34, python 0/32, rust 1/5, javascript 0/5.

The 2 verifier timeouts are Harbor's 1800 s verifier budget (`verifier.timeout_sec`) exceeded by the test
suite; the `OSError` is a tool call whose command exceeded the kernel argument-length limit. No sample emitted a
pseudo tool call as plain text.

## Patches for upstream

**Known issue (results affected).** Every DeepSWE task denies network to its agent and verifier, but the
inspect_harbor version used here ignored those settings, so agents had internet access. Fixed in
[meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200). Until it is
released, install inspect_harbor from `eldarkurtic/inspect_harbor@fix/enforce-agent-network-mode`, as the
`*_Qwen3.8-Flash-Next*` DeepSWE dirs do, and rerun before quoting these numbers.

- `inspect_harbor` up to 0.7.5 ignored the `[[verifier.collect]]` hook every DeepSWE task declares (the
  `git diff --binary <base_commit> HEAD > /logs/artifacts/model.patch` that the grader applies in a `separate`
  verifier environment) and graded the pristine base state. Fixed upstream in
  https://github.com/meridianlabs-ai/inspect_harbor/pull/155 (0.7.6): the hook runs in the agent container, the
  repository is reset to the base commit and `tests/test.sh` runs there (the verifier image is the task image plus
  the test files, so the approximation only differs in untracked state, which the reset removes). Only committed
  work is graded, as the task instructions state.
- vitest sizes its worker pool from the visible CPU count (384 here; Harbor's `cpus = 2` is a quota and does not
  change the count), the workers exhaust the task's 8 GB memory cap and the run is OOM-killed, so every test is
  "missing from report" on six tasks (dynamodb-toolbox x2, effect, meriyah, superjson, true-myth). Fixed by making
  the shared verifier frame re-execute itself under `taskset` on a random 4-core slice; upstream:
  https://github.com/datacurve-ai/deep-swe/pull/100.
- `inspect_harbor` 0.7.x ran the verifier as `bash -l /tests/test.sh`; `/etc/profile` in these images drops the
  project virtualenv from `PATH`, so `pytest` is "missing" (igel-persist-feature-schema). Fixed upstream in
  inspect_harbor 1.0.0 (https://github.com/meridianlabs-ai/inspect_harbor/pull/174).

This directory consumes the dataset fix from the fork: `assets/harbor_registry.json` pins the 113 tasks of the hub
package to commit `89080fcd34` of https://github.com/eldarkurtic/deep-swe (branch `fix/verifier-cpu-affinity`, the
content of PR #100), and `run_eval.sh` loads it with `inspect_harbor/harbor -T registry_path=... -T
dataset_name_version=deep-swe-1-1@1.1`. It needs `inspect-harbor>=1.0` for the `PATH` fix. The results below were
produced with inspect_harbor 0.7.6 on a local copy of the hub package with the same verifier change plus a `PATH`
restore that 0.7.6 needed; the pinned setup was verified afterwards with `--solver inspect_harbor/oracle` on the
igel and dynamodb-toolbox tasks. Once PR #100 is merged and the hub package republished, switch back to
`inspect_harbor/datacurve_deep_swe_1_1 -T ref=<digest>` and delete the registry file.
