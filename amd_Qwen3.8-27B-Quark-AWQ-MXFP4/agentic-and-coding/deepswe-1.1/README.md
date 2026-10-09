# amd/Qwen3.8-27B-Quark-AWQ-MXFP4 — DeepSWE 1.1

| | |
|---|---|
| Task | `inspect_harbor/datacurve_deep_swe_1_1` (113 tasks, Harbor hub package digest `sha256:5affcd53…`, loaded through `assets/harbor_registry.json`, which pins the tasks to the fork commit carrying the verifier fix, see Patches) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per task (public.ecr.aws, 2 CPUs / 8 GB requested), bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (base model; the recipe has no MXFP4 variant) |
| Model | https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (revision `5233554c5f`): W4A4 MXFP4 (AMD Quark, AWQ) |
| Task source | https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1 |

**Protocol note.** The task, agent and generation settings are those of `Qwen_Qwen3.8-27B/agentic-and-coding/deepswe-1.1`:
- Inspect's generic ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context
  compaction; no message limit), not the Claude Code harness of the model card.
- A single `--time-limit 14400` instead of Harbor's per-task agent budget.
- Containers without Harbor's CPU cap (`override_cpus=0`) and with `inspect_harbor`'s 6 GB memory floor.
- One epoch (pass@1); per-turn max_tokens 65536; sampling from the checkpoint's `generation_config.json` (identical to
  the base model's); reasoning effort xhigh.

The KV cache is bf16, as for the base model and `RedHatAI_Qwen3.8-27B-MXFP4` (served with the same config). What
differs from the base run:
- The serving stack: AMD MI355X, vLLM 0.31.0 ROCm build, AITER W4A4 MXFP4 GEMM, Triton attention, `inspect-harbor`
  1.0.0. The base ran on 8x H100 with vLLM 0.29.0 and `inspect-harbor` 0.7.6.
- The base kept every turn of a sample on one prefix cache by serving from a single TP=8 replica, which does not start
  here (see `local_orchestrator.sh`). Instead, the 113 tasks are split into 4 shards (tasks sorted by name; shard k
  takes every 4th task from k, the same split for both models). Each shard is served by its own single-GPU vLLM server
  (TP=1, DP=1) on GPUs 4-7, with 4-7 samples in flight per shard.

The numbers are therefore not directly comparable to the model card (42.2 for the base model with Claude Code), and
only roughly to the base run.

## Files

- `assets/harbor_registry.json` — Harbor registry entry pinning the 113 tasks to the fork commit that carries the verifier fix (see Patches).
- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here: parallelism, context length and
  reasoning/tool parsers from the base model's recipe, plus the attention backend and load strategy.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task and registry pin,
  epochs, sampling params, per-turn max tokens, concurrency, sandbox count, time limit, CPU override, error threshold).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM container and of the
  `inspect-harbor` 1.0 venv that produced the results below, with the image digest and host in the headers.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM ROCm
  container (rootless podman), points Inspect's sandboxes at rootless Docker, waits for the server, activates the
  Inspect venv, runs `run_eval.sh`, tears down. `LOG_DIR` puts the server log and the eval log side by side, so shards
  don't collide. **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

The server runs from the official vLLM ROCm image, so it needs no venv. `local_orchestrator.sh` expects:
- rootless podman and `numactl` for the server, with `/dev/kfd` and `/dev/dri` accessible to your user;
- Docker with the compose plugin for the sandboxes. The orchestrator exports `DOCKER_HOST` for a rootless daemon
  (`unix:///run/user/<uid>/docker.sock`, image store under `~/.local/share/docker`); with a rootful daemon, drop that
  export.

Rootless Docker needs a runc runtime with `NoNewKeyring` in `daemon.json`; otherwise each container takes one kernel key
and exhausts a user's quota of 200. It cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`.

Create the client environment as below. The commands use the path that `local_orchestrator.sh` activates; if you keep
it elsewhere, change the path there. `client_requirements.txt` is the exact `pip freeze` that produced the results
below.

```bash
# server: vLLM 0.31.0 ROCm image (digest in server_requirements.txt)
podman pull docker.io/vllm/vllm-openai-rocm:v0.31.0
# client: Inspect + inspect-harbor 1.0 (no harbor/litellm dependency, so no openai pin conflict)
python3.12 -m venv ~/.venvs/inspect-harbor-1.0 && source ~/.venvs/inspect-harbor-1.0/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "inspect-harbor==1.0.0" "inspect_ai==0.3.264" httpx2 "openai==3.14.0"
```

Model weights are fetched from the Hugging Face Hub on first use into `~/.cache/huggingface` (or `HF_HOME`), which the
orchestrator mounts into the container. Nothing here is gated; export `HF_TOKEN` (never in the repo) to avoid the Hub's
anonymous rate limits. The 113 task images (about 122 GB, sharing most layers) come from public.ecr.aws, which throttles
bursts: pre-pull them one by one with `docker pull` before the first run.

### Run the eval

Single server (all 8 GPUs behind one endpoint), as the scripts default to:

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

The results below come from 4 single-GPU shards, run on GPUs 4-7 next to the RedHatAI model's 4 shards on GPUs 0-3:

```bash
for k in 0 1 2 3; do
  names=$(python3 -c "import json; e=json.load(open('assets/harbor_registry.json')); e=e[0] if isinstance(e,list) else e; print(json.dumps(['*'+t for t in sorted(x['name'] for x in e['tasks'])[$k::4]]))")
  GPUS=$((4+k)) DP=1 PORT=$((8004+k)) LOG_DIR=./logs/shard$k MAX_CONNECTIONS=7 MAX_SANDBOXES=7 FAIL_ON_ERROR=0.25 \
    ./local_orchestrator.sh -T "dataset_task_names=$names" &
done; wait
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`. An interrupted shard is resumed with
`inspect eval-retry logs/shard<k>/<file>.eval` against a server started the same way; it keeps the finished samples and
reruns the unfinished and errored ones.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), no message limit, time_limit=14400s, client_timeout=7200, tool_timeout=300s, override_cpus=0, epochs=1; kv_cache_dtype=auto (bf16), attention_backend=TRITON_ATTN, vLLM 0.31.0 (ROCm); 4 shards x 1 MI355X (TP=1, DP=1), 7 samples in flight per shard, fail_on_error=0.25 per shard; the run was interrupted by a host crash after 96 of 113 samples and resumed with inspect eval-retry at 4 samples in flight per shard, which reran the 17 unfinished and 10 errored samples`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch, all 113 tasks; sample errors count as unresolved) | 0.221 (25 of 113) |
| stderr (binomial) | 0.0392 |
| mean partial score (fraction of required tests passing, grader's `partial`; sample errors count as 0) | 0.852 |
| samples | 113 of 113 tasks, 1 epoch |
| time-limit hits (14400 s) | 2 (anko-default-function-arguments, numba-stencil-boundary-modes) |
| samples whose agent never committed (no `model.patch`, graded as the base state) | 2 (the two time-limit hits) |
| turns truncated at max_tokens | 0 |
| context compactions | 179 |
| sample errors (scored as unresolved; all `Compaction insufficient`) | 9 (dynamodb-toolbox-lazy-recursive-schemas, opa-template-string-reconstruction, oxvg-structural-selector-preservation, scriggo-method-declarations, sqlfmt-create-table-ddl-formatting, tomlkit-toml-table-converters, valibot-recursive-schema-composition, wasmi-trap-coredumps, yaegi-go-embed-directives) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 15,773,635 | 147,318 | 101,971 |
| median | 13,868,689 | 123,653 | 82,096 |
| max | 47,109,188 | 699,021 | 599,878 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 132, mean
202, max 2,783. Wall time per sample: median 3,259 s, mean 3,732 s, max 14,410 s. The agent ends when the model replies
without a tool call; the verifier then collects `git diff <base> HEAD` as `model.patch` and grades it, so only committed
work counts. Total run time 4 h 12 min (until the crash) + 4 h 0 min (resume) on 4x MI355X.

Per language (resolved / tasks): typescript 10/35, go 8/34, python 6/34, javascript 1/5, rust 0/5.

All 9 sample errors are Inspect's `RuntimeError: Compaction insufficient`. After hundreds of agent turns, the agent's
own messages exceed the compaction threshold of 90% of the 262k context, and Inspect stops the sample with an error
instead of grading the committed state. They are scored as unresolved, so that every model is scored on the
same 113 tasks (excluding them would also flatter the score: these are the longest, unfinished trajectories).

## Patches for upstream

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
dataset_name_version=deep-swe-1-1@1.1`. The results below were produced with exactly this setup and `inspect-harbor`
1.0.0, which carries the other two fixes. It was checked with `--solver inspect_harbor/oracle` on
abs-module-cache-flags (Go), dynamodb-toolbox-lazy-recursive-schemas (the vitest fix) and igel-persist-feature-schema
(the `PATH` fix): 3/3 resolved, `model.patch` collected and applied. Once PR #100 is merged and the hub package
republished, switch back to `inspect_harbor/datacurve_deep_swe_1_1 -T ref=<digest>` and delete the registry file.
