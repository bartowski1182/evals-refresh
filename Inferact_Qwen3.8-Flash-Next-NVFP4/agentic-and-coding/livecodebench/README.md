# Inferact/Qwen3.8-Flash-Next-NVFP4 — LiveCodeBench (Harbor, 100-task sample)

| | |
|---|---|
| Task | `inspect_harbor/livecodebench` (100 tasks sampled from LiveCodeBench release_v6) |
| Category | agentic-and-coding (subclass: code-gen) |
| Exec tier | `agent` — Docker container per task, bash/python tools, hidden-test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |
| Task source | https://hub.harborframework.com/datasets/livecodebench/livecodebench |

**Protocol note.** The model card's "LiveCodeBench v6" (91.9 for Qwen/Qwen3.8-Flash-Next) is single-shot pass@1
over the full release_v6 set (1,055 problems). This task is a fixed 100-problem sample solved agentically (the
model can run public tests and iterate), so the two numbers are not comparable. One epoch (pass@1) instead of the
avg@3 of the other LiveCodeBench dirs of this repo, by request. Checkpoint: NVFP4 (W4A4) MoE experts, ModelOpt
format, revision `103a760` (the recipe's default variant).

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

`local_orchestrator.sh` activates `~/.venvs/vllm-nightly` for the server and `~/.venvs/inspect-harbor` for the
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
uv venv ~/.venvs/inspect-harbor --python 3.12 && source ~/.venvs/inspect-harbor/bin/activate
uv pip install -r client_requirements.txt   # or: uv pip install "inspect-harbor==1.0.0" httpx2 && uv pip install "openai==3.14.0"
```

Docker with the compose plugin (v2.x or later) and buildx >= 0.17 is required. The results below were produced with
a rootless Docker 29 daemon (containerd image store on local NVMe, 512 network pools, runc with `NoNewKeyring`);
`local_orchestrator.sh` points `DOCKER_HOST` at its socket. Inspect builds a task's image before its first sample
if it is missing (serially, before any sample starts). If you pre-built the `hb__<hash>` images, pass
`--sandbox-prebuilt` to skip that step.

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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=100, max_sandboxes=100, time_limit=3600s, client_timeout=7200, TP=4 DP=1, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (pass@1, 1 epoch) | 0.940 |
| stderr | 0.024 |
| samples | 100 (100 tasks x 1 epoch) |
| time-limit hits (3600 s) | 0 |
| no solution submitted (scored 0) | 1 |
| turns truncated at max_tokens | 0 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 166,143 | 10,211 | 7,206 |
| median | 61,018 | 4,594 | 2,334 |
| max | 2,212,675 | 78,633 | 67,003 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 8, mean
10, max 38. Wall time per sample: median 121 s, mean 186 s, max 1,074 s. Total run time about 18 min on 4x B200
(one TP=4 replica, 100 concurrent samples), with the other model's runs on the other four GPUs of the host at the
same time.

Failed tasks: 3308, 3653, abc397_d, arc188_c, arc192_b, arc195_c.

## Patches for upstream

Two fixes to the Harbor LiveCodeBench dataset are proposed upstream in
[harbor-framework/harbor-datasets](https://github.com/harbor-framework/harbor-datasets):
[#257](https://github.com/harbor-framework/harbor-datasets/pull/257) makes `tests/test.sh` write
`/logs/verifier/reward.txt` (reward 0) when the agent never created `/app/solution.py` (otherwise the sample
becomes an error and aborts the eval), and [#256](https://github.com/harbor-framework/harbor-datasets/pull/256)
moves the per-task `COPY` after `apt-get` in `environment/Dockerfile` so the 100 images share their apt layer.
Until they are merged, `run_eval.sh` loads the dataset through `inspect_harbor/harbor` from
`assets/harbor_registry.json`: the upstream `livecodebench@6.0` entry with every task pinned to commit `f7852012df`
of [eldarkurtic/harbor-datasets, branch
`livecodebench-eldar-fixes`](https://github.com/eldarkurtic/harbor-datasets/tree/livecodebench-eldar-fixes)
(upstream `main` plus both PRs).
