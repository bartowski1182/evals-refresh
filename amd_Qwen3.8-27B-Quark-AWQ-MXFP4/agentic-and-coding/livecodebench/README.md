# amd/Qwen3.8-27B-Quark-AWQ-MXFP4 — LiveCodeBench (Harbor, 100-task sample)

| | |
|---|---|
| Task | `inspect_harbor/livecodebench` (100 tasks sampled from LiveCodeBench release_v6, loaded through `assets/harbor_registry.json`, see Patches) |
| Category | agentic-and-coding (subclass: code-gen) |
| Exec tier | `agent` — Docker container per task, bash/python tools, hidden-test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (base model; the recipe has no MXFP4 variant) |
| Model | https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (revision `5233554c5f`): W4A4 MXFP4 (AMD Quark, AWQ) |
| Task source | https://hub.harborframework.com/datasets/livecodebench/livecodebench |

**Protocol note.** The task, agent and generation settings are those of
`Qwen_Qwen3.8-27B/agentic-and-coding/livecodebench`: a fixed 100-problem sample solved agentically with Inspect's
ReAct agent (the model may run public tests and iterate, and hidden tests grade `/app/solution.py`), per-turn
max_tokens 65536, a 3600 s time limit per sample, sampling from the checkpoint's `generation_config.json` (identical to
the base model's), reasoning effort xhigh, avg@3. The KV cache is bf16, as for the base model and
`RedHatAI_Qwen3.8-27B-MXFP4` (served with the same config). Two things differ from the base run:
- `-T override_cpus=0`: the tasks' 1-CPU cap is dropped, because rootless Docker on this host cannot apply CPU
  limits. Memory caps are kept.
- The serving stack: 4x AMD MI355X (TP=1, DP=4), vLLM 0.31.0 ROCm build, AITER W4A4 MXFP4 GEMM, Triton attention.
  The base ran on 8x H100 (TP=2, DP=4) with vLLM 0.29.0.

The model card's "LiveCodeBench v6" is single-shot pass@1 over the full release_v6 set, so this number is not
comparable to it.

## Files

- `assets/harbor_registry.json` — Harbor registry entry for `livecodebench@6.0` pinned to a commit that
  carries our upstream fixes (see Patches); `run_eval.sh` loads the dataset from it.
- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here: parallelism, context length and
  reasoning/tool parsers from the base model's recipe, plus the attention backend and load strategy.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task, epochs, sampling
  params, reasoning effort, per-turn max tokens, concurrency, sandbox count, time limit, CPU override).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM container and of the `inspect-harbor`
  venv that produced the results below, with the image digest and host in the headers.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM ROCm
  container (rootless podman), points Inspect's sandboxes at rootless Docker, waits for the server, activates the
  Inspect venv, runs `run_eval.sh`, tears down. **This is the only file you should modify** to match the environment
  you run in.

## Run

### Setup (once per host)

The server runs from the official vLLM ROCm image, so it needs no venv. `local_orchestrator.sh` expects:
- rootless podman and `numactl` for the server, with `/dev/kfd` and `/dev/dri` accessible to your user;
- a Docker daemon for the sandboxes. On the reference host this is rootless Docker at
  `unix:///run/user/<uid>/docker.sock`; drop the `DOCKER_HOST` export for a rootful daemon.

Create the client environment as below. The commands use the path that `local_orchestrator.sh` activates; if you keep
it elsewhere, change the path there. `client_requirements.txt` is the exact `pip freeze` that produced the results
below.

```bash
# server: vLLM 0.31.0 ROCm image (digest in server_requirements.txt)
podman pull docker.io/vllm/vllm-openai-rocm:v0.31.0
# client: Inspect + inspect-harbor
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install --no-deps -r client_requirements.txt   # or:
# pip install "inspect-harbor==0.7.5" httpx2 "openai==3.14.0"
```

`inspect-harbor` pulls in `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible providers need
`openai>=3`. The freeze therefore contains `openai==3.14.0`, and pip refuses to resolve it, hence `--no-deps`. The
resulting `pip check` warning is harmless because Inspect's agent path never imports `litellm`. `httpx2` is imported by
`inspect-ai` but not declared.

Model weights are fetched from the Hugging Face Hub on first use into `~/.cache/huggingface` (or `HF_HOME`), which the
orchestrator mounts into the container. Nothing here is gated; export `HF_TOKEN` (never in the repo) to avoid the Hub's
anonymous rate limits.

### Run the eval

```bash
./local_orchestrator.sh                         # all 8 GPUs (TP=1, DP=8)
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`. The results below come from
two runs with the same config: `GPUS=4,5,6,7 DP=4 PORT=8001 EPOCHS=1 ./local_orchestrator.sh` (epoch 1), then the same with
`EPOCHS=2` (epochs 2-3), each with the RedHatAI model's run on GPUs 0-3 at the same time.

On the first run Harbor downloads the 100 tasks into `~/.cache/harbor/tasks/packages/livecodebench/`, then Inspect builds
one image per task, one after another, before any sample starts. With the Dockerfile patch described under Patches, the
apt layer is shared, so this takes a few minutes and ~2 GB. During image builds `inspect ctl` reports no running eval;
watch the eval output instead. Once samples start, 100 containers run at once.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=100, max_sandboxes=100, time_limit=3600s, client_timeout=7200, override_cpus=0, epochs=3 (mean; runs of 1 + 2 epochs); kv_cache_dtype=auto (bf16), attention_backend=TRITON_ATTN, vLLM 0.31.0 (ROCm), TP=1 DP=4 on 4x MI355X`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (avg@3, all 300 samples; sample errors count as failures) | 0.870 |
| stderr | 0.028 |
| per-epoch pass rate | 0.850 / 0.870 / 0.890 |
| samples | 300 (100 tasks x 3 epochs, in 2 runs: 100 + 200) |
| tasks passing 3/3, 2/3, 1/3, 0/3 | 77 / 13 / 4 / 6 |
| time-limit hits (3600 s) | 15 (none of them passed the verifier) |
| no solution file (scored 0) | 26: 15 time-limit hits; 11 samples passed the code to `submit()` instead of writing `/app/solution.py` |
| sample errors | 0 |
| turns truncated at max_tokens | 15 |
| context compactions | 4 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 363,777 | 26,459 | 22,212 |
| median | 106,131 | 9,584 | 6,407 |
| max | 6,580,755 | 215,413 | 210,726 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 9, mean 12, max 53. Wall time per sample: median 160 s, mean 523 s, max 3,610 s.
Total run time 1 h 0 min (epoch 1) + 1 h 5 min (epochs 2-3) on 4x MI355X (100 concurrent samples).

Always-failing tasks (0/3): abc363_f, abc397_d, arc182_e, arc184_d, arc192_b, arc196_c.

## Patches for upstream

Two fixes to the Harbor LiveCodeBench dataset are proposed upstream in
[harbor-framework/harbor-datasets](https://github.com/harbor-framework/harbor-datasets):

- [#257](https://github.com/harbor-framework/harbor-datasets/pull/257) — `tests/test.sh` writes
  `/logs/verifier/reward.txt` (reward 0) before its early exit when the agent never created
  `/app/solution.py`. Without it the sample becomes an *error* (excluded from accuracy) and
  Inspect's default fail-on-error policy aborts the eval.
- [#256](https://github.com/harbor-framework/harbor-datasets/pull/256) — `environment/Dockerfile`
  copies the per-task files *after* `apt-get`, so the 100 images share the apt layer (~1 min and
  ~2 GB to build) instead of each storing ~750 MB (~90 min and ~75 GB).

**Until they are merged**, `run_eval.sh` does not use the hub package. It loads the dataset through
`inspect_harbor/harbor` from `assets/harbor_registry.json`: the upstream registry entry for
`livecodebench@6.0` with every task pinned to commit `f7852012df9d20a0a0fe9af36e1f632a6094975c` of
[eldarkurtic/harbor-datasets, branch `livecodebench-eldar-fixes`](https://github.com/eldarkurtic/harbor-datasets/tree/livecodebench-eldar-fixes),
which is upstream `main` plus both PRs. Harbor fetches the task files from GitHub at that commit, so
nothing has to be patched by hand.

Once both PRs are merged and the hub package is republished, replace the two `-T` lines in
`run_eval.sh` with `inspect eval inspect_harbor/livecodebench` and delete `assets/harbor_registry.json`.
