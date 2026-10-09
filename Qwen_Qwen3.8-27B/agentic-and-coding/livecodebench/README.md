# Qwen/Qwen3.8-27B — LiveCodeBench (Harbor, 100-task sample)

| | |
|---|---|
| Task | `inspect_harbor/livecodebench` (100 tasks sampled from LiveCodeBench release_v6) |
| Category | agentic-and-coding (subclass: code-gen) |
| Exec tier | `agent` — Docker container per task, bash/python tools, hidden-test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://hub.harborframework.com/datasets/livecodebench/livecodebench |

**Protocol note.** The model card's "LiveCodeBench v6" is single-shot pass@1 over the full
release_v6 set (1,055 problems). This task is a fixed 100-problem sample solved agentically
(the model can run public tests and iterate). The two numbers are not comparable.

## Files

- `assets/harbor_registry.json` — Harbor registry entry for `livecodebench@6.0` pinned to a commit that
  carries our upstream fixes (see Patches); `run_eval.sh` loads the dataset from it.
- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, per-turn max tokens, concurrency, sandbox count, time limit).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below. The client venv is the `inspect-harbor` one (see Run, Setup).
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

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
pip install -r client_requirements.txt          # or:
# pip install "inspect-harbor==0.7.5" httpx2 "openai==3.14.0"
```

`inspect-harbor` pulls in `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible providers need
`openai>=3`; installing `openai==3.14.0` last overrides the pin (pip prints a conflict warning that is harmless
here because Inspect's agent path never imports `litellm`). `httpx2` is imported by `inspect-ai` but not declared.

Model weights and datasets are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

Requires Docker. **What to expect on the first run:** Harbor downloads the 100 tasks into
`~/.cache/harbor/tasks/packages/livecodebench/` (a few seconds), then Inspect builds one image
per task, one after another, before any sample starts. As shipped, each build takes ~1 minute
and stores its own ~750 MB of layers, so setup is ~90 minutes and ~75 GB of disk. With the
Dockerfile patch described under Patches, the apt layer is shared: ~1 minute and ~2 GB total.
Later runs reuse the images and start within a minute. During image builds `inspect ctl` reports
no running eval; watch the eval output instead. Once samples start, 100 containers run at once.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=100, max_sandboxes=100, time_limit=3600s, client_timeout=7200, epochs=3 (mean)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (avg@3) | 0.927 |
| stderr | 0.023 |
| per-epoch pass rate | 0.94 / 0.90 / 0.94 |
| samples | 300 (100 tasks x 3 epochs) |
| tasks passing 3/3, 2/3 or 1/3, 0/3 | 89 / 7 / 4 |
| time-limit hits (3600 s) | 13 |
| no solution submitted (scored 0) | 13 |
| sample errors | 0 |
| turns truncated at max_tokens | 7 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 359,971 | 21,253 | 16,308 |
| median | 98,004 | 8,408 | 4,872 |
| max | 3,883,069 | 132,169 | 131,072 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 10, mean 12, max 56. Wall time per sample: median 229 s, mean 695 s, max 3614 s.
Total run time about 3.5 h on 8x H100 (100 concurrent samples).

Always-failing tasks (0/3): abc363_f, arc184_d, arc192_b, arc196_c.

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
