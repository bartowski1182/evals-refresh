# Qwen/Qwen3.8-27B — tau2-bench (native Inspect implementation of the tau3-bench task set)

| | |
|---|---|
| Task | `inspect_evals/tau2_airline`, `tau2_retail`, `tau2_telecom`, `tau2_banking` (375 tasks: the "base" splits of tau2-bench >= 1.0 plus banking_knowledge, the same task set as the Harbor `sierra-research/tau3-bench` package) |
| Category | general-tool-usage (subclass: tool-use) |
| Exec tier | `agent` — no sandbox; domain tools are in-process Python, a second model role simulates the customer |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://github.com/sierra-research/tau2-bench (data vendored by inspect_evals 0.20.0) |

**Protocol note.** Neither model card reports tau2/tau3-bench. This is inspect_evals's native re-implementation of
tau2-bench: the agent gets the domain policy as system prompt and the domain tools as Python tools, a user
simulator (Inspect model role `user`, temperature 0 as upstream; Qwen/Qwen3.8-27B for every model of this repo,
a fixed simulator as on the tau-bench leaderboards) plays the customer from the task's scenario, and a task
passes when the final database state matches the expected one and the required information was communicated
(the airline natural-language assertions of upstream are not graded by this implementation; banking uses the
`grep` knowledge-retrieval configuration rather than the leaderboard's `alltools`). The agent samples with the
model's own generation config instead of upstream's temperature 0 (see Patches). Up to 100 user/agent
exchanges per task. One epoch (pass^1) instead of the repo's usual avg@3. The same vLLM server also plays the user simulator (Inspect model role `user`). The numbers are therefore not
comparable to taubench.com; the Harbor-package run in `../tau3-bench` is the same task set through the
original tau2-bench runtime.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the four Inspect tasks against that server. Inspect-side params live here
  (tasks, user-simulator role, epochs, sampling params, per-turn max tokens, time limit, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs and TP/DP, starts the server(s) in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below (Python 3.12,
host details in their headers): installing from them reproduces those environments; the commented alternative gives an
equivalent fresh install with the same pinned versions.

```bash
# server: vLLM 0.29.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.29.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "inspect-evals==0.20.0" openai
```

Model weights and datasets are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

No Docker needed.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), time_limit=3600s, client_timeout=7200, up to 100 user/agent exchanges (task default), TP=8 DP=1, epochs=1; user simulator vllm/Qwen/Qwen3.8-27B at temperature 0 on the same server; 375 tasks in one run of the four domain tasks at 56 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| pass^1 over all domains (task-weighted mean, 375 tasks) | 0.699 (262 of 375) |
| stderr (binomial) | 0.0237 |
| pass^1 airline | 0.660 (33 of 50) |
| pass^1 retail | 0.816 (93 of 114) |
| pass^1 telecom | 0.904 (103 of 114) |
| pass^1 banking | 0.340 (33 of 97) |
| samples | 375 of 375 tasks, 1 epoch |
| sample errors (excluded from the score) | 0 |

| agent tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 626,879 | 6,040 | 4,038 |
| median | 134,008 | 3,880 | 2,763 |
| max | 48,616,781 | 72,116 | 30,780 |

Agent tokens exclude the user-simulator model's calls. Input tokens are cumulative over agent turns (history is
resent each turn). Agent model calls per sample: median 15, mean 20, max 361. Wall time per
sample: median 196 s, mean 347 s, max 3,600 s. Total run time 1 h 40 min on 8x H100.

## Patches for upstream

None.
