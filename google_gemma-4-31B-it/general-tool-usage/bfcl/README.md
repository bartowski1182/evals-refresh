# google/gemma-4-31B-it — BFCL

| | |
|---|---|
| Task | `inspect_evals/bfcl` (4981 samples: all V1, V2 and V3 categories of the Berkeley Function-Calling Leaderboard; gorilla repo commit pinned by inspect_evals 0.20.0) |
| Category | general-tool-usage (subclass: tool-use) |
| Exec tier | `agent` — no sandbox; native tool calling, multi-turn categories run stateful Python backends in-process |
| vLLM recipe | https://recipes.vllm.ai/Google/gemma-4-31B-it |
| Task source | https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard |

**Protocol note.** Neither model card reports BFCL. The official leaderboard's overall score (BFCL v4) also
covers the agentic V4 categories (web search, which needs a SerpAPI key, and memory, which needs a two-step
snapshot workflow) and the retired `rest` category; this run covers the V1 single-turn (AST-matched), V2 live
(user-contributed) and V3 multi-turn (stateful backends) categories only, so per-category accuracies are
comparable to the leaderboard's and the overall number is not. Tools are passed natively (FC mode) and
parsed by vLLM's tool parser; the model reasons before every call. avg@3 as usual in this repo.

## Files

- `assets/generate_config.json` — Inspect generate config that cannot be given as a CLI flag; here it
  carries the per-request `chat_template_kwargs` that enable thinking.
- `assets/tool_chat_template_gemma4.jinja` — vLLM's Gemma 4 chat template (from the recipe), used by `serve.sh`.
- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (epochs, sampling params, per-turn max tokens, time limit, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs and TP/DP, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
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
pip install -r client_requirements.txt          # what produced the results (inspect_evals 0.20.0); or the current setup, the
# scorer fix straight from the fork branch (see Patches):
# pip install "inspect-evals[bfcl] @ git+https://github.com/eldarkurtic/inspect_evals@fix/bfcl-multi-turn-scorer-short-trajectories" openai
```

Model weights and datasets are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 5`

No Docker needed. The first run clones the gorilla
repository at the pinned commit for the data and the multi-turn backends.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=64, chat_template_kwargs={enable_thinking: true}, max_tokens=65536 (per turn), time_limit=7200s, client_timeout=7200, TP=8 DP=1, epochs=3, 64 samples in flight; all categories in one run`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy over all 4981 samples (avg@3, sample-weighted) | 0.748 |
| stderr (binomial on 4981 samples) | 0.0062 |
| non-live AST (simple, multiple, parallel, parallel_multiple) | 0.851 (1150 samples) |
| non-live exec (AST-scored) | 0.881 (240 samples) |
| live AST | 0.807 (1351 samples) |
| relevance / irrelevance | 0.758 (1140 samples) |
| multi-turn | 0.557 (1000 samples) |
| sql | 0.217 (100 samples) |
| samples | 14943 (4981 x 3 epochs) |
| time-limit hits (7200 s) | 0 |
| turns truncated at max_tokens | 0 |
| sample errors (excluded from the score) | 0 |

Per category (accuracy, samples): simple_python 0.935 (400), simple_java 0.420 (100), simple_javascript 0.373 (50), multiple 0.928 (200), parallel 0.912 (200), parallel_multiple 0.880 (200), exec_simple 0.963 (100), exec_multiple 0.907 (50), exec_parallel 0.840 (50), exec_parallel_multiple 0.692 (40), live_simple 0.804 (258), live_multiple 0.809 (1053), live_parallel 0.854 (16), live_parallel_multiple 0.750 (24), irrelevance 0.799 (240), live_relevance 0.875 (16), live_irrelevance 0.745 (884), multi_turn_base 0.773 (200), multi_turn_miss_func 0.625 (200), multi_turn_miss_param 0.638 (200), multi_turn_long_context 0.712 (200), multi_turn_composite 0.038 (200), sql 0.217 (100).

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 20,521 | 868 | 670 |
| median | 495 | 253 | 189 |
| max | 4,311,668 | 65,536 | 24,280 |

Input tokens are cumulative over turns (multi-turn categories resend history). Model calls per sample: median 1, mean 3.6, max 132.
Wall time per sample: median 4 s, mean 28 s, max 2,337 s. Total run time 1 h 48 min on 8x H100.

## Patches for upstream

`inspect_evals/bfcl/score/multi_turn_scorer.py` raises `ValueError: zip() argument 2 is shorter than argument 1`
instead of scoring 0 when a multi-turn sample ends (time or message limit) with fewer turns than the ground truth
has, so such samples become sample errors excluded from accuracy. Reported as
https://github.com/UKGovernmentBEIS/inspect_evals/issues/2544; the fix (missing turns count as empty, task version
8-B) is branch `fix/bfcl-multi-turn-scorer-short-trajectories` of https://github.com/eldarkurtic/inspect_evals,
which the client venv installs from (see Setup). The results below were produced with inspect_evals 0.20.0; the
first pass hit the defect on 51 samples and the affected categories were rerun at a longer time limit, after which
no sample ends early, so the fix does not change the reported numbers.
