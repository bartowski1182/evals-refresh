# Qwen/Qwen3.8-27B — GPQA-Diamond

| | |
|---|---|
| Task | `inspect_evals/gpqa_diamond` (198 questions, 4 choices) |
| Category | knowledge-and-reasoning (subclass: science-reasoning) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |

`cot=True` (the package default): every question is asked with the package's chain-of-thought multiple-choice
prompt (`ANSWER: $LETTER`), answer choices shuffled with the package's fixed seed (`shuffle_choices=42`). The
dataset is the simple-evals `gpqa_diamond.csv`, verified by SHA256 inside `inspect_evals` (no HF revision).
The package default is 4 epochs; this run uses the repo convention of 3 (avg@3).

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, max tokens, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
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
if set). Export `HF_TOKEN` in your shell before running (never in the repo): the dataset `Idavidrein/gpqa` is gated,
accept its terms at https://huggingface.co/datasets/Idavidrein/gpqa with the account the token belongs to.
Without a token, anonymous Hub downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 5`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=131072, max_connections=128, client_timeout=7200, epochs=3 (mean)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (avg@3) | 0.9057 |
| stderr | 0.0193 |
| per-epoch accuracy | 0.899 / 0.909 / 0.909 |
| samples | 594 (198 x 3 epochs) |
| truncated (stop_reason=max_tokens) | 0 |
| unparsed answers | 1 (model stopped after 53k reasoning tokens with an empty final answer) |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 316 | 13,255 | 12,947 |
| median | 282 | 7,560 | 7,340 |
| max | 2,844 | 98,187 | 97,907 |

Total run time about 25 min on 8x H100 (128 concurrent samples).

## Patches for upstream

None.

