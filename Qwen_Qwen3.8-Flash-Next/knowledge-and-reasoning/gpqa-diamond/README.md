# Qwen/Qwen3.8-Flash-Next — GPQA-Diamond

| | |
|---|---|
| Task | `inspect_evals/gpqa_diamond` (198 questions, 4 choices) |
| Category | knowledge-and-reasoning (subclass: science-reasoning) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |

`cot=True` (the package default): every question is asked with the package's chain-of-thought multiple-choice
prompt (`ANSWER: $LETTER`), answer choices shuffled with the package's fixed seed. One epoch (a single sample per
question) instead of the repo's usual avg@3, by request. The Qwen/Qwen3.8-Flash-Next card reports 91.7 for this
model without stating its epochs, so a one-epoch score is comparable in protocol but carries a standard error of
about 2 points. Checkpoint: the original BF16 weights, revision `de4b8e4`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtypes, reasoning/tool parsers, text-only mode), taken from the official recipe (Blackwell, features
  tool_calling, reasoning, text_only). TP=4 × DP=1 on four GPUs: the recipe gives at least 423 GB of VRAM for the BF16
  weights, so one replica spans the four GPUs.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, max tokens, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs
  (4-7, port 8003, so two models can run side by side on one 8-GPU host) and the local
  checkpoint path, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below (Python 3.12,
host details in their headers).

```bash
# server: vLLM nightly 0.30.1rc1.dev418+g50319916a (the recipe requires a nightly build; CUDA 13 wheels)
uv venv ~/.venvs/vllm-nightly --python 3.12 && source ~/.venvs/vllm-nightly/bin/activate
uv pip install -r server_requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/50319916a3c72ec1358f9e29c7c282d316d86f78 \
  --extra-index-url https://download.pytorch.org/whl/cu130
# client: Inspect
uv venv ~/.venvs/inspect --python 3.12 && source ~/.venvs/inspect/bin/activate
uv pip install -r client_requirements.txt          # or: uv pip install "inspect-evals==0.20.0" openai
```

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`); unset it to let vLLM fetch
`Qwen/Qwen3.8-Flash-Next` from the Hugging Face Hub. Loading copies the 51B n-gram table into pinned host memory
(about 100 GB per server), which takes several minutes. Export `HF_TOKEN` in your shell before running (never in
the repo); the dataset `Idavidrein/gpqa` is gated, accept its terms at
https://huggingface.co/datasets/Idavidrein/gpqa with the account the token belongs to.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 5`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=131072, max_connections=256, client_timeout=7200, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy | 0.904 |
| stderr | 0.0210 |
| samples | 198 (1 epoch) |
| truncated (stop_reason=max_tokens) | 0 |
| unparsed answers | 0 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 316 | 15,177 | 14,854 |
| median | 282 | 7,008 | 6,580 |
| max | 2,844 | 124,392 | 124,066 |

Total run time about 56 min on 4x B200 (TP=4 × DP=1), sharing the server with the other two knowledge-and-reasoning tasks of this model, which ran at the same time.

## Patches for upstream

None.
