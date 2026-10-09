# Qwen/Qwen3.8-Flash-Next — MMLU-Pro

| | |
|---|---|
| Task | `inspect_evals/mmlu_pro` (12,032 questions, 10 choices, 14 subjects, 0-shot) |
| Category | knowledge-and-reasoning (subclass: knowledge-recall) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |

`fewshot=0` (the package default): every question is asked zero-shot with the package's chain-of-thought
multiple-choice prompt (`ANSWER: $LETTER`). Dataset pinned at revision `527feea0af`. One epoch, as for
Qwen/Qwen3.8-27B in this repo. The Qwen/Qwen3.8-Flash-Next card does not report MMLU-Pro, so there is no reference
number. Checkpoint: the original BF16 weights, revision `de4b8e4`.

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

`local_orchestrator.sh` activates two virtualenvs, `~/.venvs/vllm-nightly` for the server and `~/.venvs/inspect`
for the Inspect client; create them as below (edit the paths in `local_orchestrator.sh` if you keep them
elsewhere). `server_requirements.txt` and `client_requirements.txt` are the exact `pip freeze` of the environments
that produced the results below (Python 3.12, host details in their headers).

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
the repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 5`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=131072, max_connections=1024, client_timeout=7200, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy | 0.885 |
| stderr | 0.0029 |
| samples | 12,032 (1 epoch) |
| truncated (stop_reason=max_tokens) | 1 |
| unparsed answers | 1 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 308 | 4,381 | 4,209 |
| median | 275 | 702 | 507 |
| max | 1,767 | 131,072 | 131,072 |

Per subject: math 0.947, biology 0.932, physics 0.931, chemistry 0.924, business 0.919, computer science 0.905, economics 0.904, engineering 0.873, psychology 0.871, philosophy 0.860, other 0.833, health 0.828, history 0.803, law 0.786.

Total run time about 109 min on 4x B200 (TP=4 × DP=1), sharing the server with the other two knowledge-and-reasoning tasks of this model, which ran at the same time.

## Patches for upstream

None.
