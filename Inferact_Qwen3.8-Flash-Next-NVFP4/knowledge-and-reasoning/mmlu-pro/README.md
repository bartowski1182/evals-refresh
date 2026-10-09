# Inferact/Qwen3.8-Flash-Next-NVFP4 — MMLU-Pro

| | |
|---|---|
| Task | `inspect_evals/mmlu_pro` (12,032 questions, 10 choices, 14 subjects, 0-shot) |
| Category | knowledge-and-reasoning (subclass: knowledge-recall) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |

`fewshot=0` (the package default): every question is asked zero-shot with the package's chain-of-thought
multiple-choice prompt (`ANSWER: $LETTER`). Dataset pinned at revision `527feea0af`. One epoch, as for
Qwen/Qwen3.8-27B in this repo. Neither the Qwen/Qwen3.8-Flash-Next card nor the Inferact card reports MMLU-Pro, so
there is no reference number. Checkpoint: NVFP4 (W4A4) on the MoE experts, ModelOpt format; the 51B n-gram
embedding table stays BF16. Revision `103a760`. This is the recipe's default variant.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtypes, reasoning/tool parsers, text-only mode), taken from the official recipe (Blackwell, features
  tool_calling, reasoning, text_only). TP=1 × DP=4 on four GPUs: vLLM keeps the n-gram table in pinned host memory
  and shards the MoE experts across the DP group.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, max tokens, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs
  (4-7, port 8001, so both NVFP4 checkpoints can run side by side on one 8-GPU host) and the local
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
`Inferact/Qwen3.8-Flash-Next-NVFP4` from the Hugging Face Hub. Loading copies the 51B n-gram table into pinned host
memory (about 100 GB per server), which takes several minutes. Export `HF_TOKEN` in your shell before running
(never in the repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

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
| accuracy | 0.880 |
| stderr | 0.0030 |
| samples | 12,032 (1 epoch) |
| truncated (stop_reason=max_tokens) | 0 |
| unparsed answers | 1 (the model stopped after its reasoning with an empty final answer) |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 308 | 4,621 | 4,448 |
| median | 275 | 724 | 534 |
| max | 1,767 | 113,572 | 113,244 |

Per subject: math 0.952, biology 0.941, physics 0.924, business 0.918, chemistry 0.914, economics 0.895, computer science 0.885, psychology 0.875, engineering 0.865, philosophy 0.862, health 0.828, other 0.821, history 0.811, law 0.764.

Total run time about 55 min on 4x B200 (TP=1 × DP=4), sharing the server with the other two knowledge-and-reasoning tasks of this model, which ran at the same time.

## Patches for upstream

None.
