# nvidia/Qwen3.8-Flash-Next-NVFP4 — MMLU-Pro

| | |
|---|---|
| Task | `inspect_evals/mmlu_pro` (12,032 questions, 10 choices, 14 subjects, 0-shot) |
| Category | knowledge-and-reasoning (subclass: knowledge-recall) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |

`fewshot=0` (the package default): every question is asked zero-shot with the package's chain-of-thought
multiple-choice prompt (`ANSWER: $LETTER`). Dataset pinned at revision `527feea0af`. One epoch, as for
Qwen/Qwen3.8-27B in this repo. Neither the Qwen/Qwen3.8-Flash-Next card nor the nvidia card reports MMLU-Pro, so
there is no reference number. Checkpoint: NVFP4 (W4A4) on the routed MoE experts of the main model, ModelOpt
`MIXED_PRECISION` format; the 51B n-gram embedding table is per-tensor FP8 and the MTP routed experts are 128x128
block-scaled FP8, both copied from Qwen/Qwen3.8-Flash-Next-FP8 (MTP is not used here). Revision `fc694b5`. Loading
the FP8 table needs vLLM commit `d4d703caf` or later, which the nightly used here includes.

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
  (0-3, port 8002, so two models can run side by side on one 8-GPU host) and the local
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
`nvidia/Qwen3.8-Flash-Next-NVFP4` from the Hugging Face Hub. Loading copies the 51B n-gram table into pinned host
memory (about 50 GB per server; the table is FP8), which takes several minutes. Export `HF_TOKEN` in your shell
before running (never in the repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

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
| truncated (stop_reason=max_tokens) | 1 |
| unparsed answers | 2 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 308 | 4,755 | 4,578 |
| median | 275 | 729 | 528 |
| max | 1,767 | 131,072 | 131,072 |

Per subject: math 0.954, biology 0.939, physics 0.925, chemistry 0.920, business 0.911, computer science 0.905, economics 0.904, engineering 0.871, psychology 0.870, philosophy 0.844, other 0.823, health 0.817, history 0.803, law 0.756.

Total run time about 57 min on 4x B200 (TP=1 × DP=4), sharing the server with the other two knowledge-and-reasoning tasks of this model, which ran at the same time.

## Patches for upstream

None.
