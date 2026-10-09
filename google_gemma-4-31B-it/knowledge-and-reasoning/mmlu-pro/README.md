# google/gemma-4-31B-it — MMLU-Pro

| | |
|---|---|
| Task | `inspect_evals/mmlu_pro` (12,032 questions, 10 choices, 14 subjects, 0-shot) |
| Category | knowledge-and-reasoning (subclass: knowledge-recall) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Google/gemma-4-31B-it |

`fewshot=0` (the package default): every question is asked zero-shot with the package's chain-of-thought
multiple-choice prompt (`ANSWER: $LETTER`). Dataset pinned at revision `527feea0af`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, max tokens, concurrency).
- `assets/generate_config.json` — Inspect generate config that cannot be given as a CLI flag; here it
  carries the per-request `chat_template_kwargs` that enable thinking.
- `assets/tool_chat_template_gemma4.jinja` — vLLM's Gemma 4 chat template (from the recipe), used by `serve.sh`.
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
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 5`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=64, chat_template_kwargs={"enable_thinking": true}, max_tokens=131072, max_connections=256, client_timeout=7200, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy | 0.871 |
| stderr | 0.0031 |
| samples | 12,032 (1 epoch) |
| truncated (stop_reason=max_tokens) | 0 |
| unparsed answers | 0 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 276 | 3,570 | 3,006 |
| median | 244 | 2,655 | 2,062 |
| max | 1,738 | 15,947 | 14,976 |

Per subject: math 0.958, biology 0.932, chemistry 0.915, physics 0.915, business 0.904,
economics 0.902, computer science 0.885, psychology 0.862, engineering 0.836, philosophy 0.832,
other 0.808, health 0.808, history 0.780, law 0.757.

Total run time about 1 h 30 min on 8x H100 (256 concurrent samples). The model card reports 85.2%.

## Patches for upstream

None.
