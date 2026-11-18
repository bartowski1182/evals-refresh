# Qwen/Qwen3.5-27B — MRCR

| | |
|---|---|
| Task | `inspect_evals/mrcr` (openai/mrcr @ `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, 2/4/8-needle combined; filtered to served `max_model_len=262144`) |
| Category | long-context (subclass: long-context) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.5-27B |

Native 262,144 context (no YaRN). Thinking is on (`enable_thinking=true`); Qwen3.5 has no `reasoning_effort` levels. Sampling follows `generation_config.json` (temperature 0.6, top_p 0.95, top_k 20); the model card recommends temperature 1.0 for thinking-mode general tasks. The card does not report MRCR / GDM-MRCRv2 for this checkpoint, so there is no card figure to compare.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, concurrency, `max_model_len`).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below:

```bash
# server: vLLM 0.34.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.34.0 "transformers>=5.8.0"
# client: Inspect + neuralmagic/inspect_evals (MRCR lives on this fork, not on PyPI)
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "git+https://github.com/neuralmagic/inspect_evals.git@e64c20fd6c49b4848461090630699d398ead7ba0" openai "transformers>=5.8.0"
```

Pre-download the weights into `HF_HUB_CACHE` and export `HF_HUB_OFFLINE=1` for the server only so DP
workers do not hit Hub rate limits. Export `HF_TOKEN` in your shell if needed (never in the repo).

### Run the eval

```bash
EPOCHS=1 ./local_orchestrator.sh --limit 5   # smoke test; inspect the samples first
./local_orchestrator.sh
inspect view --log-dir ./logs
```

## Results

With model generation config:

`temperature=0.6, top_p=0.95, top_k=20, enable_thinking=true, max_model_len=262144, gen_kwargs={temperature:0.6, top_p:0.95, top_k:20}, max_connections=16, client_timeout=7200, epochs=1` (Qwen3.5 has no `reasoning_effort` levels)

the following scores and token stats are obtained:

| metric | value |
|---|---|
| mrcr_2needle (pass@1) | 0.986 (592 of 592) |
| mrcr_2needle stderr | 0.00475 |
| mrcr_2needle auc | 98.09 |
| mrcr_4needle (pass@1) | 0.954 (592 of 592) |
| mrcr_4needle stderr | 0.00831 |
| mrcr_4needle auc | 93.38 |
| mrcr_8needle (pass@1) | 0.919 (593 of 593) |
| mrcr_8needle stderr | 0.0108 |
| mrcr_8needle auc | 89.29 |
| samples | 1777 (592 / 592 / 593 after filtering 208 / 208 / 207 of 800 per needle count that exceeded `max_model_len=262144`) |
| truncated (stop_reason=max_tokens) | 26 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 63,862 | 4,624 | 4,238 |
| median | 32,750 | 3,803 | 3,372 |
| max | 260,448 | 65,536 | 65,536 |

Reasoning is parsed into a separate content part for every sample (1777/1777). Mean output is mostly reasoning.

MRCR writes a plain-text metrics summary beside the `.eval` log. Full `.txt` contents:

```
mrcr_2needle 0.9861165072078566
mrcr_2needle_auc 98.0948281102169
mrcr_2needle_overall 0.9861165072078566
mrcr_2needle_mean 0.9861165072078566
mrcr_2needle_stderr 0.0047526033936700005
mrcr_2needle_count_gt_8k_le_16k 96.0
mrcr_2needle_gt_8k_le_16k 0.9998523476091642
mrcr_2needle_stderr_gt_8k_le_16k 3.445954133487344e-05
mrcr_2needle_count_gt_32k_le_64k 96.0
mrcr_2needle_gt_32k_le_64k 0.9999018395391035
mrcr_2needle_stderr_gt_32k_le_64k 4.6231375279955466e-05
mrcr_2needle_count_gt_0_le_8k 100.0
mrcr_2needle_gt_0_le_8k 0.9897251594323557
mrcr_2needle_stderr_gt_0_le_8k 0.009997420951296751
mrcr_2needle_count_gt_16k_le_32k 102.0
mrcr_2needle_gt_16k_le_32k 0.9998258857772981
mrcr_2needle_stderr_gt_16k_le_32k 6.165397522668496e-05
mrcr_2needle_count_gt_64k_le_128k 100.0
mrcr_2needle_gt_64k_le_128k 0.9998841647440383
mrcr_2needle_stderr_gt_64k_le_128k 2.6129687840421275e-05
mrcr_2needle_count_gt_128k_le_256k 98.0
mrcr_2needle_gt_128k_le_256k 0.9271571176927922
mrcr_2needle_stderr_gt_128k_le_256k 0.026142371411111164
mrcr_4needle 0.9539793079010789
mrcr_4needle_auc 93.38021528340597
mrcr_4needle_overall 0.9539793079010789
mrcr_4needle_mean 0.9539793079010789
mrcr_4needle_stderr 0.008305346043359092
mrcr_4needle_count_gt_8k_le_16k 95.0
mrcr_4needle_gt_8k_le_16k 0.9893842923450467
mrcr_4needle_stderr_gt_8k_le_16k 0.010422051235057296
mrcr_4needle_count_gt_16k_le_32k 96.0
mrcr_4needle_gt_16k_le_32k 0.9997558958521596
mrcr_4needle_stderr_gt_16k_le_32k 7.443627897617332e-05
mrcr_4needle_count_gt_0_le_8k 103.0
mrcr_4needle_gt_0_le_8k 0.9996912271491806
mrcr_4needle_stderr_gt_0_le_8k 0.00010012914852332794
mrcr_4needle_count_gt_128k_le_256k 104.0
mrcr_4needle_gt_128k_le_256k 0.8107268759723447
mrcr_4needle_stderr_gt_128k_le_256k 0.03711839147535111
mrcr_4needle_count_gt_32k_le_64k 100.0
mrcr_4needle_gt_32k_le_64k 0.9544553348491747
mrcr_4needle_stderr_gt_32k_le_64k 0.019901345642622466
mrcr_4needle_count_gt_64k_le_128k 94.0
mrcr_4needle_gt_64k_le_128k 0.9793441651111172
mrcr_4needle_stderr_gt_64k_le_128k 0.014163443036069782
mrcr_8needle 0.9192698322903885
mrcr_8needle_auc 89.28965180489399
mrcr_8needle_overall 0.9192698322903885
mrcr_8needle_mean 0.9192698322903885
mrcr_8needle_stderr 0.010798380590194903
mrcr_8needle_count_gt_64k_le_128k 102.0
mrcr_8needle_gt_64k_le_128k 0.9537498008161183
mrcr_8needle_stderr_gt_64k_le_128k 0.0202217421699824
mrcr_8needle_count_gt_16k_le_32k 100.0
mrcr_8needle_gt_16k_le_32k 0.990646678016272
mrcr_8needle_stderr_gt_16k_le_32k 0.009080580903113956
mrcr_8needle_count_gt_128k_le_256k 99.0
mrcr_8needle_gt_128k_le_256k 0.7126107374389169
mrcr_8needle_stderr_gt_128k_le_256k 0.04409018169984659
mrcr_8needle_count_gt_8k_le_16k 92.0
mrcr_8needle_gt_8k_le_16k 0.9391104617594918
mrcr_8needle_stderr_gt_8k_le_16k 0.024138627939346777
mrcr_8needle_count_gt_0_le_8k 105.0
mrcr_8needle_gt_0_le_8k 0.9724217291248186
mrcr_8needle_stderr_gt_0_le_8k 0.015540617504647563
mrcr_8needle_count_gt_32k_le_64k 95.0
mrcr_8needle_gt_32k_le_64k 0.9445153264936541
mrcr_8needle_stderr_gt_32k_le_64k 0.02188763243351956
```

Serving was `TP=1 DP=4` on four B200s (~1.83M KV tokens per replica at 262144; `--max-connections 16`). Wall time about 155 min of eval after server start + exact-tokenizer filtering.

## Patches for upstream

Client pinned to `neuralmagic/inspect_evals@e64c20fd` (`update_mrcr`): per-needle metric blocks and the `.txt` metrics summary beside the `.eval` log. Once that lands on `main`, point the pin at the upstream commit instead.
