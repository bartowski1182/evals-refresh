# Qwen/Qwen3.5-27B — MRCR

| | |
|---|---|
| Task | `inspect_evals/mrcr` (openai/mrcr @ `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, 2/4/8-needle combined; filtered to served `max_model_len=262144`) |
| Category | long-context (subclass: long-context) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.5-27B |

Native 262,144 context (no YaRN). Thinking is on (`enable_thinking=true`); Qwen3.5 has no `reasoning_effort` levels. Sampling follows the model card's thinking-mode general-task recommendation (`temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `repetition_penalty=1.0`), not `generation_config.json` (which sets temperature 0.6). The card does not report MRCR / GDM-MRCRv2 for this checkpoint, so there is no card figure to compare. A prior run at temperature 0.6 is kept under `logs/superseded_genconfig_t0.6/`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params via `GEN_KWARGS`, concurrency, `max_model_len`).
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

Pre-download the weights into the HF hub cache and export `HF_HUB_OFFLINE=1` for the server only so DP
workers do not hit Hub rate limits. Export `HF_TOKEN` in your shell if needed (never in the repo).

### Run the eval

```bash
EPOCHS=1 ./local_orchestrator.sh --limit 5   # smoke test; inspect the samples first
./local_orchestrator.sh
inspect view --log-dir ./logs
```

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, gen_kwargs={temperature:1.0, top_p:0.95, top_k:20, presence_penalty:1.5, repetition_penalty:1.0, do_sample:true}, enable_thinking=true, max_model_len=262144, max_connections=16, client_timeout=7200, epochs=1` (`do_sample` is ignored by vLLM's OpenAI server; Qwen3.5 has no `reasoning_effort` levels)

the following scores and token stats are obtained:

| metric | value |
|---|---|
| mrcr_2needle (pass@1) | 0.985 (592 of 592) |
| mrcr_2needle stderr | 0.00498 |
| mrcr_2needle auc | 97.85 |
| mrcr_4needle (pass@1) | 0.956 (592 of 592) |
| mrcr_4needle stderr | 0.00814 |
| mrcr_4needle auc | 93.37 |
| mrcr_8needle (pass@1) | 0.893 (593 of 593) |
| mrcr_8needle stderr | 0.0123 |
| mrcr_8needle auc | 84.84 |
| samples | 1777 (592 / 592 / 593 after filtering 208 / 208 / 207 of 800 per needle count that exceeded `max_model_len=262144`) |
| truncated (stop_reason=max_tokens) | 16 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 63,862 | 4,205 | 3,817 |
| median | 32,750 | 3,456 | 3,062 |
| max | 260,448 | 24,506 | 24,245 |

Reasoning is parsed into a separate content part for every sample (1777/1777).

MRCR writes a plain-text metrics summary beside the `.eval` log. Full `.txt` contents:

```
mrcr_2needle 0.9847803847153508
mrcr_2needle_auc 97.85248260120261
mrcr_2needle_overall 0.9847803847153508
mrcr_2needle_mean 0.9847803847153508
mrcr_2needle_stderr 0.004984672176693571
mrcr_2needle_count_gt_8k_le_16k 96.0
mrcr_2needle_gt_8k_le_16k 0.9894454161449896
mrcr_2needle_stderr_gt_8k_le_16k 0.010415268618675894
mrcr_2needle_count_gt_32k_le_64k 96.0
mrcr_2needle_gt_32k_le_64k 0.9998486932806001
mrcr_2needle_stderr_gt_32k_le_64k 6.965655888515295e-05
mrcr_2needle_count_gt_0_le_8k 100.0
mrcr_2needle_gt_0_le_8k 0.9997251594323557
mrcr_2needle_stderr_gt_0_le_8k 6.284126372132428e-05
mrcr_2needle_count_gt_16k_le_32k 102.0
mrcr_2needle_gt_16k_le_32k 0.9998158345966863
mrcr_2needle_stderr_gt_16k_le_32k 6.218990567768892e-05
mrcr_2needle_count_gt_64k_le_128k 100.0
mrcr_2needle_gt_64k_le_128k 0.999878585165823
mrcr_2needle_stderr_gt_64k_le_128k 2.794608003583025e-05
mrcr_2needle_count_gt_128k_le_256k 98.0
mrcr_2needle_gt_128k_le_256k 0.9191445271219512
mrcr_2needle_stderr_gt_128k_le_256k 0.02750120049325477
mrcr_4needle 0.9557961896561341
mrcr_4needle_auc 93.3668138837144
mrcr_4needle_overall 0.9557961896561341
mrcr_4needle_mean 0.9557961896561341
mrcr_4needle_stderr 0.00813580206419312
mrcr_4needle_count_gt_8k_le_16k 95.0
mrcr_4needle_gt_8k_le_16k 0.9893842923450467
mrcr_4needle_stderr_gt_8k_le_16k 0.010422051235057296
mrcr_4needle_count_gt_16k_le_32k 96.0
mrcr_4needle_gt_16k_le_32k 0.9922035243113688
mrcr_4needle_stderr_gt_16k_le_32k 0.007561290011844648
mrcr_4needle_count_gt_0_le_8k 103.0
mrcr_4needle_gt_0_le_8k 0.9997038852168
mrcr_4needle_stderr_gt_0_le_8k 9.969510834898122e-05
mrcr_4needle_count_gt_128k_le_256k 104.0
mrcr_4needle_gt_128k_le_256k 0.8286608139457816
mrcr_4needle_stderr_gt_128k_le_256k 0.035694042696127144
mrcr_4needle_count_gt_32k_le_64k 100.0
mrcr_4needle_gt_32k_le_64k 0.9716847756324902
mrcr_4needle_stderr_gt_32k_le_64k 0.01614663432825244
mrcr_4needle_count_gt_64k_le_128k 94.0
mrcr_4needle_gt_64k_le_128k 0.9603148487108492
mrcr_4needle_stderr_gt_64k_le_128k 0.01942016139288407
mrcr_8needle 0.8927609580112726
mrcr_8needle_auc 84.8380070846834
mrcr_8needle_overall 0.8927609580112726
mrcr_8needle_mean 0.8927609580112726
mrcr_8needle_stderr 0.012296550913452756
mrcr_8needle_count_gt_64k_le_128k 102.0
mrcr_8needle_gt_64k_le_128k 0.8958183219943118
mrcr_8needle_stderr_gt_64k_le_128k 0.029682648480686448
mrcr_8needle_count_gt_16k_le_32k 100.0
mrcr_8needle_gt_16k_le_32k 0.9808510112925315
mrcr_8needle_stderr_gt_16k_le_32k 0.01325224705328058
mrcr_8needle_count_gt_128k_le_256k 99.0
mrcr_8needle_gt_128k_le_256k 0.6280887842544877
mrcr_8needle_stderr_gt_128k_le_256k 0.04650514371237863
mrcr_8needle_count_gt_8k_le_16k 92.0
mrcr_8needle_gt_8k_le_16k 0.9282494015797695
mrcr_8needle_stderr_gt_8k_le_16k 0.026197100461133712
mrcr_8needle_count_gt_0_le_8k 105.0
mrcr_8needle_gt_0_le_8k 0.9719017916830227
mrcr_8needle_stderr_gt_0_le_8k 0.015541271867455108
mrcr_8needle_count_gt_32k_le_64k 95.0
mrcr_8needle_gt_32k_le_64k 0.9507290043659085
mrcr_8needle_stderr_gt_32k_le_64k 0.02141539332465019
```

Serving was `TP=1 DP=4` on four B200s (~1.83M KV tokens per replica at 262144; `--max-connections 16`). Wall time about 142 min of eval after server start + exact-tokenizer filtering.

## Patches for upstream

Client pinned to `neuralmagic/inspect_evals@e64c20fd` (`update_mrcr`): per-needle metric blocks and the `.txt` metrics summary beside the `.eval` log. Once that lands on `main`, point the pin at the upstream commit instead.
