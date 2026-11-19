# Qwen/Qwen3.8-27B — MRCR

| | |
|---|---|
| Task | `inspect_evals/mrcr` (openai/mrcr @ `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, 2/4/8-needle combined; filtered to served `max_model_len=262144`) |
| Category | long-context (subclass: long-context) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |

The Qwen3.8-27B card reports Context Arena / GDM-MRCRv2 averages over runs that include contexts up to 1M (YaRN). This dir serves the native 262,144 window (no YaRN), so samples longer than that are dropped by the task's `max_model_len` filter and the number is not comparable to the card's full 1M figure. Sampling matches `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20) with thinking at `xhigh`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, concurrency, `max_model_len`).
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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_model_len=262144, gen_kwargs={temperature:1.0, top_p:0.95, top_k:20}, max_connections=16, client_timeout=7200, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| mrcr_2needle (pass@1) | 0.981 (589 of 589) |
| mrcr_2needle stderr | 0.00518 |
| mrcr_2needle auc | 97.58 |
| mrcr_4needle (pass@1) | 0.950 (591 of 591) |
| mrcr_4needle stderr | 0.00857 |
| mrcr_4needle auc | 93.48 |
| mrcr_8needle (pass@1) | 0.848 (593 of 593) |
| mrcr_8needle stderr | 0.0140 |
| mrcr_8needle auc | 80.88 |
| samples | 1773 (589 / 591 / 593 after filtering 211 / 209 / 207 of 800 per needle count that exceeded `max_model_len=262144`) |
| truncated (stop_reason=max_tokens) | 4 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 64,080 | 1,356 | 971 |
| median | 32,951 | 1,045 | 688 |
| max | 261,618 | 22,181 | 21,741 |

MRCR writes a plain-text metrics summary beside the `.eval` log. Full `.txt` contents:

```
mrcr_2needle 0.9809828253663506
mrcr_2needle_auc 97.57699771057993
mrcr_2needle_overall 0.9809828253663506
mrcr_2needle_mean 0.9809828253663506
mrcr_2needle_stderr 0.005175249235469698
mrcr_2needle_count_gt_0_le_8k 100.0
mrcr_2needle_gt_0_le_8k 0.9784436733156809
mrcr_2needle_stderr_gt_0_le_8k 0.013876626271640782
mrcr_2needle_count_gt_32k_le_64k 97.0
mrcr_2needle_gt_32k_le_64k 0.985655097687892
mrcr_2needle_stderr_gt_32k_le_64k 0.010584392493788371
mrcr_2needle_count_gt_8k_le_16k 94.0
mrcr_2needle_gt_8k_le_16k 0.9882649063737511
mrcr_2needle_stderr_gt_8k_le_16k 0.01066742410345362
mrcr_2needle_count_gt_16k_le_32k 101.0
mrcr_2needle_gt_16k_le_32k 0.9887457253814621
mrcr_2needle_stderr_gt_16k_le_32k 0.009911967115745279
mrcr_2needle_count_gt_64k_le_128k 100.0
mrcr_2needle_gt_64k_le_128k 0.9583268220246606
mrcr_2needle_stderr_gt_64k_le_128k 0.019626475914927315
mrcr_2needle_count_gt_128k_le_256k 97.0
mrcr_2needle_gt_128k_le_256k 0.9871450584367064
mrcr_2needle_stderr_gt_128k_le_256k 0.007004828078800255
mrcr_4needle 0.9504273075838023
mrcr_4needle_auc 93.4843649095913
mrcr_4needle_overall 0.9504273075838023
mrcr_4needle_mean 0.9504273075838023
mrcr_4needle_stderr 0.008567772746118005
mrcr_4needle_count_gt_128k_le_256k 104.0
mrcr_4needle_gt_128k_le_256k 0.9123901697110853
mrcr_4needle_stderr_gt_128k_le_256k 0.025805878765437603
mrcr_4needle_count_gt_8k_le_16k 95.0
mrcr_4needle_gt_8k_le_16k 0.9476682927244603
mrcr_4needle_stderr_gt_8k_le_16k 0.022098760794078815
mrcr_4needle_count_gt_16k_le_32k 94.0
mrcr_4needle_gt_16k_le_32k 0.975228214168362
mrcr_4needle_stderr_gt_16k_le_32k 0.015161955898105846
mrcr_4needle_count_gt_32k_le_64k 101.0
mrcr_4needle_gt_32k_le_64k 0.9323675334799932
mrcr_4needle_stderr_gt_32k_le_64k 0.024504949599846924
mrcr_4needle_count_gt_64k_le_128k 95.0
mrcr_4needle_gt_64k_le_128k 0.9366517441478465
mrcr_4needle_stderr_gt_64k_le_128k 0.024862034324952285
mrcr_4needle_count_gt_0_le_8k 102.0
mrcr_4needle_gt_0_le_8k 0.9996371040774494
mrcr_4needle_stderr_gt_0_le_8k 0.0001261995007998512
mrcr_8needle 0.848293411409883
mrcr_8needle_auc 80.87958144630032
mrcr_8needle_overall 0.848293411409883
mrcr_8needle_mean 0.848293411409883
mrcr_8needle_stderr 0.01398808336065562
mrcr_8needle_count_gt_32k_le_64k 97.0
mrcr_8needle_gt_32k_le_64k 0.8277884892119443
mrcr_8needle_stderr_gt_32k_le_64k 0.036780876188672464
mrcr_8needle_count_gt_16k_le_32k 98.0
mrcr_8needle_gt_16k_le_32k 0.9511191529795151
mrcr_8needle_stderr_gt_16k_le_32k 0.020779622698069627
mrcr_8needle_count_gt_128k_le_256k 101.0
mrcr_8needle_gt_128k_le_256k 0.7118397566589801
mrcr_8needle_stderr_gt_128k_le_256k 0.042235060979278746
mrcr_8needle_count_gt_64k_le_128k 101.0
mrcr_8needle_gt_64k_le_128k 0.8191226073505194
mrcr_8needle_stderr_gt_64k_le_128k 0.03589326576355305
mrcr_8needle_count_gt_8k_le_16k 94.0
mrcr_8needle_gt_8k_le_16k 0.8381435793776968
mrcr_8needle_stderr_gt_8k_le_16k 0.03683712489638715
mrcr_8needle_count_gt_0_le_8k 102.0
mrcr_8needle_gt_0_le_8k 0.9423542871965347
mrcr_8needle_stderr_gt_0_le_8k 0.02233884321288989
```

Serving was `TP=1 DP=4` on four B200s (~1.83M KV tokens per replica at 262144; `--max-connections 16`). Wall time about 65 min of eval after ~8 min of server start + exact-tokenizer filtering.

## Patches for upstream

Client pinned to `neuralmagic/inspect_evals@e64c20fd` (`update_mrcr`): per-needle metric blocks and the `.txt` metrics summary beside the `.eval` log. Once that lands on `main`, point the pin at the upstream commit instead.
