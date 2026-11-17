# RedHatAI/Qwen3.8-27B-NVFP4 — MRCR

| | |
|---|---|
| Task | `inspect_evals/mrcr` (openai/mrcr @ `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, 2/4/8-needle combined; filtered to served `max_model_len=262144`) |
| Category | long-context (subclass: long-context) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |

The Qwen3.8-27B card reports Context Arena / GDM-MRCRv2 averages over runs that include contexts up to 1M (YaRN). This dir serves the native 262,144 window (recipe NVFP4 low-latency path, no YaRN), so samples longer than that are dropped by the task's `max_model_len` filter and the number is not comparable to the card's full 1M figure. Sampling matches `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20) with thinking at `xhigh`.

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

Pre-download the weights into `HF_HUB_CACHE` (this host uses `/data/hub_cache`) and export `HF_HUB_OFFLINE=1`
for the server only so DP workers do not hit Hub rate limits. Export `HF_TOKEN` in your shell if needed (never in the repo).

### Run the eval

```bash
EPOCHS=1 ./local_orchestrator.sh --limit 5   # smoke test; inspect the samples first
./local_orchestrator.sh
inspect view --log-dir ./logs
```

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_model_len=262144, gen_kwargs={temperature:1.0, top_p:0.95, top_k:20}, max_connections=32, client_timeout=7200, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| mrcr_2needle (pass@1) | 0.976 (589 of 589) |
| mrcr_2needle stderr | 0.00562 |
| mrcr_2needle auc | 96.33 |
| mrcr_4needle (pass@1) | 0.942 (591 of 591) |
| mrcr_4needle stderr | 0.00900 |
| mrcr_4needle auc | 91.75 |
| mrcr_8needle (pass@1) | 0.808 (593 of 593) |
| mrcr_8needle stderr | 0.0154 |
| mrcr_8needle auc | 75.13 |
| samples | 1773 (589 / 591 / 593 after filtering 211 / 209 / 207 of 800 per needle count that exceeded `max_model_len=262144`) |
| truncated (stop_reason=max_tokens) | 8 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 64,080 | 1,544 | 1,152 |
| median | 32,951 | 967 | 614 |
| max | 261,618 | 35,746 | 35,270 |

MRCR writes a plain-text metrics summary beside the `.eval` log. Full `.txt` contents:

```
mrcr_2needle 0.9764882673891128
mrcr_2needle_auc 96.32939129690519
mrcr_2needle_overall 0.9764882673891128
mrcr_2needle_mean 0.9764882673891128
mrcr_2needle_stderr 0.0056175777277548454
mrcr_2needle_count_gt_0_le_8k 100.0
mrcr_2needle_gt_0_le_8k 0.9986603735200216
mrcr_2needle_stderr_gt_0_le_8k 0.0006590575429467069
mrcr_2needle_count_gt_32k_le_64k 97.0
mrcr_2needle_gt_32k_le_64k 0.9839623502150183
mrcr_2needle_stderr_gt_32k_le_64k 0.010719585835314578
mrcr_2needle_count_gt_8k_le_16k 94.0
mrcr_2needle_gt_8k_le_16k 0.9951782920503199
mrcr_2needle_stderr_gt_8k_le_16k 0.0037326509885169798
mrcr_2needle_count_gt_16k_le_32k 101.0
mrcr_2needle_gt_16k_le_32k 0.9869786217786642
mrcr_2needle_stderr_gt_16k_le_32k 0.010010780837716868
mrcr_2needle_count_gt_64k_le_128k 100.0
mrcr_2needle_gt_64k_le_128k 0.9632901884355404
mrcr_2needle_stderr_gt_64k_le_128k 0.01775325852183651
mrcr_2needle_count_gt_128k_le_256k 97.0
mrcr_2needle_gt_128k_le_256k 0.930727681169066
mrcr_2needle_stderr_gt_128k_le_256k 0.023815612101905255
mrcr_4needle 0.9420288532350912
mrcr_4needle_auc 91.75346785872077
mrcr_4needle_overall 0.9420288532350912
mrcr_4needle_mean 0.9420288532350912
mrcr_4needle_stderr 0.009003713910215035
mrcr_4needle_count_gt_128k_le_256k 104.0
mrcr_4needle_gt_128k_le_256k 0.8717373584005613
mrcr_4needle_stderr_gt_128k_le_256k 0.03105980259443003
mrcr_4needle_count_gt_8k_le_16k 95.0
mrcr_4needle_gt_8k_le_16k 0.9838473970463508
mrcr_4needle_stderr_gt_8k_le_16k 0.010705598727068515
mrcr_4needle_count_gt_16k_le_32k 94.0
mrcr_4needle_gt_16k_le_32k 0.9767042500289055
mrcr_4needle_stderr_gt_16k_le_32k 0.014577531817249552
mrcr_4needle_count_gt_32k_le_64k 101.0
mrcr_4needle_gt_32k_le_64k 0.9355744373512278
mrcr_4needle_stderr_gt_32k_le_64k 0.02351900719050813
mrcr_4needle_count_gt_64k_le_128k 95.0
mrcr_4needle_gt_64k_le_128k 0.9135405338080425
mrcr_4needle_stderr_gt_64k_le_128k 0.02716772531003692
mrcr_4needle_count_gt_0_le_8k 102.0
mrcr_4needle_gt_0_le_8k 0.9757185870776675
mrcr_4needle_stderr_gt_0_le_8k 0.01352354454664478
mrcr_8needle 0.8080135116468218
mrcr_8needle_auc 75.1269190891619
mrcr_8needle_overall 0.8080135116468218
mrcr_8needle_mean 0.8080135116468218
mrcr_8needle_stderr 0.015371903954980023
mrcr_8needle_count_gt_32k_le_64k 97.0
mrcr_8needle_gt_32k_le_64k 0.8113361288512613
mrcr_8needle_stderr_gt_32k_le_64k 0.03773795316748491
mrcr_8needle_count_gt_16k_le_32k 98.0
mrcr_8needle_gt_16k_le_32k 0.8566772469070301
mrcr_8needle_stderr_gt_16k_le_32k 0.03390160533198774
mrcr_8needle_count_gt_128k_le_256k 101.0
mrcr_8needle_gt_128k_le_256k 0.5256711740622166
mrcr_8needle_stderr_gt_128k_le_256k 0.04606879901586345
mrcr_8needle_count_gt_64k_le_128k 101.0
mrcr_8needle_gt_64k_le_128k 0.8201346417837964
mrcr_8needle_stderr_gt_64k_le_128k 0.03682768071194763
mrcr_8needle_count_gt_8k_le_16k 94.0
mrcr_8needle_gt_8k_le_16k 0.8844710231633262
mrcr_8needle_stderr_gt_8k_le_16k 0.031669539395366715
mrcr_8needle_count_gt_0_le_8k 102.0
mrcr_8needle_gt_0_le_8k 0.9552095503265101
mrcr_8needle_stderr_gt_0_le_8k 0.019218381255945553
```

Serving was `TP=1 DP=4` on four B200s (~4.43M KV tokens per replica at 262144). Wall time about 34 min of eval after ~9 min of server start + exact-tokenizer filtering.

## Patches for upstream

Client pinned to `neuralmagic/inspect_evals@e64c20fd` (`update_mrcr`): per-needle metric blocks and the `.txt` metrics summary beside the `.eval` log. Once that lands on `main`, point the pin at the upstream commit instead.
