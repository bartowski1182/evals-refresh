# Qwen/Qwen3-30B-A3B-Instruct-2507 — MRCR

| | |
|---|---|
| Task | `inspect_evals/mrcr` (openai/mrcr @ `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, 2/4/8-needle combined; filtered to served `max_model_len=262144`) |
| Category | long-context (subclass: long-context) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | no page at recipes.vllm.ai for this id; model card serve line https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507 (`vllm serve … --max-model-len 262144`). Closest sibling recipe (non-thinking Instruct-2507): https://recipes.vllm.ai/Qwen/Qwen3-235B-A22B-Instruct-2507 |

This checkpoint is the Qwen3-30B-A3B **non-thinking** Instruct-2507 update (native 262,144 context; no `<think>` blocks). Sampling matches `generation_config.json` (temperature 0.7, top_p 0.8, top_k 20). The card does not report MRCR / GDM-MRCRv2, so there is no card figure to compare.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length), taken from the model card serve line.
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

`temperature=0.7, top_p=0.8, top_k=20, max_model_len=262144, gen_kwargs={temperature:0.7, top_p:0.8, top_k:20}, max_connections=16, client_timeout=7200, epochs=1` (non-thinking Instruct-2507; no `reasoning_effort`)

the following scores and token stats are obtained:

| metric | value |
|---|---|
| mrcr_2needle (pass@1) | 0.472 (594 of 594) |
| mrcr_2needle stderr | 0.0184 |
| mrcr_2needle auc | 43.85 |
| mrcr_4needle (pass@1) | 0.247 (596 of 596) |
| mrcr_4needle stderr | 0.0143 |
| mrcr_4needle auc | 22.90 |
| mrcr_8needle (pass@1) | 0.186 (593 of 593) |
| mrcr_8needle stderr | 0.0122 |
| mrcr_8needle auc | 17.20 |
| samples | 1783 (594 / 596 / 593 after filtering 206 / 204 / 207 of 800 per needle count that exceeded `max_model_len=262144`) |
| truncated (stop_reason=max_tokens) | 4 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 63,529 | 554 | 0 |
| median | 32,596 | 402 | 0 |
| max | 261,345 | 131,072 | 0 |

Reasoning tokens are 0: this checkpoint is non-thinking and does not emit a separate reasoning content part.

MRCR writes a plain-text metrics summary beside the `.eval` log. Full `.txt` contents:

```
mrcr_2needle 0.4716490481063048
mrcr_2needle_auc 43.853524309598555
mrcr_2needle_overall 0.4716490481063048
mrcr_2needle_mean 0.4716490481063048
mrcr_2needle_stderr 0.018447787555648427
mrcr_2needle_count_gt_128k_le_256k 98.0
mrcr_2needle_gt_128k_le_256k 0.38008741934245754
mrcr_2needle_stderr_gt_128k_le_256k 0.043939834403418455
mrcr_2needle_count_gt_64k_le_128k 100.0
mrcr_2needle_gt_64k_le_128k 0.4463286189141617
mrcr_2needle_stderr_gt_64k_le_128k 0.04465553351742829
mrcr_2needle_count_gt_8k_le_16k 98.0
mrcr_2needle_gt_8k_le_16k 0.5315556508818253
mrcr_2needle_stderr_gt_8k_le_16k 0.045515816016903854
mrcr_2needle_count_gt_0_le_8k 101.0
mrcr_2needle_gt_0_le_8k 0.5009529267390531
mrcr_2needle_stderr_gt_0_le_8k 0.04587538624399341
mrcr_2needle_count_gt_32k_le_64k 98.0
mrcr_2needle_gt_32k_le_64k 0.41365207679826815
mrcr_2needle_stderr_gt_32k_le_64k 0.043982434506155126
mrcr_2needle_count_gt_16k_le_32k 99.0
mrcr_2needle_gt_16k_le_32k 0.5560757845946923
mrcr_2needle_stderr_gt_16k_le_32k 0.045496445514390846
mrcr_4needle 0.24687402355145985
mrcr_4needle_auc 22.897296407006355
mrcr_4needle_overall 0.24687402355145985
mrcr_4needle_mean 0.24687402355145985
mrcr_4needle_stderr 0.014315434035111626
mrcr_4needle_count_gt_64k_le_128k 94.0
mrcr_4needle_gt_64k_le_128k 0.21716056537492875
mrcr_4needle_stderr_gt_64k_le_128k 0.03219915889937265
mrcr_4needle_count_gt_16k_le_32k 97.0
mrcr_4needle_gt_16k_le_32k 0.23869318116168786
mrcr_4needle_stderr_gt_16k_le_32k 0.03374460619173398
mrcr_4needle_count_gt_32k_le_64k 100.0
mrcr_4needle_gt_32k_le_64k 0.28726577630797506
mrcr_4needle_stderr_gt_32k_le_64k 0.03915036022364311
mrcr_4needle_count_gt_0_le_8k 105.0
mrcr_4needle_gt_0_le_8k 0.2932986531578761
mrcr_4needle_stderr_gt_0_le_8k 0.03781052477074887
mrcr_4needle_count_gt_128k_le_256k 104.0
mrcr_4needle_gt_128k_le_256k 0.19100659464297498
mrcr_4needle_stderr_gt_128k_le_256k 0.0288431052246587
mrcr_4needle_count_gt_8k_le_16k 96.0
mrcr_4needle_gt_8k_le_16k 0.2519058777447827
mrcr_4needle_stderr_gt_8k_le_16k 0.037025243675795065
mrcr_8needle 0.18597363403644518
mrcr_8needle_auc 17.196875038997536
mrcr_8needle_overall 0.18597363403644518
mrcr_8needle_mean 0.18597363403644518
mrcr_8needle_stderr 0.012169131320486535
mrcr_8needle_count_gt_32k_le_64k 98.0
mrcr_8needle_gt_32k_le_64k 0.21417415226352862
mrcr_8needle_stderr_gt_32k_le_64k 0.03269151174289336
mrcr_8needle_count_gt_8k_le_16k 95.0
mrcr_8needle_gt_8k_le_16k 0.22020642415851385
mrcr_8needle_stderr_gt_8k_le_16k 0.03411962143612669
mrcr_8needle_count_gt_128k_le_256k 98.0
mrcr_8needle_gt_128k_le_256k 0.11712264464609932
mrcr_8needle_stderr_gt_128k_le_256k 0.01898854229709273
mrcr_8needle_count_gt_64k_le_128k 99.0
mrcr_8needle_gt_64k_le_128k 0.175954900353477
mrcr_8needle_stderr_gt_64k_le_128k 0.028017645702005337
mrcr_8needle_count_gt_16k_le_32k 98.0
mrcr_8needle_gt_16k_le_32k 0.1891796695942427
mrcr_8needle_stderr_gt_16k_le_32k 0.031075032685637674
mrcr_8needle_count_gt_0_le_8k 105.0
mrcr_8needle_gt_0_le_8k 0.1993954841540916
mrcr_8needle_stderr_gt_0_le_8k 0.031045777816088325
```

Serving was `TP=1 DP=4` on four B200s (~1.55M KV tokens per replica at 262144; `--max-connections 16`). Wall time about 114 min of eval after server start + exact-tokenizer filtering.

## Patches for upstream

Client pinned to `neuralmagic/inspect_evals@e64c20fd` (`update_mrcr`): per-needle metric blocks and the `.txt` metrics summary beside the `.eval` log. Once that lands on `main`, point the pin at the upstream commit instead.
