# RedHatAI/GLM-5.3-NVFP4 — MRCR

| | |
|---|---|
| Task | `inspect_evals/mrcr` (openai/mrcr @ `f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d`, 2/4/8-needle combined; filtered to served `max_model_len=262144`) |
| Category | long-context (subclass: long-context) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | no page at recipes.vllm.ai for this HF id; model card https://huggingface.co/RedHatAI/GLM-5.3-NVFP4. Closest base recipe: https://recipes.vllm.ai/zai-org/GLM-5.3 |

Native checkpoint window is 1M; this dir serves `max_model_len=262144` (DSA indexer warm-up OOM at 1M; AGENTS.md §9), so samples longer than that are dropped and the number is not comparable to a full-1M figure. Thinking is always on at `reasoning_effort=max`. Sampling: temperature 1.0, top_p 0.95 (base `zai-org/GLM-5.3` `generation_config.json`; the RedHatAI file lists temperature only). Speculative decoding left off. Topology: TP=2 × DP=2 on four B200s (user request; card example uses TP=4). On non-contiguous `CUDA_VISIBLE_DEVICES`, SymmMem / FlashInfer-MNNVL hangs at startup; this dir disables those paths (see `serve.sh` / `local_orchestrator.sh`).

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning parser, KV dtype), taken from the model card.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling, reasoning effort, concurrency, `max_model_len`).
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
pip install -r server_requirements.txt          # or: pip install vllm==0.34.0 "transformers>=5.15.0"
# client: Inspect + neuralmagic/inspect_evals (MRCR lives on this fork, not on PyPI)
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "git+https://github.com/neuralmagic/inspect_evals.git@e64c20fd6c49b4848461090630699d398ead7ba0" openai "transformers>=5.15.0"
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

`temperature=1.0, top_p=0.95, reasoning_effort=max, gen_kwargs={temperature:1.0, top_p:0.95}, max_model_len=262144, max_connections=6, client_timeout=7200, epochs=1` (serving: TP=2 × DP=2, `--kv-cache-dtype fp8_e4m3`, `--reasoning-parser glm45`, `--chat-template-content-format string`, `--disable-custom-all-reduce`, `--attention-config.sparse_mla_shard_q=false`, `--compilation-config.pass_config.fuse_allreduce_rms=false`; env `VLLM_ALLREDUCE_USE_SYMM_MEM=0`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`, `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`)

the following scores and token stats are obtained:

| metric | value |
|---|---|
| mrcr_2needle (pass@1) | 0.983 (600 of 600) |
| mrcr_2needle stderr | 0.00459 |
| mrcr_2needle auc | 97.64 |
| mrcr_4needle (pass@1) | 0.955 (599 of 599) |
| mrcr_4needle stderr | 0.00792 |
| mrcr_4needle auc | 94.09 |
| mrcr_8needle (pass@1) | 0.825 (599 of 599) |
| mrcr_8needle stderr | 0.0147 |
| mrcr_8needle auc | 78.36 |
| samples | 1798 (600 / 599 / 599 after filtering 200 / 201 / 201 of 800 per needle count that exceeded `max_model_len=262144`) |
| truncated (stop_reason=max_tokens) | 16 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 64,089 | 2,200 | 1,821 |
| median | 32,656 | 1,114 | 712 |
| max | 261,468 | 131,072 | 131,072 |

Reasoning is parsed into a separate content part for every sample (1798/1798). KV pool was ~950,272 tokens per DP replica (~3.62× at full window).

MRCR writes a plain-text metrics summary beside the `.eval` log. Full `.txt` contents:

```
mrcr_2needle 0.9827293567187085
mrcr_2needle_auc 97.63931381580227
mrcr_2needle_overall 0.9827293567187085
mrcr_2needle_mean 0.9827293567187085
mrcr_2needle_stderr 0.004590688466736246
mrcr_2needle_count_gt_64k_le_128k 99.0
mrcr_2needle_gt_64k_le_128k 0.9824291383081712
mrcr_2needle_stderr_gt_64k_le_128k 0.011607884932877515
mrcr_2needle_count_gt_32k_le_64k 98.0
mrcr_2needle_gt_32k_le_64k 0.956515726456871
mrcr_2needle_stderr_gt_32k_le_64k 0.018026703645444328
mrcr_2needle_count_gt_16k_le_32k 101.0
mrcr_2needle_gt_16k_le_32k 0.9968143365006892
mrcr_2needle_stderr_gt_16k_le_32k 0.0017605259451585712
mrcr_2needle_count_gt_0_le_8k 103.0
mrcr_2needle_gt_0_le_8k 0.9939656222834374
mrcr_2needle_stderr_gt_0_le_8k 0.0055188131816749555
mrcr_2needle_count_gt_8k_le_16k 98.0
mrcr_2needle_gt_8k_le_16k 0.9962858287558171
mrcr_2needle_stderr_gt_8k_le_16k 0.0034768470339128182
mrcr_2needle_count_gt_128k_le_256k 101.0
mrcr_2needle_gt_128k_le_256k 0.9697610875852383
mrcr_2needle_stderr_gt_128k_le_256k 0.015970864598098426
mrcr_4needle 0.9546883772026183
mrcr_4needle_auc 94.0919559739282
mrcr_4needle_overall 0.9546883772026183
mrcr_4needle_mean 0.9546883772026183
mrcr_4needle_stderr 0.007917604972191922
mrcr_4needle_count_gt_32k_le_64k 101.0
mrcr_4needle_gt_32k_le_64k 0.9237692361020101
mrcr_4needle_stderr_gt_32k_le_64k 0.025787104840896337
mrcr_4needle_count_gt_0_le_8k 105.0
mrcr_4needle_gt_0_le_8k 0.9927723705378046
mrcr_4needle_stderr_gt_0_le_8k 0.004643699323672526
mrcr_4needle_count_gt_128k_le_256k 99.0
mrcr_4needle_gt_128k_le_256k 0.9094887501068453
mrcr_4needle_stderr_gt_128k_le_256k 0.02718183817894204
mrcr_4needle_count_gt_64k_le_128k 100.0
mrcr_4needle_gt_64k_le_128k 0.9550103938750988
mrcr_4needle_stderr_gt_64k_le_128k 0.019622273527820165
mrcr_4needle_count_gt_8k_le_16k 98.0
mrcr_4needle_gt_8k_le_16k 0.9637015963544352
mrcr_4needle_stderr_gt_8k_le_16k 0.017177881919005048
mrcr_4needle_count_gt_16k_le_32k 96.0
mrcr_4needle_gt_16k_le_32k 0.9826392093830588
mrcr_4needle_stderr_gt_16k_le_32k 0.011394687009468293
mrcr_8needle 0.8253625537415211
mrcr_8needle_auc 78.36424836452372
mrcr_8needle_overall 0.8253625537415211
mrcr_8needle_mean 0.8253625537415211
mrcr_8needle_stderr 0.01470687703496692
mrcr_8needle_count_gt_128k_le_256k 99.0
mrcr_8needle_gt_128k_le_256k 0.6630473084563205
mrcr_8needle_stderr_gt_128k_le_256k 0.0453387458821857
mrcr_8needle_count_gt_16k_le_32k 98.0
mrcr_8needle_gt_16k_le_32k 0.8484611930730651
mrcr_8needle_stderr_gt_16k_le_32k 0.034701750887099654
mrcr_8needle_count_gt_0_le_8k 107.0
mrcr_8needle_gt_0_le_8k 0.9277824500283621
mrcr_8needle_stderr_gt_0_le_8k 0.023389600395233186
mrcr_8needle_count_gt_8k_le_16k 95.0
mrcr_8needle_gt_8k_le_16k 0.8735189232231034
mrcr_8needle_stderr_gt_8k_le_16k 0.03198281582780342
mrcr_8needle_count_gt_32k_le_64k 99.0
mrcr_8needle_gt_32k_le_64k 0.8222744218532757
mrcr_8needle_stderr_gt_32k_le_64k 0.03616409115980458
mrcr_8needle_count_gt_64k_le_128k 101.0
mrcr_8needle_gt_64k_le_128k 0.811278233763675
mrcr_8needle_stderr_gt_64k_le_128k 0.037275100478657466
```

Serving was `TP=2 DP=2` on four B200s (~950k KV tokens per replica at 262144; `--max-connections 6`). Wall time about 6h12m of eval after server start + exact-tokenizer filtering.

## Patches for upstream

Client pinned to `neuralmagic/inspect_evals@e64c20fd` (`update_mrcr`): per-needle metric blocks and the `.txt` metrics summary beside the `.eval` log. Once that lands on `main`, point the pin at the upstream commit instead.
