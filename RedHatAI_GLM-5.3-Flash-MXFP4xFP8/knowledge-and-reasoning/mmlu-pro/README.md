# RedHatAI/GLM-5.3-Flash-MXFP4xFP8 — MMLU-Pro

| | |
|---|---|
| Task | `inspect_evals/mmlu_pro` (12,032 questions, 10 choices, 14 subjects, 0-shot) |
| Category | knowledge-and-reasoning (subclass: knowledge-recall) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash (the base model's; the recipes site has no MXFP4xFP8 variant, and its NVFP4 variant is Blackwell-only) |

`fewshot=0` (the package default): every question is asked zero-shot with the package's chain-of-thought multiple-choice prompt (`ANSWER: $LETTER`). Dataset pinned at revision `527feea0af` (inside `inspect_evals`). One epoch (pass@1): 12,032 questions leave a standard error below half a point. This dir is one of the two identical copies of the task that make up the GLM-5.3-Flash pair
(`zai-org_GLM-5.3-Flash`, the native FP8 model, and `RedHatAI_GLM-5.3-Flash-MXFP4xFP8`): same task, prompt, sampling,
limits, concurrency, epochs, vLLM release and serving topology (8x H200, one TP=8 replica, BF16 KV cache, MTP), so the
two numbers are like for like; only the checkpoint differs. The GLM-5.3-Flash card shows its scores in a chart and
states no protocol for this benchmark; the number below is not comparable to it.

**Checkpoint.** `RedHatAI/GLM-5.3-Flash-MXFP4xFP8` is zai-org/GLM-5.3-Flash (native FP8) with the MoE experts'
weights quantized to MXFP4 and their input activations to FP8 (W4A8, compressed-tensors format, LLM Compressor). The KV cache is BF16 on Hopper for both models of the pair (see `serve.sh`).
Evaluated: `inference-optimization/GLM-5.3-Flash-MXFP4xFP8` revision `d5ea13e50a801fe297dc79a3953dcc0203852ba3`
(the published checkpoint with the `config.json` target-regex fix described in the pair's agentic READMEs; served
under the name `RedHatAI/GLM-5.3-Flash-MXFP4xFP8`). Experts: MXFP4 weights (group 32) with dynamic per-group FP8 activations;
dense MLP, shared experts and the MLA projections of every fourth layer: FP8 block weights with dynamic FP8
activations; everything else as in the base model. `generation_config.json`, chat template and tokenizer are
byte-identical to zai-org/GLM-5.3-Flash. vLLM loads the experts with `CompressedTensorsW4A8Mxfp4MoEMethod` on the
Humming MXFP4 MoE backend (FP8 activation schema, no fallback) and the dense layers with `CompressedTensorsW8A8Fp8`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  speculative decoding, reasoning/tool parsers, thinking default, KV-cache dtype), taken from the zai-org/GLM-5.3-Flash
  recipe's Hopper flags with the BF16 KV cache, `--max-model-len 262144` and `--language-model-only`; identical to the
  pair's other task dirs, and in the two dirs of the pair apart from `MODEL` and the checkpoint comments.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task, epochs, sampling
  params, reasoning effort, max tokens, concurrency); identical in the two dirs apart from `MODEL`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM server environment and of the
  Inspect venv that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the
  final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` from the vLLM podman
  image (`VLLM_IMAGE`, by default the pinned one) or a venv (`VLLM_VENV`) through `tools/vllm_server.sh` (GPUs 0-7,
  port 8000, one TP=8 replica, the HF cache and the task dir mounted in podman mode), waits for the server, activates
  the client venv (`INSPECT_VENV`), runs `run_eval.sh`, stops the server on exit. **This is the only file you should
  modify** to match the environment you run in.

## Run

### Setup (once per host)

Pull the server image and create the client environment as below. The commands use the default server image
(`DEFAULT_VLLM_IMAGE`) and client venv path (`INSPECT_VENV`); if you keep them elsewhere, set those variables in the
environment or change them in `local_orchestrator.sh`. Once the results are in, `server_requirements.txt` and
`client_requirements.txt` hold the exact packages that produced them (host details in their headers); create the
client environment from its file instead. Use the same server and client environments for the two dirs of the pair.

```bash
# server: the vLLM 0.32.0 podman image, docker.io/vllm/vllm-openai:v0.32.0 pinned by digest (the DEFAULT_VLLM_IMAGE),
# the same image for the two dirs. 0.32.0 is the first release whose model registry carries glm5_next and the first
# with the compressed-tensors MXFP4-weight / FP8-activation paths the MXFP4xFP8 checkpoint needs. GPU access from the
# container goes through the NVIDIA Container Toolkit's CDI spec (generated once per host, as root).
podman pull docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# client: Inspect + inspect_evals
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install "inspect-ai==0.3.263" "inspect-evals==0.20.0" "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the expert layers must load
with `CompressedTensorsW4A8Mxfp4MoEMethod` (MXFP4 weights, FP8 activations; a vLLM without vllm-project/vllm#59815
loads the checkpoint without error and runs the experts with BF16 activations), the KV cache dtype must be
`bfloat16`, and MTP must be active (`num_speculative_tokens=5`). If the checkpoint also quantizes dense linear
layers to MXFP4, the log must say `Using HummingMxFp4LinearKernel for MXFP4 GEMM` and not warn `No MXFP4 linear
kernel supports ... falling back to MXFP4 activations` (that fallback runs those layers as W4A4, which is not the
checkpoint this dir describes).

`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache
(`HF_HOME`, mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or
set `MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the
repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 5`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5), with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=0.95, top_k unset, reasoning_effort=max, max_tokens=163840, max_connections=256, client_timeout=7200, kv_cache_dtype=bfloat16, TP=8 DP=1 with MTP (num_speculative_tokens=5), max_model_len=262144, epochs=1 (pass@1)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None.
