# RedHatAI/Qwen3.8-27B-MXFP4xFP8 — MMLU-Pro

| | |
|---|---|
| Task | `inspect_evals/mmlu_pro` (12,032 questions, 10 choices, 14 subjects, 0-shot) |
| Category | knowledge-and-reasoning (subclass: knowledge-recall) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant covers the W4A4 checkpoint, there is no MXFP4xFP8 variant) |

`fewshot=0` (the package default): every question is asked zero-shot with the package's chain-of-thought
multiple-choice prompt (`ANSWER: $LETTER`). Dataset pinned at revision `527feea0af`. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs. The model cards report
MMLU-Pro measured with their own harnesses (RedHatAI: lighteval), not with this task.

**Checkpoint.** `RedHatAI/Qwen3.8-27B-MXFP4xFP8` is Qwen/Qwen3.8-27B with the linear layers' weights quantized to
MXFP4 and their input activations to FP8 (W4A8, compressed-tensors format), the FP8-activation counterpart of
`RedHatAI/Qwen3.8-27B-MXFP4` (W4A4). The repository was not public when this dir was written (2026-10-21): before
running, check that its `config.json` declares MXFP4 weights with dynamic FP8 input activations and that its
`generation_config.json` matches the base model's (temperature 1.0, top_p 0.95, top_k 20), and record the revision
here. The KV cache is served in BF16 (`--kv-cache-dtype bfloat16`), as for the other two models, so the three differ
only in the linear layers' quantization. vLLM must run the linear layers with FP8 activations on the Humming MXFP4
kernel (see Setup); a vLLM that falls back to MXFP4 activations evaluates the W4A4 checkpoint instead.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the Qwen/Qwen3.8-27B recipe, plus the BF16 KV cache; identical in the three dirs apart from
  `MODEL`.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, max tokens, concurrency); identical in the three dirs apart from `MODEL`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM server environment and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` from the vLLM podman image (`VLLM_IMAGE`, by default the pinned one) or a venv (`VLLM_VENV`) through `tools/vllm_server.sh` (GPUs 0-7, port 8000, TP=2 x DP=4 from `serve.sh`, the HF cache and the task dir mounted in podman mode), waits for the server, activates the client venv (`INSPECT_VENV`), runs `run_eval.sh`, stops the server on exit.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Pull the server image and create the client environment as below. The commands use the default server image
(`DEFAULT_VLLM_IMAGE`) and client venv path (`INSPECT_VENV`); if you keep them elsewhere, set those variables in the
environment or change them in `local_orchestrator.sh`. Once the results are in, `server_requirements.txt`
(`tools/vllm_server.sh` provides `vllm_freeze`, which prints `pip freeze` of the selected environment) and
`client_requirements.txt` hold the exact packages that produced them (host details in their headers); create the
client environment from its file instead. Use the same server and client environments for the three dirs of the
comparison set.

```bash
# server: the vLLM 0.32.0 podman image, docker.io/vllm/vllm-openai:v0.32.0 pinned by digest (the
# DEFAULT_VLLM_IMAGE), the same image for the three dirs. 0.32.0 is the first release with the dense MXFP4-weight /
# FP8-activation scheme the MXFP4xFP8 checkpoint needs; the humming-kernels dependency of its CUDA wheels provides the
# only MXFP4 GEMM of that release that takes FP8 activations (CUDA, SM75+), so the comparison set needs an NVIDIA host.
# GPU access from the container goes through the NVIDIA Container Toolkit's CDI spec (generated once per host, as root).
podman pull docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# To serve from a venv instead, create one with the vLLM version server_requirements.txt records (0.32.0, whose CUDA
# wheels pull in humming-kernels) and pass it as VLLM_VENV; the image is the reference environment.
# client: Inspect
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install "inspect-ai==0.3.263" "inspect-evals==0.20.0" "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the log must say
`Using HummingMxFp4LinearKernel for MXFP4 GEMM`, and the KV cache dtype must be `bfloat16`. A warning `No MXFP4 linear
kernel supports ... falling back to MXFP4 activations` (no Humming, or a ROCm host, whose AITER kernel is W4A4 only)
means the checkpoint runs as W4A4, i.e. as `RedHatAI/Qwen3.8-27B-MXFP4`, and the run is not the one this dir
describes.

`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache (`HF_HOME`,
mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or set
`MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=131072, max_connections=256, client_timeout=7200, kv_cache_dtype=bfloat16, TP=2 DP=4, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None.
