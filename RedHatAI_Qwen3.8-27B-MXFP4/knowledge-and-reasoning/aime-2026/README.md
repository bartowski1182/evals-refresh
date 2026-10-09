# RedHatAI/Qwen3.8-27B-MXFP4 — AIME 2026

| | |
|---|---|
| Task | `inspect_evals/aime2026` (30 problems (AIME I + II), integer answers) |
| Category | knowledge-and-reasoning (subclass: math-reasoning) |
| Exec tier | `none` — no Docker sandbox, no tools (the vLLM server itself runs in a container) |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; it has no MXFP4 variant) |

This is Qwen/Qwen3.8-27B with the linear layers' weights and activations quantized to MXFP4 (W4A4, GPTQ with AWQ
smoothing, LLM Compressor, compressed-tensors format), checkpoint revision `651eed99386d`. Task and prompt are those
of the base model's run (`../../../Qwen_Qwen3.8-27B/knowledge-and-reasoning/aime-2026`): answers are matched
numerically after stripping `\boxed{}`; the prompt asks for `ANSWER: $ANSWER` on the last line. Dataset pinned at
revision `79037aebdb`. Sampling, reasoning effort, token budget and Inspect environment are the base model's. Two
things differ: avg@3 (here in two runs, 1 + 2 epochs) instead of the base model's avg@8, and the hardware and serving
stack (AMD MI355X, vLLM 0.31.0 for ROCm, instead of NVIDIA H100, vLLM 0.29.0). The checkpoint also declares a static
per-tensor FP8 KV-cache scheme (calibrated `k_scale`/`v_scale`), which vLLM applies by default; `serve.sh` overrides
it with `--kv-cache-dtype bfloat16`, so the KV cache is BF16 as for the base model and for
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`, and only the MXFP4 quantization differs between the two checkpoints. MXFP4 GEMMs
run natively on gfx950 (vLLM's `AiterMxfp4LinearKernel`); attention runs on vLLM's Triton backend
(`--attention-backend TRITON_ATTN`) instead of its ROCm default, which decodes about 14x slower at long context. The
model card does not report AIME 2026.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the base model's recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, reasoning effort, max tokens, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` inside the vLLM container and of the
  Inspect venv that produced the results below, with the image and host (GPUs, driver, ROCm, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: sets GPUs, starts `serve.sh` inside the
  vLLM ROCm container in the background, activates the Inspect venv, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

`local_orchestrator.sh` runs `serve.sh` in the official vLLM ROCm image under rootless podman (it needs read-write
access to `/dev/kfd` and `/dev/dri/renderD*`) and the Inspect client from the virtualenv `~/.venvs/inspect`; create
them as below (edit the image and paths in `local_orchestrator.sh` if you run vLLM differently, e.g. from vLLM's
ROCm wheels on a host with glibc >= 2.39). `server_requirements.txt` is the `pip freeze` of that image;
`client_requirements.txt` is the exact `pip freeze` of the client environment that produced the results below,
the same packages as the base model's client environment.

```bash
# server: vLLM 0.31.0 for ROCm 7.2, from the official image pinned by digest
podman pull docker.io/vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f
# client: Inspect (Python 3.12)
uv venv --python 3.12 --seed ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "inspect-evals==0.20.0" openai
# weights: into the HF hub cache that local_orchestrator.sh mounts into the container (HF_HUB_CACHE, default ~/hf_hub_cache)
HF_HUB_CACHE=~/hf_hub_cache hf download RedHatAI/Qwen3.8-27B-MXFP4 --revision 651eed99386dae5645cd4949024e8ebd163a533b
```

Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited. The server runs with `HF_HUB_OFFLINE=1` and needs the weights downloaded as above.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 5`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=131072, max_connections=128, client_timeout=7200, epochs=3 (mean; 1 + 2 epochs in two runs)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (avg@3) | 0.9778 |
| stderr | 0.0154 |
| per-epoch accuracy | 1.000 / 0.967 / 0.967 |
| samples | 90 (30 problems x 3 epochs, in 2 runs: 30 + 60) |
| problems solved 3/3 | 28 (problem 10: 2/3, problem 20: 2/3) |
| truncated (stop_reason=max_tokens) | 0 |
| unparsed answers | 0 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 274 | 24,779 | 23,945 |
| median | 271 | 20,378 | 19,824 |
| max | 512 | 130,325 | 128,590 |

Run time: about 17 min for the 1-epoch run on 1x MI355X and 26 min for the 2-epoch run on 3x MI355X (TP=1, one replica per GPU; 128 concurrent samples).

## Patches for upstream

None.
