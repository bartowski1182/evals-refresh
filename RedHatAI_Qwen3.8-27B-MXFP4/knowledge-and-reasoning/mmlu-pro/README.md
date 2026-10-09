# RedHatAI/Qwen3.8-27B-MXFP4 — MMLU-Pro

| | |
|---|---|
| Task | `inspect_evals/mmlu_pro` (12,032 questions, 10 choices, 14 subjects, 0-shot) |
| Category | knowledge-and-reasoning (subclass: knowledge-recall) |
| Exec tier | `none` — no Docker sandbox, no tools (the vLLM server itself runs in a container) |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; it has no MXFP4 variant) |

This is Qwen/Qwen3.8-27B with the linear layers' weights and activations quantized to MXFP4 (W4A4, GPTQ with AWQ
smoothing, LLM Compressor, compressed-tensors format), checkpoint revision `651eed99386d`. Task and prompt are those
of the base model's run (`../../../Qwen_Qwen3.8-27B/knowledge-and-reasoning/mmlu-pro`): `fewshot=0` (the package
default): every question is asked zero-shot with the package's chain-of-thought multiple-choice prompt (`ANSWER:
$LETTER`). Dataset pinned at revision `527feea0af`. Sampling, reasoning effort, token budget and Inspect environment
are the base model's. It runs one epoch, as for the base model; the hardware and serving stack differ (AMD MI355X,
vLLM 0.31.0 for ROCm, instead of NVIDIA H100, vLLM 0.29.0). The checkpoint also declares a static per-tensor FP8
KV-cache scheme (calibrated `k_scale`/`v_scale`), which vLLM applies by default; `serve.sh` overrides it with
`--kv-cache-dtype bfloat16`, so the KV cache is BF16 as for the base model and for `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`,
and only the MXFP4 quantization differs between the two checkpoints. MXFP4 GEMMs run natively on gfx950 (vLLM's
`AiterMxfp4LinearKernel`); attention runs on vLLM's Triton backend (`--attention-backend TRITON_ATTN`) instead of its
ROCm default, which decodes about 14x slower at long context. The model card does not report MMLU-Pro.

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
access to `/dev/kfd` and `/dev/dri/renderD*`) and the Inspect client from a virtualenv; create them as below, and edit
the image and paths in `local_orchestrator.sh` if you keep the virtualenv elsewhere or run vLLM differently (e.g. from
vLLM's ROCm wheels on a host with glibc >= 2.39). `server_requirements.txt` is the `pip freeze` of that image;
`client_requirements.txt` is the exact `pip freeze` of the client environment that produced the results below, the
same packages as the base model's client environment.

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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=131072, max_connections=256, client_timeout=7200, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy | 0.8440 |
| stderr | 0.0033 |
| samples | 12,032 (12,032 questions x 1 epoch) |
| truncated (stop_reason=max_tokens) | 0 |
| unparsed answers | 7 (generation ended inside the reasoning, empty final answer) |
| sample errors | 0 |
| requests retried after the 7200 s client timeout | 7 (questions 1782, 9745, 9774, 9847, 11559, 11794, 11997: regenerated from scratch; 2 scored correct) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 308 | 5,048 | 4,890 |
| median | 275 | 890 | 702 |
| max | 1,767 | 123,288 | 122,920 |

Per subject: math 0.944, biology 0.923, chemistry 0.920, physics 0.909, business 0.891, economics 0.883, computer science 0.873, psychology 0.830, engineering 0.827, health 0.784, philosophy 0.774, other 0.752, history 0.743, law 0.655.

Total run time about 4 h 24 min on 3x MI355X (TP=1, DP=3; 256 concurrent samples).

## Patches for upstream

None.
