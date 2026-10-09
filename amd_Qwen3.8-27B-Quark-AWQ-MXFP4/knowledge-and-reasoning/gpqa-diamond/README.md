# amd/Qwen3.8-27B-Quark-AWQ-MXFP4 — GPQA-Diamond

| | |
|---|---|
| Task | `inspect_evals/gpqa_diamond` (198 questions, 4 choices) |
| Category | knowledge-and-reasoning (subclass: science-reasoning) |
| Exec tier | `none` — no Docker sandbox, no tools (the vLLM server itself runs in a container) |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; it has no MXFP4 variant) |

This is Qwen/Qwen3.8-27B with the language model's linear layers quantized to MXFP4 weights and dynamic MXFP4
activations (W4A4, AWQ, AMD Quark format; `lm_head`, the MTP head and the vision tower stay BF16), checkpoint revision
`5233554c5fa5`. Task, prompt and generation settings are those of the base model's run
(`../../../Qwen_Qwen3.8-27B/knowledge-and-reasoning/gpqa-diamond`): `cot=True` (the package's chain-of-thought
multiple-choice prompt, `ANSWER: $LETTER`), choices shuffled with the package's fixed seed (`shuffle_choices=42`), the
simple-evals `gpqa_diamond.csv` verified by SHA256 inside `inspect_evals`, and the same sampling, reasoning effort,
token budget and Inspect environment. It runs avg@3 as the base model does (here in two runs, 1 + 2 epochs); the
hardware and serving stack differ (AMD MI355X, vLLM 0.31.0 for ROCm, instead of NVIDIA H100, vLLM 0.29.0). The model
card's serve command adds `--trust-remote-code` and `--max-model-len 16384`; neither is used here (the checkpoint has
no remote code, and the base model's 262,144 context is kept). The KV cache is BF16, as for the base model and for
`RedHatAI/Qwen3.8-27B-MXFP4` (whose FP8 KV-cache scheme is overridden), so only the MXFP4 quantization differs between
the two checkpoints. MXFP4 GEMMs run natively on gfx950 (vLLM's `AiterMxfp4LinearKernel`); attention runs on vLLM's
Triton backend (`--attention-backend TRITON_ATTN`) instead of its ROCm default, which decodes about 14x slower at long
context. The model card reports only GSM8K, so there is no card number to compare with.

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
HF_HUB_CACHE=~/hf_hub_cache hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --revision 5233554c5fa56afda40150556b95573c2d7d29c0
```

Export `HF_TOKEN` in your shell before running (never in the repo): the dataset `Idavidrein/gpqa` is gated, accept
its terms at https://huggingface.co/datasets/Idavidrein/gpqa with the account the token belongs to. The server runs
with `HF_HUB_OFFLINE=1` and needs the weights downloaded as above.

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
| accuracy (avg@3) | 0.8653 |
| stderr | 0.0217 |
| per-epoch accuracy | 0.874 / 0.864 / 0.859 |
| samples | 594 (198 questions x 3 epochs, in 2 runs: 198 + 396) |
| truncated (stop_reason=max_tokens) | 0 |
| unparsed answers | 0 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 316 | 17,445 | 17,153 |
| median | 282 | 9,049 | 8,648 |
| max | 2,844 | 129,345 | 129,076 |

Run time: about 38 min for the 1-epoch run on 1x MI355X and 1 h 14 min for the 2-epoch run on 1x MI355X (TP=1, one replica per GPU; 128 concurrent samples).

## Patches for upstream

None.
