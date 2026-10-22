# RedHatAI/Qwen3.8-27B-MXFP4xFP8 — tau2-bench (native Inspect implementation of the tau3-bench task set)

| | |
|---|---|
| Task | `inspect_evals/tau2_airline`, `tau2_retail`, `tau2_telecom`, `tau2_banking` (375 tasks: the "base" splits of tau2-bench >= 1.0 plus banking_knowledge, the same task set as the Harbor `sierra-research/tau3-bench` package) |
| Category | general-tool-usage (subclass: tool-use) |
| Exec tier | `agent` — no sandbox; domain tools are in-process Python, a second model role simulates the customer |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant covers the W4A4 checkpoint, there is no MXFP4xFP8 variant) |
| Task source | https://github.com/sierra-research/tau2-bench (data vendored by inspect_evals 0.20.0) |

**Protocol note.** No model card reports tau2/tau3-bench. This is inspect_evals's native re-implementation of
tau2-bench: the agent gets the domain policy as system prompt and the domain tools as Python tools, a user
simulator (Inspect model role `user`, temperature 0 as upstream; Qwen/Qwen3.8-27B for every model of this repo,
a fixed simulator as on the tau-bench leaderboards) plays the customer from the task's scenario, and a task
passes when the final database state matches the expected one and the required information was communicated
(the airline natural-language assertions of upstream are not graded by this implementation; banking uses the
`grep` knowledge-retrieval configuration rather than the leaderboard's `alltools`). The agent samples with the
model's own generation config instead of upstream's temperature 0. Up to 100 user/agent exchanges per task. One
epoch (pass^1). A second vLLM server (GPUs 4-7, TP=4) serves Qwen/Qwen3.8-27B as the user simulator while the model
under test runs on GPUs 0-3 (TP=4), in all three dirs (the Gemma pattern), so the simulator is served identically
for the three models and a quantized model never plays its own customer. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs. The numbers are not comparable to
taubench.com; the Harbor-package run in `../tau3-bench` is the same task set through the original tau2-bench runtime.

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
  (tasks, user-simulator role, epochs, sampling params, per-turn max tokens, time limit, concurrency); identical in the three dirs apart from `MODEL`.
- `assets/serve_user_model.sh` — vLLM server for the Qwen/Qwen3.8-27B user simulator on a second port (recipe flags),
  started by `local_orchestrator.sh`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` inside the vLLM image and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` (GPUs 0-3) and `assets/serve_user_model.sh` (GPUs 4-7) in two containers of the vLLM podman image, one TP=4 replica each, with the HF cache and the task dir mounted, waits for both, activates the client venv, runs `run_eval.sh`, stops the containers.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Pull the server image and create the client environment as below. The commands use the image reference and the venv
path that `local_orchestrator.sh` uses; if you keep them elsewhere, change them there. Once the results are in,
`server_requirements.txt` (`pip freeze` inside the image) and `client_requirements.txt` hold the exact packages that
produced them (host details in their headers); create the client environment from its file instead. Use the same image
and client environments for the three dirs of the comparison set.

```bash
# server: the vLLM 0.32.0 podman image, docker.io/vllm/vllm-openai:v0.32.0 pinned by digest (IMAGE in
# local_orchestrator.sh), the same image for the three dirs. 0.32.0 is the first release with the dense MXFP4-weight /
# FP8-activation scheme the MXFP4xFP8 checkpoint needs; the humming-kernels dependency of its CUDA wheels provides the
# only MXFP4 GEMM of that release that takes FP8 activations (CUDA, SM75+), so the comparison set needs an NVIDIA host.
# GPU access from the container goes through the NVIDIA Container Toolkit's CDI spec (generated once per host, as root).
podman pull docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# client: Inspect + inspect_evals
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install "inspect-ai==0.3.263" "inspect-evals==0.20.0" "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the log must say
`Using HummingMxFp4LinearKernel for MXFP4 GEMM`, and the KV cache dtype must be `bfloat16`. A warning `No MXFP4 linear
kernel supports ... falling back to MXFP4 activations` (no Humming, or a ROCm host, whose AITER kernel is W4A4 only)
means the checkpoint runs as W4A4, i.e. as `RedHatAI/Qwen3.8-27B-MXFP4`, and the run is not the one this dir
describes.

No Docker needed. The user-simulator server needs the Qwen/Qwen3.8-27B weights (BF16, about 55 GB) in the same cache.
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache (`HF_HOME`,
mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or set
`MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5), with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), time_limit=3600s, client_timeout=7200, up to 100 user/agent exchanges (task default), kv_cache_dtype=bfloat16, TP=4 DP=1 on GPUs 0-3, epochs=1; user simulator vllm/Qwen/Qwen3.8-27B at temperature 0 on a second server (GPUs 4-7, TP=4); 56 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None.
