# RedHatAI/Qwen3.8-27B-MXFP4 — AIME 2026

| | |
|---|---|
| Task | `inspect_evals/aime2026` (30 problems (AIME I + II), integer answers) |
| Category | knowledge-and-reasoning (subclass: math-reasoning) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant serves this checkpoint with the same flags) |

Answers are matched numerically after stripping `\boxed{}`; the prompt asks for `ANSWER: $ANSWER` on the last line.
Dataset pinned at revision `79037aebdb`. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs. With only 30 problems one epoch carries a standard error of about
9 points at 50% accuracy, so differences between the three models below that are noise; more epochs (`EPOCHS=N`,
avg@N) only if the user asks, and then for all three. The model cards' AIME numbers are measured with other harnesses,
not with this task.

**Checkpoint.** `RedHatAI/Qwen3.8-27B-MXFP4` is Qwen/Qwen3.8-27B with the linear layers' weights and input
activations quantized to MXFP4 (W4A4, compressed-tensors format, group size 32), the MXFP4-activation counterpart of
`RedHatAI/Qwen3.8-27B-MXFP4xFP8` (W4A8). The checkpoint also declares an FP8 KV-cache scheme, which vLLM would apply
under its default `--kv-cache-dtype auto`; `serve.sh` overrides it with `bfloat16`, so the KV cache is BF16 as for the
other two models and the three differ only in the linear layers' quantization. vLLM 0.32.0 is used for the three
dirs (see Setup). Record the checkpoint revision and the MXFP4 kernel vLLM chose (`logs/serve.log`) here before the
run.

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

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the log must name the MXFP4
kernel (`Using <Kernel> for MXFP4 GEMM`), the KV cache dtype must be `bfloat16` (the checkpoint's FP8 KV scheme is
overridden), and the default sampling params vLLM takes from the checkpoint's `generation_config.json` must be
temperature 1.0, top_p 0.95, top_k 20.

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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=131072, max_connections=128, client_timeout=7200, kv_cache_dtype=bfloat16, TP=2 DP=4, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None.
