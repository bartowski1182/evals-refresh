# zai-org/GLM-5.3-Flash — AIME 2026

| | |
|---|---|
| Task | `inspect_evals/aime2026` (30 problems (AIME I + II), integer answers) |
| Category | knowledge-and-reasoning (subclass: math-reasoning) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash (default variant: the native FP8 checkpoint) |

Answers are matched numerically after stripping `\boxed{}`; the prompt asks for `ANSWER: $ANSWER` on the last line. Dataset pinned at revision `79037aebdb` (inside `inspect_evals`). Eight epochs, reported as avg@8 (the mean over the 8 × 30 answers), at the user's request: with only 30 problems one epoch carries a standard error of about 9 points at 50% accuracy. This dir is one of the two identical copies of the task that make up the GLM-5.3-Flash pair
(`zai-org_GLM-5.3-Flash`, the native FP8 model, and `RedHatAI_GLM-5.3-Flash-MXFP4xFP8`): same task, prompt, sampling,
limits, concurrency, epochs, vLLM release and serving topology (8x H200, one TP=8 replica, BF16 KV cache, MTP), so the
two numbers are like for like; only the checkpoint differs. The GLM-5.3-Flash card shows its scores in a chart and
states no protocol for this benchmark; the number below is not comparable to it.

**Checkpoint.** `zai-org/GLM-5.3-Flash` is the vendor's native FP8 release (320B total / 18B active parameters,
hybrid KDA linear attention and NoPE sparse MLA, 288 experts with 8 routed per token, one MTP layer; e4m3 weights with
dynamic activation scaling, attention projections, embeddings, lm_head and the mHC/KDA parameters unquantized),
revision `46a050bec1` (2026-10-15). Its `generation_config.json` sets temperature 1.0 and top_p 0.95 and no top_k.
It is the reference for the quantized copy `RedHatAI/GLM-5.3-Flash-MXFP4xFP8` (the other dir of the pair). The KV
cache is BF16 on Hopper for both models of the pair (see `serve.sh`).

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

Check `logs/serve.log` at start-up: the quantization method must be `fp8`, the KV cache dtype must be `bfloat16`,
and MTP must be active (`num_speculative_tokens=5`).

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

`temperature=1.0, top_p=0.95, top_k unset, reasoning_effort=max, max_tokens=163840, max_connections=128, client_timeout=7200, kv_cache_dtype=bfloat16, TP=8 DP=1 with MTP (num_speculative_tokens=5), max_model_len=262144, epochs=8 (avg@8)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None.
