# RedHatAI/Qwen3.8-27B-MXFP4 — BFCL

| | |
|---|---|
| Task | `inspect_evals/bfcl` (4981 samples: all V1, V2 and V3 categories of the Berkeley Function-Calling Leaderboard; gorilla repo commit pinned by inspect_evals 0.20.0) |
| Category | general-tool-usage (subclass: tool-use) |
| Exec tier | `agent` — no sandbox; native tool calling, multi-turn categories run stateful Python backends in-process |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant serves this checkpoint with the same flags) |
| Task source | https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard |

**Protocol note.** No model card reports BFCL. The official leaderboard's overall score (BFCL v4) also covers the
agentic V4 categories (web search, which needs a SerpAPI key, and memory, which needs a two-step snapshot workflow)
and the retired `rest` category; this run covers the V1 single-turn (AST-matched), V2 live (user-contributed) and V3
multi-turn (stateful backends) categories only, so per-category accuracies are comparable to the leaderboard's and the
overall number is not. Tools are passed natively (FC mode) and parsed by vLLM's tool parser; the model reasons before
every call. One epoch (pass@1); one TP=8 replica so every turn of a sample hits one prefix cache. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs.

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
  (epochs, sampling params, per-turn max tokens, time limit, concurrency); identical in the three dirs apart from `MODEL`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` inside the vLLM image and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM podman image (GPUs 0-7, port 8000, one TP=8 replica, the HF cache and the task dir mounted), waits for the server, activates the client venv, runs `run_eval.sh`, stops the container.
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
# client: Inspect + inspect_evals with the bfcl extra, from the fork branch that carries the multi-turn scorer fix (see Patches)
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install "inspect-evals[bfcl] @ git+https://github.com/eldarkurtic/inspect_evals@fix/bfcl-multi-turn-scorer-short-trajectories" "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the log must name the MXFP4
kernel (`Using <Kernel> for MXFP4 GEMM`), the KV cache dtype must be `bfloat16` (the checkpoint's FP8 KV scheme is
overridden), and the default sampling params vLLM takes from the checkpoint's `generation_config.json` must be
temperature 1.0, top_p 0.95, top_k 20.

No Docker needed. The first run clones the gorilla repository at the pinned commit for the data and the multi-turn backends.
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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), time_limit=7200s, client_timeout=7200, kv_cache_dtype=bfloat16, TP=8 DP=1, epochs=1, 64 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

`inspect_evals/bfcl/score/multi_turn_scorer.py` raises `ValueError: zip() argument 2 is shorter than argument 1`
instead of scoring 0 when a multi-turn sample ends (time or message limit) with fewer turns than the ground truth
has, so such samples become sample errors. Reported as
https://github.com/UKGovernmentBEIS/inspect_evals/issues/2544; the fix (missing turns count as empty, task version
8-B) is branch `fix/bfcl-multi-turn-scorer-short-trajectories` of https://github.com/eldarkurtic/inspect_evals, which
the client venv installs from (see Setup).
