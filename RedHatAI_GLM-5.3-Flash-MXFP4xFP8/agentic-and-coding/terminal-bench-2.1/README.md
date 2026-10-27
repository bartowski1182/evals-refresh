# RedHatAI/GLM-5.3-Flash-MXFP4xFP8 — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task (Docker Hub), bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash (the base model's; the recipes site has no MXFP4xFP8 variant, and its NVFP4 variant is Blackwell-only) |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The GLM-5.3-Flash card reports Terminal-Bench 2.1 = 84.3 with a Claude Code 2.1.207 scaffold
(temperature 1.0, top_p 1, max_new_tokens 65536, 6 h timeout). `inspect_harbor` instead solves every task with
Inspect's generic ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context
compaction), so the number below is **not comparable** to that card figure or to the Terminus 2 leaderboard.
Harbor's per-task agent timeouts (600 s to 12000 s, most 900 s) are not applied; a single `--time-limit 7200` is
used for every task. Containers run without Harbor's per-task CPU cap (`override_cpus=0`) and with
`inspect_harbor`'s 6 GB memory floor; the tasks have public network by design. One epoch (pass@1), 89 samples in
flight (all tasks at once). The wall-clock limit binds harder against a shared server than against an API endpoint.
This dir is one of the two identical copies of the task that make up the GLM-5.3-Flash pair (`zai-org_GLM-5.3-Flash`,
the native FP8 model, and `RedHatAI_GLM-5.3-Flash-MXFP4xFP8`): same task, agent, prompt, sampling, limits,
concurrency, epochs (one, pass@1), vLLM release and serving topology (8x H200, one TP=8 replica, BF16 KV cache), so
the two numbers are like for like; only the checkpoint differs.

**Checkpoint.** `RedHatAI/GLM-5.3-Flash-MXFP4xFP8` is zai-org/GLM-5.3-Flash (native FP8) with the MoE experts'
weights quantized to MXFP4 and their input activations to FP8 (W4A8, compressed-tensors format, LLM Compressor). The
repository was not public when this dir was written (2026-10-27; it was being uploaded under
`inference-optimization/GLM-5.3-Flash-MXFP4xFP8`, to be moved to `RedHatAI`): before running, check that its
`config.json` declares MXFP4 weights with dynamic FP8 input activations (and which layers it covers: experts only, or
dense linear layers too), that its `generation_config.json` matches the base model's (temperature 1.0, top_p 0.95,
no top_k), and record the revision here. The KV cache is BF16 on Hopper for both models of the pair (see `serve.sh`).

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  BF16 KV cache, MTP speculative decoding, reasoning/tool parsers, text-only mode, default reasoning effort), taken
  from the base model's recipe (Hopper, Tensor Parallel strategy), with two documented departures (`--max-model-len
  262144`, `--language-model-only`); identical in the two dirs of the pair apart from `MODEL` and the checkpoint
  comments.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task and dataset pin,
  epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency, CPU override, error
  threshold); identical in the two dirs apart from `MODEL`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM server environment and of the
  Inspect venv that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the
  final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` from the vLLM podman
  image (`VLLM_IMAGE`, by default the pinned one) or a venv (`VLLM_VENV`) through `tools/vllm_server.sh` (GPUs 0-7,
  port 8000, one TP=8 replica, the HF cache and the task dir mounted in podman mode), points Docker at the rootless
  daemon, waits for the server, activates the client venv (`INSPECT_VENV`), runs `run_eval.sh`, stops the server on
  exit. **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Pull the server image and create the client environment as below. The commands use the default server image
(`DEFAULT_VLLM_IMAGE`) and client venv path (`INSPECT_VENV`); if you keep them elsewhere, set those variables in the
environment or change them in `local_orchestrator.sh`. Once the results are in, `server_requirements.txt`
(`tools/vllm_server.sh` provides `vllm_freeze`, which prints `pip freeze` of the selected environment) and
`client_requirements.txt` hold the exact packages that produced them (host details in their headers); create the
client environment from its file instead. Use the same server and client environments for the two dirs of the pair.

```bash
# server: the vLLM 0.32.0 podman image, docker.io/vllm/vllm-openai:v0.32.0 pinned by digest (the DEFAULT_VLLM_IMAGE),
# the same image for the two dirs. 0.32.0 is the first release whose model registry carries glm5_next (the recipe,
# written before it, asks for a nightly or the vllm/vllm-openai:glm53-flash image) and the first with the
# compressed-tensors MXFP4-weight / FP8-activation MoE path (vllm-project/vllm#59815) and the dense Humming MXFP4
# kernel that this checkpoint may need (CUDA, SM75+; the H200 qualifies). FlashInfer 0.6.17 or newer is required for
# the NoPE sparse-MLA kernels (the image ships 0.7.x). GPU access from the container goes through the NVIDIA
# Container Toolkit's CDI spec (generated once per host, as root).
podman pull docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# To serve from a venv instead, create one with the vLLM version server_requirements.txt records (0.32.0) and pass it
# as VLLM_VENV; the image is the reference environment. If the release cannot load the checkpoint, point VLLM_IMAGE
# at a vLLM nightly image instead and record it.
# client: Inspect + inspect_harbor (stock 1.0.0; the tasks have public network by design, so the network-mode fork
# of the DeepSWE dir changes nothing here)
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install "inspect-ai==0.3.273" "inspect-harbor==1.0.0" httpx2 "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the expert layers must load
with `CompressedTensorsW4A8Mxfp4MoEMethod` (MXFP4 weights, FP8 activations; a vLLM without vllm-project/vllm#59815
loads the checkpoint without error and runs the experts with BF16 activations), the KV cache dtype must be
`bfloat16`, and MTP must be active (`num_speculative_tokens=5`). If the checkpoint also quantizes dense linear
layers to MXFP4, the log must say `Using HummingMxFp4LinearKernel for MXFP4 GEMM` and not warn `No MXFP4 linear
kernel supports ... falling back to MXFP4 activations` (that fallback runs those layers as W4A4, which is not the
checkpoint this dir describes). Read the `GPU KV cache size` line too: the concurrency in `run_eval.sh` puts all 89
tasks in flight at once.

Docker with the compose plugin (v2.x or later) is required for the sandboxes (podman serves the model only).
`local_orchestrator.sh` exports `DOCKER_HOST` for a rootless daemon (`unix:///run/user/<uid>/docker.sock`); with a
rootful daemon drop that export. Rootless Docker cannot apply CPU limits, hence `-T override_cpus=0` in
`run_eval.sh`. Inspect pulls a task's prebuilt image before its first sample if it is missing (serially, before any
sample starts; about 15 min for the 89 images, against Docker Hub's anonymous pull limit, see `local_env_notes.md`).
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache
(`HF_HOME`, mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or
set `MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the
repo); the Harbor task files are fetched from GitHub at the pinned commit.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5) after the leak audit of section 11 (the tasks are networked: scan every trajectory for downloads of the
benchmark's own `solution/` or `tests/` files), with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=0.95, top_k unset, reasoning_effort=max, max_tokens=65536 (per turn), max_connections=89, max_sandboxes=89, time_limit=7200s, client_timeout=7200, tool_timeout=300s, override_cpus=0, kv_cache_dtype=bfloat16, TP=8 DP=1 with MTP (num_speculative_tokens=5), max_model_len=262144, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None (dataset/task). Known tooling behavior, documented for the run: Inspect's `vllm` provider does not learn the
served context window (262,144) in time for the ReAct agent's compaction threshold, which resolves at sample start,
so samples fall back to Inspect's 128,000 default context window for compaction decisions. This is identical across
models evaluated this way (the served model name is not in Inspect's registry) and does not affect comparability
between them; it only makes compaction more conservative than the served 262k window.
