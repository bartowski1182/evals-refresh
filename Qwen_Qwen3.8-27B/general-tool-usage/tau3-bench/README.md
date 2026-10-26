# Qwen/Qwen3.8-27B — tau3-bench

| | |
|---|---|
| Task | `inspect_harbor/sierra_research_tau3_bench` (375 tasks: airline 50, retail 114, telecom 114, banking_knowledge 97; Harbor hub package digest `sha256:a57304f6…`; the tasks are generated from the adapter fork that carries the image fix, see Patches) |
| Category | general-tool-usage (subclass: tool-use) |
| Exec tier | `agent` — two Docker containers per sample (task image plus the tau2-bench MCP runtime sidecar), domain tools and the user conversation through MCP, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://hub.harborframework.com/datasets/sierra-research/tau3-bench |

**Protocol note.** No model card reports tau3-bench. The task set is tau2-bench >= 1.0 with the banking_knowledge
domain (the voice modality of tau3 is not part of the package). Inspect's generic ReAct agent (`bash`, `python`,
`update_plan`, plus the task's MCP tools: `start_conversation`, `send_message_to_user`, `end_conversation` and the
domain tools) drives the conversation; the user simulator and the natural-language assertion judge (91 tasks) are
LLM calls made by the runtime and the verifier through an OpenAI-compatible endpoint, both Qwen/Qwen3.8-27B here (a
fixed simulator as on the tau-bench leaderboards; the leaderboard uses gpt-5.2 at low reasoning effort). A second
vLLM server (GPUs 4-7, TP=4) serves Qwen/Qwen3.8-27B as the simulator and judge while the model under test runs on
GPUs 0-3 (TP=4), in all three dirs (the Gemma pattern), so they are served identically for the three models. One epoch
(pass^1); a uniform `--time-limit 3600` (Harbor's agent budget) as a runaway guard. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs. The numbers are not
comparable to taubench.com; the native Inspect re-implementation of the same task set is in `../tau2-bench`.

**Checkpoint.** `Qwen/Qwen3.8-27B` is the BF16 base model of the comparison set, the reference the two RedHatAI
quantized checkpoints are measured against. The KV cache is BF16 (`--kv-cache-dtype bfloat16`, this checkpoint's
default, spelled out so that the three `serve.sh` are identical), and vLLM 0.32.0 is used for the three dirs (see
Setup). Record the checkpoint revision here before the run.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the Qwen/Qwen3.8-27B recipe, plus the BF16 KV cache; identical in the three dirs apart from
  `MODEL`.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (dataset copy, epochs, sampling params, per-turn max tokens, concurrency, sandbox count, time limit, simulator/judge endpoint variables); identical in the three dirs apart from `MODEL`.
- `assets/serve_user_model.sh` — vLLM server for the Qwen/Qwen3.8-27B user simulator on a second port (recipe flags),
  started by `local_orchestrator.sh`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM server environment and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` (GPUs 0-3) and `assets/serve_user_model.sh` (GPUs 4-7, published on all interfaces for the task containers) from the vLLM podman image (`VLLM_IMAGE`, by default the pinned one) or a venv (`VLLM_VENV`) through `tools/vllm_server.sh`, one TP=4 replica each, with the HF cache and the task dir mounted in podman mode, points Docker at the rootless daemon, exports the host address containers can reach, waits for both, activates the client venv (`INSPECT_VENV`), runs `run_eval.sh`, stops the servers on exit.
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
# client: Inspect + inspect_harbor with MCP-server support from the fork branch of PR #186 (see Patches)
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install "inspect-harbor @ git+https://github.com/eldarkurtic/inspect_harbor@feat/mcp-servers-and-compose-sidecars" mcp httpx2
pip install "openai==3.14.0"   # last: overrides litellm's openai<3 pin (harmless, Inspect's agent path never imports litellm)
```

Check `logs/serve.log` at start-up: `max_model_len` 262144, KV cache dtype `bfloat16`, and the default sampling
params vLLM takes from the checkpoint's `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20).

Docker with the compose plugin (v2.x or later) is required for the sandboxes (podman serves the model only).
`local_orchestrator.sh` exports `DOCKER_HOST` for a rootless daemon (`unix:///run/user/<uid>/docker.sock`); with a
rootful daemon drop that export. Rootless Docker needs
a runc runtime with `NoNewKeyring` in `daemon.json` (one kernel key per container otherwise exhausts a user's quota of
200) and cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`. Check the image-store disk space before
the first run (see `local_env_notes.md`).
Every sample is a two-service compose project with its own network, so the daemon's `default-address-pools` must hold
at least `--max-sandboxes` networks (Docker's default runs out near 40). Each task builds two images (about 1.9 GB
each, layers shared across tasks). The user-simulator server needs the Qwen/Qwen3.8-27B weights (BF16) in the same
cache.

The task set is generated from the adapter fork that carries the image fix of harbor#3401 (see Patches), with
tau2-bench checked out at the commit the images pin; `run_eval.sh` reads it from `DATASET_PATH`:

```bash
git clone -b fix/tau3-bench-image-deps https://github.com/eldarkurtic/harbor.git ~/github/eldarkurtic/harbor
git clone https://github.com/sierra-research/tau2-bench.git ~/github/eldarkurtic/tau2-bench   # next to the harbor clone
git -C ~/github/eldarkurtic/tau2-bench checkout b7ea9074c1cba482b30687fecdb5c8425fd6f619
cd ~/github/eldarkurtic/harbor/adapters/tau3-bench && uv run tau3-bench --output-dir ~/.cache/harbor/local/tau3-bench-cc9e4b7c --overwrite
```
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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), time_limit=3600s, client_timeout=7200, tool_timeout=300s, override_cpus=0, kv_cache_dtype=bfloat16, TP=4 DP=1 on GPUs 0-3, epochs=1; user simulator and assertion judge openai/Qwen/Qwen3.8-27B on a second server (GPUs 4-7, TP=4; simulator at reasoning effort low, as the package defaults); 56 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

Two upstream defects had to be fixed to run this package faithfully:

- `inspect_harbor` (up to 1.0.0) ignores `[[environment.mcp_servers]]` and leaves the main service of a task's
  docker-compose.yaml incomplete, so the agent gets no domain tools and the sidecar's `/logs/agent` is not shared
  with the verifier. Fix: https://github.com/meridianlabs-ai/inspect_harbor/pull/186 (branch
  `feat/mcp-servers-and-compose-sidecars` of https://github.com/eldarkurtic/inspect_harbor), which wires each MCP
  server as Inspect MCP tools through a stdio bridge started inside the service that hosts it, completes the main
  service from the task's Dockerfile and `[environment.env]`, marks it as Inspect's default sandbox, and turns
  Harbor's log-directory bind mounts into named volumes shared by all services. The client venv installs that branch
  (see Setup); once released, pin the release.
- The task images clone tau2-bench's moving `main` at build time and install it without `websockets`, which
  `import tau2` now needs, so the verifier and the reference solution crash in the main container. Fix:
  https://github.com/harbor-framework/harbor/pull/3401 (branch `fix/tau3-bench-image-deps` of
  https://github.com/eldarkurtic/harbor): the adapter's Dockerfile templates pin the tau2-bench checkout
  (`TAU2_BENCH_REF`, commit `b7ea9074`) and add `websockets`. The task set is generated from that branch with the
  adapter (see Setup); once merged and the hub package republished, switch back to
  `inspect_harbor/sierra_research_tau3_bench -T ref=<digest>`.

Task content, tools and verifiers are unchanged by either fix.
