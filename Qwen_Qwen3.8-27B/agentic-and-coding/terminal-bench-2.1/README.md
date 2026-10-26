# Qwen/Qwen3.8-27B — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task (Docker Hub), bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The base model card reports "Terminal Bench 2.1 (Terminus)", i.e. with the Terminus 2 agent used by
the official leaderboard. `inspect_harbor` solves every task with Inspect's generic ReAct agent (`bash`, `python`,
`update_plan`, `submit`; 300 s tool timeout; automatic context compaction), not Terminus 2. Harbor's per-task agent
timeouts (600 s to 12000 s, most 900 s) are not applied; a single `--time-limit 7200` is used for every task.
Containers run without Harbor's per-task CPU cap (`override_cpus=0`) and with `inspect_harbor`'s 6 GB memory floor;
the tasks have public network by design. One epoch (pass@1), 89 samples in flight (all tasks at once). The wall-clock
limit binds harder against a shared server than against an API endpoint, so the number is not comparable to the
leaderboard. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs.

**Checkpoint.** `Qwen/Qwen3.8-27B` is the BF16 base model of the comparison set, the reference the two RedHatAI
quantized checkpoints are measured against. The KV cache is BF16 (`--kv-cache-dtype bfloat16`, this checkpoint's
default, spelled out so that the three `serve.sh` are identical), and vLLM 0.32.0 is used for the three dirs (see
Setup). Record the checkpoint revision here before the run.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the Qwen/Qwen3.8-27B recipe, plus the BF16 KV cache; identical in the three dirs apart from
  `MODEL`.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and package digest, epochs, sampling params, reasoning effort, per-turn max tokens, concurrency, sandbox count, time limit); identical in the three dirs apart from `MODEL`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM server environment and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` from the vLLM podman image (`VLLM_IMAGE`, by default the pinned one) or a venv (`VLLM_VENV`) through `tools/vllm_server.sh` (GPUs 0-7, port 8000, TP=2 x DP=4 from `serve.sh`, the HF cache and the task dir mounted in podman mode), points Docker at the rootless daemon, waits for the server, activates the client venv (`INSPECT_VENV`), runs `run_eval.sh`, stops the server on exit.
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
# client: Inspect + inspect_harbor
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install "inspect-harbor==1.0.0" httpx2 && pip install "openai==3.14.0"
```

Check `logs/serve.log` at start-up: `max_model_len` 262144, KV cache dtype `bfloat16`, and the default sampling
params vLLM takes from the checkpoint's `generation_config.json` (temperature 1.0, top_p 0.95, top_k 20).

Docker with the compose plugin (v2.x or later) is required for the sandboxes (podman serves the model only).
`local_orchestrator.sh` exports `DOCKER_HOST` for a rootless daemon (`unix:///run/user/<uid>/docker.sock`); with a
rootful daemon drop that export. Rootless Docker needs
a runc runtime with `NoNewKeyring` in `daemon.json` (one kernel key per container otherwise exhausts a user's quota of
200) and cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`. Check the image-store disk space before
the first run (see `local_env_notes.md`).

**What to expect on the first run:** Harbor downloads the 89 task definitions (59 MB), then Inspect pulls one prebuilt
image per task from Docker Hub as containers start (21 GB compressed, about 70 GB on disk, nothing is built). Docker
Hub's anonymous limit is 100 pulls per hour per IP, so pre-pull with `docker pull` if needed. The task files, including
`solution/solve.sh` and `tests/`, are public, so audit every run for agents that download their own task's files
(AGENTS.md, section 11).
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache (`HF_HOME`,
mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or set
`MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the repo); the Harbor task files come from the hub package at the pinned digest.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5) after the leak audit of section 11, with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=89, max_sandboxes=89, time_limit=7200s, client_timeout=7200, override_cpus=0, kv_cache_dtype=bfloat16, TP=2 DP=4, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None. Known tooling issues, documented for the run: Inspect's `vllm` provider learns the served context window
(262,144) from `/v1/models` on a sample's first model call, but the ReAct agent's compaction threshold (90% of the
window) is resolved at sample start, so samples that start before the first call fall back to Inspect's 128,000
default; trajectories whose own messages exceed the threshold after compaction end with `RuntimeError: Compaction
insufficient` and are scored as failures.
