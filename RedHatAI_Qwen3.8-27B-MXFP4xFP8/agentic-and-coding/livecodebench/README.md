# RedHatAI/Qwen3.8-27B-MXFP4xFP8 — LiveCodeBench (Harbor, 100-task sample)

| | |
|---|---|
| Task | `inspect_harbor/livecodebench` (100 tasks sampled from LiveCodeBench release_v6, loaded through `assets/harbor_registry.json`, see Patches) |
| Category | agentic-and-coding (subclass: code-gen) |
| Exec tier | `agent` — Docker container per task, bash/python tools, hidden-test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant covers the W4A4 checkpoint, there is no MXFP4xFP8 variant) |
| Task source | https://hub.harborframework.com/datasets/livecodebench/livecodebench |

**Protocol note.** The base model card's "LiveCodeBench v6" is single-shot pass@1 over the full release_v6 set
(1,055 problems). This task is a fixed 100-problem sample solved agentically (the model can run public tests and
iterate) with Inspect's generic ReAct agent, per-turn max_tokens 65536, `--time-limit 3600`, `override_cpus=0`, one
epoch (pass@1), so the two numbers are not comparable. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs. The tasks set no network policy, so the agent has
internet: audit the trajectories for lookups of published solutions (AGENTS.md, section 11).

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
  (task and dataset pin, epochs, sampling params, reasoning effort, per-turn max tokens, concurrency, sandbox count, time limit); identical in the three dirs apart from `MODEL`.
- `assets/harbor_registry.json` — Harbor registry entry for `livecodebench@6.0` pinned to a commit that carries our
  upstream fixes (see Patches); `run_eval.sh` loads the dataset from it.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` inside the vLLM image and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM podman image (GPUs 0-7, port 8000, TP=2 x DP=4 from `serve.sh`, the HF cache and the task dir mounted), points Docker at the rootless daemon, waits for the server, activates the client venv, runs `run_eval.sh`, stops the container.
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
# client: Inspect + inspect_harbor
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install "inspect-harbor==1.0.0" httpx2 && pip install "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the log must say
`Using HummingMxFp4LinearKernel for MXFP4 GEMM`, and the KV cache dtype must be `bfloat16`. A warning `No MXFP4 linear
kernel supports ... falling back to MXFP4 activations` (no Humming, or a ROCm host, whose AITER kernel is W4A4 only)
means the checkpoint runs as W4A4, i.e. as `RedHatAI/Qwen3.8-27B-MXFP4`, and the run is not the one this dir
describes.

Docker with the compose plugin (v2.x or later) is required for the sandboxes (podman serves the model only).
`local_orchestrator.sh` exports `DOCKER_HOST` for a rootless daemon (`unix:///run/user/<uid>/docker.sock`); with a
rootful daemon drop that export. Rootless Docker needs
a runc runtime with `NoNewKeyring` in `daemon.json` (one kernel key per container otherwise exhausts a user's quota of
200) and cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`. Check the image-store disk space before
the first run (see `local_env_notes.md`).

**What to expect on the first run:** Harbor fetches the 100 task files from GitHub at the pinned commit, then Inspect
builds one image per task, one after another, before any sample starts (with the pinned Dockerfile fix the apt layer
is shared: about 1 minute and 2 GB in total). Later runs reuse the images; pass `--sandbox-prebuilt` to skip the
check. During image builds `inspect ctl` reports no running eval; watch the eval output instead.
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache (`HF_HOME`,
mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or set
`MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the repo); the Harbor task files are fetched from GitHub at the pinned commit.

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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=100, max_sandboxes=100, time_limit=3600s, client_timeout=7200, override_cpus=0, kv_cache_dtype=bfloat16, TP=2 DP=4, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

Two fixes to the Harbor LiveCodeBench dataset are proposed upstream in
[harbor-framework/harbor-datasets](https://github.com/harbor-framework/harbor-datasets):
[#257](https://github.com/harbor-framework/harbor-datasets/pull/257) makes `tests/test.sh` write
`/logs/verifier/reward.txt` (reward 0) when the agent never created `/app/solution.py` (otherwise the sample becomes
an error and aborts the eval), and [#256](https://github.com/harbor-framework/harbor-datasets/pull/256) moves the
per-task `COPY` after `apt-get` in `environment/Dockerfile` so the 100 images share their apt layer. Until they are
merged, `run_eval.sh` loads the dataset through `inspect_harbor/harbor` from `assets/harbor_registry.json`: the
upstream `livecodebench@6.0` entry with every task pinned to commit `f7852012df` of
[eldarkurtic/harbor-datasets, branch `livecodebench-eldar-fixes`](https://github.com/eldarkurtic/harbor-datasets/tree/livecodebench-eldar-fixes)
(upstream `main` plus both PRs). Once merged and the hub package republished, replace the two `-T` lines in
`run_eval.sh` with `inspect eval inspect_harbor/livecodebench` and delete `assets/harbor_registry.json`.
