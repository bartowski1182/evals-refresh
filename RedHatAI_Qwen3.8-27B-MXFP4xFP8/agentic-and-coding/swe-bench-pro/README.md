# RedHatAI/Qwen3.8-27B-MXFP4xFP8 — SWE-bench Pro

| | |
|---|---|
| Task | `inspect_harbor/scale_ai_swe_bench_pro` (731 tasks, Harbor hub package digest `sha256:88411d32…`, loaded through `assets/harbor_registry.json`, which pins the tasks to a fork commit that closes the solution leaks, see Patches) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one Docker image per task built from `FROM jefzda/sweap-images:<instance>` (Docker Hub) plus a git reset, bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant covers the W4A4 checkpoint, there is no MXFP4xFP8 variant) |
| Task source | https://hub.harborframework.com/datasets/scale-ai/swe-bench-pro |

**Protocol note.** The base model card reports SWE-bench Pro with the Claude Code harness. `inspect_harbor` solves
every task with Inspect's generic ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout;
automatic context compaction; no message limit), not that harness; a single `--time-limit 14400` wall-clock guard is
used. Containers run without Harbor's CPU cap (`override_cpus=0`) and with `inspect_harbor`'s 6 GB memory floor. One
epoch (pass@1); one TP=8 replica so every turn of a sample hits one prefix cache. This dir is copied from the
Qwen3.8-Flash-Next SWE-bench Pro dir (AGENTS.md, section 11): the tasks run from a dataset commit whose agent phase
has no internet and no git history past the base commit (the published tasks leak the reference solution, see
Patches); the verifier keeps internet, as in the upstream harness. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs. The number is not comparable to the model
card.

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
  (task and registry pin, epochs, sampling params, per-turn max tokens, concurrency, sandbox count, time limit, CPU override, error threshold); identical in the three dirs apart from `MODEL`.
- `assets/harbor_registry.json` — Harbor registry entry with every task pinned to the fork commit carrying the
  dataset fixes (see Patches).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` inside the vLLM image and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM podman image (GPUs 0-7, port 8000, one TP=8 replica, the HF cache and the task dir mounted), points Docker at the rootless daemon, waits for the server, activates the client venv, runs `run_eval.sh`, stops the container.
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
# client: Inspect + the inspect_harbor fork that enforces the tasks' network policy (see Patches). Install the fork
# by commit with --no-deps on top of the release's dependencies: its pyproject's uv exclude-newer breaks a resolving
# install, and a local-directory install records no commit in pip freeze.
python3.12 -m venv ~/.venvs/inspect-harbor-fix && source ~/.venvs/inspect-harbor-fix/bin/activate
pip install "inspect-harbor==1.0.0" httpx2
pip install --no-deps "inspect_harbor @ git+https://github.com/eldarkurtic/inspect_harbor@cb6192598173009fd637e12cf5ec603c3c341359"
pip install "openai==3.14.0"
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

Inspect builds a task's image before its first sample if it is missing (serially, before any sample starts). If you
pre-built the `hb__<hash>` images, pass `--sandbox-prebuilt` to skip that step. The 731 task images take about 3 TB,
and each is built from a Docker Hub base image (100 anonymous pulls per hour per IP), so pull the bases ahead of time
at that rate, or through `mirror.gcr.io` (see `local_env_notes.md`, which also covers the openlibrary base images
whose file owners exceed rootless Docker's subordinate UID range). Run the reference solution first
(`--solver inspect_harbor/oracle`) on the images you built, and hold back tasks it fails that are not among the known
oracle failures listed in the Flash-Next dir's README; hold back the same tasks for the three models. After the run,
check that the sample containers had no route out during the agent phase (AGENTS.md, section 11).
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache (`HF_HOME`,
mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or set
`MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the repo); the Harbor task files are fetched from GitHub at the pinned commit.

### Run the eval

```bash
./local_orchestrator.sh                  # add --sandbox-prebuilt if the task images are pre-built
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5) after the leak audit of section 11, with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), no message limit, max_connections=56, max_sandboxes=56, time_limit=14400s, client_timeout=7200, tool_timeout=300s, override_cpus=0, kv_cache_dtype=bfloat16, TP=8 DP=1, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

The published tasks leak the reference solution to the agent in two ways: no task sets a network policy, so agents
downloaded newer releases or the current source of the project under test; and the task images keep the upstream
clone (`origin/*` branches, later tags and the gold commit itself). The fix is branch
[`swebenchpro/no-solution-leak`](https://github.com/eldarkurtic/harbor-datasets/tree/swebenchpro/no-solution-leak)
of https://github.com/eldarkurtic/harbor-datasets (commit `d81053e87b`, on top of `swebenchpro/verifier-env-fixes`,
i.e. harbor-framework/harbor-datasets#258), which `assets/harbor_registry.json` pins every task to. Upstream PR:
[harbor-framework/harbor-datasets#263](https://github.com/harbor-framework/harbor-datasets/pull/263). Per task:
`[agent] network_mode = "no-network"` in `task.toml` (the verifier keeps network: 61 reference fixes change
dependency manifests whose packages the tests download); `environment/Dockerfile` drops remotes, non-ancestor tags,
other branches, stash/notes/replace refs and reflogs after the reset and runs `gc --prune=now`; the gold tests ship
as `tests/gold_tests.tar`, extracted by `tests/test.sh` with a plain `tar -xf` instead of a checkout of the gold
commit; nodebb tasks that `npm install` at test time get the packages at build time with `npm_config_offline=true`.
inspect_harbor 1.0 applies only `[environment].network_mode`, so the client venv installs branch
[`fix/enforce-agent-network-mode`](https://github.com/eldarkurtic/inspect_harbor/tree/fix/enforce-agent-network-mode)
of https://github.com/eldarkurtic/inspect_harbor (commit `cb61925`), which applies `[agent]` and `[verifier]`
`network_mode` per phase (the container starts with `network_mode: none` and is moved onto Docker's default bridge
for the verifier); upstream PR [meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200).
Also consumed: the verifier fixes of #258 (ansible-test `RLIMIT_NOFILE`, stale `/etc/pip.conf`) and inspect_harbor
1.0's non-login verifier shell (meridianlabs-ai/inspect_harbor#174). Once merged and the hub package republished,
switch back to `inspect_harbor/scale_ai_swe_bench_pro -T ref=<digest>` and delete the registry file.
