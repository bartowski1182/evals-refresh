# RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 — DeepSWE 1.1

| | |
|---|---|
| Task | `inspect_harbor/datacurve_deep_swe_1_1` (113 tasks, Harbor hub package digest `sha256:5affcd53…`, loaded through `assets/harbor_registry.json`, which pins the tasks to the fork commit carrying the verifier fix) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per task (public.ecr.aws, 2 CPUs / 8 GB requested), bash/python tools, hidden test verifier |
| vLLM recipe | https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B |
| Task source | https://hub.harborframework.com/datasets/datacurve/deep-swe-1-1 |

**Protocol note.** The Muse Spark Open card reports 60 for the BF16 model with its "preferred harness" (not named
on the card). `inspect_harbor` solves every task with Inspect's generic ReAct agent (`bash`, `python`,
`update_plan`, `submit`; 300 s tool timeout; automatic context compaction; no message limit); a single
`--time-limit 14400` wall-clock guard is used, `override_cpus=0`, one epoch (pass@1). The container runs without
network, as every DeepSWE task declares for its agent and verifier (stock inspect_harbor 1.0 ignores that and runs
with internet; see Patches), so the numbers are not comparable to the card, and not to the DeepSWE dirs of this
repo that ran without the network fix.
Checkpoint: `RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8`, revision `a37f61bf1a` (compressed-tensors: MXFP4 expert weights
with dynamic FP8 activations, FP8-block attention and expert in/out projections), quantized from
`meta-models/Muse-Spark-Open-816B-A42B` (revision `0fb02478b0`). The model card is Meta's card of the base model; the
numbers it reports are for the BF16 weights at xhigh reasoning effort, temperature 1.0, top_p 1.0, top_k disabled. The
checkpoint's `generation_config.json` sets top_k 64, which this dir follows (the server default; Inspect does not forward
top_k), so sampling differs from the card in top_k.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (TP=8 + expert parallel, 1M
  context, `muse` reasoning and tool-call parsers, default reasoning effort, text-only mode), taken from the official
  recipe (NVIDIA, strategy Tensor + Expert Parallel, features reasoning and tool_calling; no DSpark speculative decoding).
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and dataset pin, epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
- `assets/harbor_registry.json` — Harbor registry entry with every task pinned to the fork commit carrying the
  dataset fixes (see Patches).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM image and of the Inspect venv that
  produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not
  present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM podman
  image (GPUs 0-7, port 8000, one TP=8 replica so each sample's prefix cache stays on one replica, the local checkpoint
  and the task dir mounted), points Inspect's sandboxes at Docker, waits for the server, activates the client venv,
  runs `run_eval.sh`, stops the container.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Build the server image and create the client environment as below. The commands use the image name and venv path that
`local_orchestrator.sh` uses; if you keep them elsewhere, change them there. Once the results are in,
`server_requirements.txt` (`pip freeze` inside the image) and `client_requirements.txt` hold the exact packages that
produced them; create the environments from those files instead.

```bash
# server: a podman image of vLLM with Muse Spark support, which is in no release or nightly wheel: the model lands
# with vllm-project/vllm#61879 (draft, branch muse-spark-support, head 0626d646d0 on 2026-10-20; depends on
# flashinfer-ai/flashinfer#6544). The branch also carries the compressed-tensors MXFP4-weight / FP8-activation MoE
# path this checkpoint needs (vllm-project/vllm#59815, merged 2026-10-14). Build the branch's own Dockerfile into the
# image local_orchestrator.sh runs (IMAGE, default localhost/vllm-dev:latest), or point IMAGE at the recipe's image
# docker.io/vllm/vllm-openai:muse-spark-open@sha256:c91025b5be1039a93ca975d1d44f352b0e770bb6c8b142b5056cf92204ca137c
# (pushed 2026-10-15 from an earlier state of the branch). GPU access needs the NVIDIA Container Toolkit's CDI spec.
git clone --branch muse-spark-support https://github.com/vllm-project/vllm ~/src/vllm-muse
cd ~/src/vllm-muse && git checkout 0626d646d0
podman build -f docker/Dockerfile --target vllm-openai -t vllm-dev .
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # once per host
# client: Inspect + the inspect_harbor fork that enforces the tasks' network policy (see Patches). Install the fork
# by commit with --no-deps on top of the release's dependencies: its pyproject's uv exclude-newer breaks a resolving
# install, and a local-directory install records no commit in pip freeze.
uv venv ~/.venvs/inspect-harbor-fix --python 3.12 && source ~/.venvs/inspect-harbor-fix/bin/activate
uv pip install "inspect-harbor==1.0.0" httpx2
uv pip install --no-deps "inspect_harbor @ git+https://github.com/eldarkurtic/inspect_harbor@cb6192598173009fd637e12cf5ec603c3c341359"
uv pip install "openai==3.14.0"
```

If the image's vLLM is an editable install of a source checkout, `local_orchestrator.sh` mounts that checkout (`VLLM_SRC`) at
`/workspace/vllm`, where the image expects it.

Whichever image you use, check `logs/serve.log` at start-up: the expert layers must load with
`CompressedTensorsW4A8Mxfp4MoEMethod` (MXFP4 weights, FP8 activations). A vLLM without vllm-project/vllm#59815 loads
the checkpoint without error but ignores its FP8 activation scheme and runs the experts with BF16 activations.

Docker with the compose plugin (v2.x or later) and buildx >= 0.17 is required; the other dirs of this repo use a
rootless Docker 29 daemon (containerd image store on local NVMe, 512 network pools, runc with `NoNewKeyring`), and
`local_orchestrator.sh` points `DOCKER_HOST` at its socket. Inspect builds a task's image before its first sample
if it is missing (serially, before any sample starts). If you pre-built the `hb__<hash>` images, pass
`--sandbox-prebuilt` to skip that step. The 113 task images (about 122 GB) come from ECR public
(`public.ecr.aws/d3j8x8q7/swe-bench-202605`, the v1.1 tags), which throttles bursts; pull them ahead of time, one at a
time.

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`, 453 GB; pre-download it with `hf download`
and keep it on local disk); set `MODEL_PATH` to the HF id to let vLLM fetch it. Export `HF_TOKEN` in your shell before
running (never in the repo); the Harbor task files are fetched from GitHub at the pinned commit.

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

`temperature=1.0, top_p=1.0, top_k=64, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=48, max_sandboxes=48, time_limit=14400s, client_timeout=7200, override_cpus=0, TP=8 DP=1 (expert parallel), epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

- **The task's own network policy was not enforced.** Every DeepSWE `task.toml` declares `[agent]` and `[verifier]`
  `network_mode = "no-network"`, but inspect_harbor 1.0 reads network policy only from `[environment]`, so agents get
  public internet. Fixed on branch [`fix/enforce-agent-network-mode`](https://github.com/eldarkurtic/inspect_harbor/tree/fix/enforce-agent-network-mode)
  of https://github.com/eldarkurtic/inspect_harbor (commit `cb61925`, on top of v1.0.0), which applies `[agent]` and
  `[verifier]` `network_mode` per phase; with both denied, the container runs with `network_mode: none` throughout.
  The client venv installs inspect_harbor from that commit; check that the sample containers have no route out.
  Upstream PR: [meridianlabs-ai/inspect_harbor#200](https://github.com/meridianlabs-ai/inspect_harbor/pull/200).
- **Git history.** The v1.1 task images (`public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1`) are built from the
  tasks' `environment/Dockerfile`, which keeps only the base commit's history (all 113 checked in the
  Qwen3.8-Flash-Next run: every ref reachable from `HEAD`, no unreachable commits). The v1.0 images keep future
  `origin/*` refs; do not use them.
- Carried over from the earlier DeepSWE dirs and consumed here: the `[[verifier.collect]]` hook (inspect_harbor
  0.7.6, meridianlabs-ai/inspect_harbor#155), the vitest worker-pool OOM fixed with `taskset` in the pinned fork
  commit `89080fcd34` of https://github.com/eldarkurtic/deep-swe (datacurve-ai/deep-swe#100), and the verifier's
  non-login shell (inspect_harbor 1.0).
