# RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 — SWE-bench Verified

| | |
|---|---|
| Task | `inspect_evals/swe_bench` (500 instances, `princeton-nlp/SWE-bench_Verified` test split, revision `c104f840cc`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per instance (official SWE-bench images, Docker Hub), no internet, test-based grading |
| vLLM recipe | https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B |
| Task source | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench (eval version 5-C) |

**Protocol note.** The Muse Spark Open card does not report SWE-bench Verified. `inspect_evals` solves each
instance with its default ReAct agent (`python`, `bash_session`, `text_editor`, `submit`; 210 s tool timeout) with
`--message-limit 200` and a `--time-limit 14400` guard, no internet (the task's default), graded by the official
`swebench` harness (FAIL_TO_PASS and PASS_TO_PASS must all pass). One epoch (pass@1). The sandbox images are the
official SWE-bench images (`-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`)
instead of `inspect_evals`' default Epoch AI images, which keep every release tag of the project and so expose the
reference fix offline (see Patches). This matches the Qwen3.8-Flash-Next dirs of this repo and differs from the older
SWE-bench Verified dirs, which used the Epoch AI images.
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
# image local_orchestrator.sh runs (VLLM_IMAGE, default localhost/vllm-dev:latest), or point VLLM_IMAGE at the recipe's image
# docker.io/vllm/vllm-openai:muse-spark-open@sha256:c91025b5be1039a93ca975d1d44f352b0e770bb6c8b142b5056cf92204ca137c
# (pushed 2026-10-15 from an earlier state of the branch). GPU access needs the NVIDIA Container Toolkit's CDI spec.
git clone --branch muse-spark-support https://github.com/vllm-project/vllm ~/src/vllm-muse
cd ~/src/vllm-muse && git checkout 0626d646d0
podman build -f docker/Dockerfile --target vllm-openai -t vllm-dev .
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # once per host
# client: Inspect + inspect_evals, with swebench < 5 (see Patches)
uv venv ~/.venvs/inspect-swebench --python 3.12 && source ~/.venvs/inspect-swebench/bin/activate
uv pip install "inspect-ai==0.3.263" "inspect-evals[swe_bench]==0.20.0" "swebench==4.1.0" "openai==3.14.0"
```

If the image's vLLM is an editable install of a source checkout, `local_orchestrator.sh` mounts that checkout (`VLLM_SRC`) at
`/workspace/vllm`, where the image expects it.

Whichever image you use, check `logs/serve.log` at start-up: the expert layers must load with
`CompressedTensorsW4A8Mxfp4MoEMethod` (MXFP4 weights, FP8 activations). A vLLM without vllm-project/vllm#59815 loads
the checkpoint without error but ignores its FP8 activation scheme and runs the experts with BF16 activations.

Docker with the compose plugin (v2.x or later) is required; the other dirs of this repo use a rootless Docker 29
daemon (containerd image store on local NVMe, 512 network pools, runc with `NoNewKeyring`), and
`local_orchestrator.sh` points `DOCKER_HOST` at its socket. The 500 instance images (official SWE-bench images
`swebench/sweb.eval.x86_64.*`, about 5 GB each before layer sharing) are pulled, not built, from Docker Hub (100
anonymous pulls per hour per IP); pull them ahead of time at that rate, or through `mirror.gcr.io` after checking
that it serves the same manifest digest as Docker Hub. The 34 matplotlib images contain files owned by UIDs beyond
rootless Docker's 65,536 subordinate UIDs; see `local_env_notes.md` for the workaround.

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`, 453 GB; pre-download it with `hf download`
and keep it on local disk); set `MODEL_PATH` to the HF id to let vLLM fetch it. Export `HF_TOKEN` in your shell before
running (never in the repo); the dataset is downloaded from the Hugging Face Hub at the pinned revision.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5) after the leak audit of section 11, with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=1.0, top_k=64, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=56, max_sandboxes=56, time_limit=14400s, message_limit=200, client_timeout=7200, TP=8 DP=1 (expert parallel), epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

(1) Solution leak in the default images: `inspect_evals`' default Epoch AI images
(`ghcr.io/epoch-research/swe-bench.eval.x86_64.<id>`) keep every release tag of the project, up to 2024 releases,
reaching thousands of commits after the instance's base commit, plus unreachable commit objects, so the reference
fix is readable offline (`git show <later tag>:<file>`, `git log --all`). The official SWE-bench images, built with
the current harness (remote removed, tags newer than the base commit deleted, reflog expired, `gc --prune`), are
clean; this dir selects them with `-T image_name_template`. Reported upstream as
[UKGovernmentBEIS/inspect_evals#2632](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2632). (2)
`inspect_evals` declares `swebench>=3.0.15` but its scorer needs swebench < 5
(https://github.com/UKGovernmentBEIS/inspect_evals/issues/2545); fixed on branch
`fix/swe-bench-cap-swebench-below-5` of https://github.com/eldarkurtic/inspect_evals; the client venv pins swebench
4.1.0 (same scorer code).
