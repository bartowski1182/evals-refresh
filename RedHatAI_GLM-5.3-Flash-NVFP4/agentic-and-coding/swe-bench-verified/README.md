# RedHatAI/GLM-5.3-Flash-NVFP4 — SWE-bench Verified

| | |
|---|---|
| Task | `inspect_evals/swe_bench` (500 instances, `princeton-nlp/SWE-bench_Verified` test split, revision `c104f840cc`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per instance (official SWE-bench images, Docker Hub), no internet, test-based grading |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash |
| Task source | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench (eval version 5-C) |

**Protocol note.** The GLM-5.3-Flash card does not report SWE-bench Verified. `inspect_evals` solves each instance
with its default ReAct agent (`python`, `bash_session`, `text_editor`, `submit`; 210 s tool timeout) with
`--message-limit 200` and a `--time-limit 14400` guard, no internet (the task's default), graded by the official
`swebench` harness (FAIL_TO_PASS and PASS_TO_PASS must all pass). One epoch (pass@1). The sandbox images are the
official SWE-bench images (`-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`)
instead of `inspect_evals`' default Epoch AI images, which keep every release tag of the project and so expose the
reference fix offline (see Patches). Checkpoint: NVFP4 (mixed precision — W4A4 on most MoE experts, FP8 on the last
expert layer and the MTP module), compressed-tensors format (LLM Compressor), base `zai-org/GLM-5.3-Flash`,
revision `18d55bf`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtype, MTP speculative decoding, reasoning/tool parsers, text-only mode), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task and dataset pin,
  epochs, sampling params, reasoning effort, per-turn max tokens, message/time limits, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `uv pip freeze` of the vLLM and Inspect venvs that
  produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs (0-7,
  port 8000), one TP=8 replica (so each sample's prefix cache stays on one replica), the local checkpoint path,
  starts `serve.sh` in the background, runs `run_eval.sh`, tears down. **This is the only file you should modify**
  to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `uv pip freeze` of the environments that produced the results below
(Python 3.12, host details in their headers).

```bash
# server: a vLLM nightly that carries the glm5_next integration (0.32.0rc1.dev249+gfaf417b75, commit faf417b75).
# Stock nightly loads this checkpoint (no model-specific fork needed). flashinfer-python 0.7.1 provides the sparse
# MLA kernels; CUDA_HOME must point at a CUDA 13 toolkit so its JIT path works.
uv venv ~/.venvs/glm-vllm --python 3.12 && source ~/.venvs/glm-vllm/bin/activate
uv pip install -r server_requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/faf417b75 \
  --extra-index-url https://download.pytorch.org/whl/cu130
# (server_requirements.txt records vLLM as an editable checkout; installing the nightly wheel for commit faf417b75
#  gives an equivalent environment, since the glm5_next support is upstream.)

# client: Inspect + inspect_evals
uv venv ~/.venvs/glm-inspect-swebench --python 3.12 && source ~/.venvs/glm-inspect-swebench/bin/activate
uv pip install -r client_requirements.txt   # or: uv pip install "inspect-ai==0.3.263" "inspect-evals[swe_bench]==0.20.0" "swebench==4.1.0" "openai==3.14.0"
```

Docker with the compose plugin (v2.x or later) is required. The results below were produced with the host's system
(rootful) Docker 25 daemon (overlay2 image store on local NVMe); the invoking user is in the `docker` group, so
`local_orchestrator.sh` leaves `DOCKER_HOST` at its default. The 500 instance images (official SWE-bench images
`swebench/sweb.eval.x86_64.*`, about 5 GB each before layer sharing) are pulled, not built, from Docker Hub (100
anonymous pulls per hour per IP); pull them ahead of time, or through `mirror.gcr.io` (which serves the same manifest
digests) after checking the digest, then run.

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`); unset it to let vLLM fetch
`RedHatAI/GLM-5.3-Flash-NVFP4` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running (never in
the repo); the dataset is downloaded from the Hugging Face Hub at the pinned revision.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

_Run pending — this section will be filled with the all-samples accuracy (pass@1), stderr, sample errors,
time-limit hits, truncations, leak-audit result, and token stats once the full 500-instance run completes._

## Patches for upstream

Solution leak in the default images: `inspect_evals`' default Epoch AI images
(`ghcr.io/epoch-research/swe-bench.eval.x86_64.<id>`) keep every release tag of the project, so the reference fix of
an instance can be read offline (`git show <later tag>:<file>`), and `git log --all` can surface the fix commit. The
official SWE-bench images, built with the current harness (remote removed, tags newer than the base commit deleted,
reflog expired, `gc --prune`), are clean; this dir selects them with
`-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`. Reported upstream as
[UKGovernmentBEIS/inspect_evals#2632](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2632). Separately,
`inspect_evals` declares `swebench>=3.0.15` but its scorer needs swebench < 5
([UKGovernmentBEIS/inspect_evals#2545](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2545)); the client
venv pins `swebench==4.1.0` (same scorer code).

No model-specific vLLM fork is needed: the stock nightly (commit faf417b75) loads and serves this checkpoint (the
Qwen4Exp PLE fix that the Qwen3.8-Flash-Next dirs required does not apply to `glm5_next`).
