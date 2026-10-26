# RedHatAI/Qwen3.8-27B-MXFP4 — SWE-bench Verified

| | |
|---|---|
| Task | `inspect_evals/swe_bench` (500 instances, `princeton-nlp/SWE-bench_Verified` test split, revision `c104f840cc`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per instance (official SWE-bench images, Docker Hub), no internet, test-based grading |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant serves this checkpoint with the same flags) |
| Task source | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench (eval version 5-C) |

**Protocol note.** The base model card does not report SWE-bench Verified. `inspect_evals` solves each instance with
its default ReAct agent (`python`, `bash_session`, `text_editor`, `submit`; 210 s tool timeout) with
`--message-limit 200` and a `--time-limit 14400` guard, no internet (the task's default), graded by the official
`swebench` harness (FAIL_TO_PASS and PASS_TO_PASS must all pass). One epoch (pass@1); one TP=8 replica so every turn
of a sample hits one prefix cache. This dir is copied from the Qwen3.8-Flash-Next SWE-bench Verified dir (AGENTS.md,
section 11): the sandbox images are the official SWE-bench images
(`-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`) instead of `inspect_evals`'
default Epoch AI images, which keep every release tag of the project and so expose the reference fix offline (see
Patches). This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs.

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
  (task and image template, epochs, sampling params, reasoning effort, per-turn max tokens, message and time limits, concurrency, sandbox count); identical in the three dirs apart from `MODEL`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM server environment and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` from the vLLM podman image (`VLLM_IMAGE`, by default the pinned one) or a venv (`VLLM_VENV`) through `tools/vllm_server.sh` (GPUs 0-7, port 8000, one TP=8 replica, the HF cache and the task dir mounted in podman mode), points Docker at the rootless daemon, waits for the server, activates the client venv (`INSPECT_VENV`), runs `run_eval.sh`, stops the server on exit.
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
# client: Inspect + inspect_evals with the swe_bench extra (swebench pinned < 5, see Patches)
python3.12 -m venv ~/.venvs/inspect-swebench && source ~/.venvs/inspect-swebench/bin/activate
pip install "inspect-ai==0.3.263" "inspect-evals[swe_bench]==0.20.0" "swebench==4.1.0" "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `compressed-tensors`, the log must name the MXFP4
kernel (`Using <Kernel> for MXFP4 GEMM`), the KV cache dtype must be `bfloat16` (the checkpoint's FP8 KV scheme is
overridden), and the default sampling params vLLM takes from the checkpoint's `generation_config.json` must be
temperature 1.0, top_p 0.95, top_k 20.

Docker with the compose plugin (v2.x or later) is required for the sandboxes (podman serves the model only).
`local_orchestrator.sh` exports `DOCKER_HOST` for a rootless daemon (`unix:///run/user/<uid>/docker.sock`); with a
rootful daemon drop that export. Rootless Docker needs
a runc runtime with `NoNewKeyring` in `daemon.json` (one kernel key per container otherwise exhausts a user's quota of
200) and cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`. Check the image-store disk space before
the first run (see `local_env_notes.md`).

The 500 instance images (official SWE-bench images `swebench/sweb.eval.x86_64.*`, about 5 GB each before layer
sharing) are pulled, not built, from Docker Hub (100 anonymous pulls per hour per IP); pull them ahead of time at
that rate, or through `mirror.gcr.io` (see `local_env_notes.md`, which also covers the 34 matplotlib images whose
file owners exceed rootless Docker's subordinate UID range). Before an instance runs, check its image for commits
newer than the base commit (AGENTS.md, section 11).
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache (`HF_HOME`,
mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or set
`MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the repo); the dataset is downloaded from the Hugging Face Hub at the pinned revision.

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

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=56, max_sandboxes=56, time_limit=14400s, message_limit=200, client_timeout=7200, kv_cache_dtype=bfloat16, TP=8 DP=1, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

(1) Solution leak in the default images: `inspect_evals`' default Epoch AI images
(`ghcr.io/epoch-research/swe-bench.eval.x86_64.<id>`) keep every release tag of the project, thousands of commits
after the instance's base commit, plus unreachable commit objects, so the reference fix is readable offline (e.g.
`git show 3.0:django/contrib/auth/validators.py` in django__django-11099). The official SWE-bench images, built with
the current harness (remote removed, tags newer than the base commit deleted, reflog expired, `gc --prune`), are
clean except for the matplotlib images, which keep release tags of maintenance branches dated before the base commit.
Reported upstream as [UKGovernmentBEIS/inspect_evals#2632](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2632);
`run_eval.sh` selects the official images with `-T image_name_template`. (2) `inspect_evals` declares
`swebench>=3.0.15` but its scorer needs swebench < 5
(https://github.com/UKGovernmentBEIS/inspect_evals/issues/2545); fixed on branch
`fix/swe-bench-cap-swebench-below-5` of https://github.com/eldarkurtic/inspect_evals; the client venv pins swebench
4.1.0 (same scorer code).
