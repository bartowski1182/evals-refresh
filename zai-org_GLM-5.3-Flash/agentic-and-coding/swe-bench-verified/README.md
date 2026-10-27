# zai-org/GLM-5.3-Flash — SWE-bench Verified

| | |
|---|---|
| Task | `inspect_evals/swe_bench` (500 instances, `princeton-nlp/SWE-bench_Verified` test split, revision `c104f840cc`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per instance (official SWE-bench images, Docker Hub), no internet, test-based grading |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash (default variant: the native FP8 checkpoint) |
| Task source | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench (eval version 5-C) |

**Protocol note.** The GLM-5.3-Flash card does not report SWE-bench Verified. `inspect_evals` solves each instance
with its default ReAct agent (`python`, `bash_session`, `text_editor`, `submit`; 210 s tool timeout) with
`--message-limit 200` and a `--time-limit 14400` guard, no internet (the task's default), graded by the official
`swebench` harness (FAIL_TO_PASS and PASS_TO_PASS must all pass). One epoch (pass@1); one TP=8 replica so every turn
of a sample hits one prefix cache. The sandbox images are the official SWE-bench images
(`-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`) instead of `inspect_evals`'
default Epoch AI images, which keep every release tag of the project and so expose the reference fix offline (see
Patches). This dir is one of the two identical copies of the task that make up the GLM-5.3-Flash pair
(`zai-org_GLM-5.3-Flash`, the native FP8 model, and `RedHatAI_GLM-5.3-Flash-MXFP4xFP8`): same task, agent, prompt,
sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology (8x H200, one TP=8 replica,
BF16 KV cache), so the two numbers are like for like; only the checkpoint differs.

**Checkpoint.** `zai-org/GLM-5.3-Flash` is the vendor's native FP8 release (320B total / 18B active parameters,
hybrid KDA linear attention and NoPE sparse MLA, 288 experts with 8 routed per token, one MTP layer; e4m3 weights with
dynamic activation scaling, attention projections, embeddings, lm_head and the mHC/KDA parameters unquantized),
revision `46a050bec1` (2026-10-15). Its `generation_config.json` sets temperature 1.0 and top_p 0.95 and no top_k.
It is the reference for the quantized copy `RedHatAI/GLM-5.3-Flash-MXFP4xFP8` (the other dir of the pair). The KV
cache is BF16 on Hopper for both models of the pair (see `serve.sh`).

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  BF16 KV cache, MTP speculative decoding, reasoning/tool parsers, text-only mode, default reasoning effort), taken
  from the base model's recipe (Hopper, Tensor Parallel strategy), with two documented departures (`--max-model-len
  262144`, `--language-model-only`); identical in the two dirs of the pair apart from `MODEL` and the checkpoint
  comments.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task and image
  template, epochs, sampling params, reasoning effort, per-turn max tokens, message/time limits, concurrency, error
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
# written before it, asks for a nightly or the vllm/vllm-openai:glm53-flash image); the MXFP4xFP8 dir of the pair
# needs 0.32.0 for its quantization path too. FlashInfer 0.6.17 or newer is required for the NoPE sparse-MLA kernels
# (the image ships 0.7.x). GPU access from the container goes through the NVIDIA Container Toolkit's CDI spec
# (generated once per host, as root).
podman pull docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# To serve from a venv instead, create one with the vLLM version server_requirements.txt records (0.32.0) and pass it
# as VLLM_VENV; the image is the reference environment. If the release cannot load the checkpoint, point VLLM_IMAGE
# at a vLLM nightly image instead and record it.
# client: Inspect + inspect_evals with the swe_bench extra (swebench pinned < 5, see Patches)
python3.12 -m venv ~/.venvs/inspect-swebench && source ~/.venvs/inspect-swebench/bin/activate
pip install "inspect-ai==0.3.263" "inspect-evals[swe_bench]==0.20.0" "swebench==4.1.0" "openai==3.14.0"
```

Check `logs/serve.log` at start-up: the quantization method must be `fp8`, the KV cache dtype must be `bfloat16`,
and MTP must be active (`num_speculative_tokens=5`). Read the `GPU KV cache size` line too: the concurrency in `run_eval.sh` assumes
the pool holds 48 agent histories.

Docker with the compose plugin (v2.x or later) is required for the sandboxes (podman serves the model only).
`local_orchestrator.sh` exports `DOCKER_HOST` for a rootless daemon (`unix:///run/user/<uid>/docker.sock`); with a
rootful daemon drop that export. Rootless Docker maps only the user's subordinate UIDs, and 32 of the 34 matplotlib
images contain files owned by higher UIDs (see `local_env_notes.md` for the rewrite that loads them). The 500
instance images (official SWE-bench images `swebench/sweb.eval.x86_64.*`, about 5 GB each before layer sharing) are
pulled, not built, from Docker Hub (100 anonymous pulls per hour per IP); pull them ahead of time through
`mirror.gcr.io` (which serves the same manifest digests, see `local_env_notes.md`), then run.
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache
(`HF_HOME`, mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or
set `MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the
repo); the dataset is downloaded from the Hugging Face Hub at the pinned revision.

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

`temperature=1.0, top_p=0.95, top_k unset, reasoning_effort=max, max_tokens=65536 (per turn), message_limit=200, max_connections=48, max_sandboxes=48, time_limit=14400s, client_timeout=7200, tool_timeout=210s, kv_cache_dtype=bfloat16, TP=8 DP=1 with MTP (num_speculative_tokens=5), max_model_len=262144, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

(1) Solution leak in the default images: `inspect_evals`' default Epoch AI images
(`ghcr.io/epoch-research/swe-bench.eval.x86_64.<id>`) keep every release tag of the project, so the reference fix of
an instance can be read offline (`git show <later tag>:<file>`), and `git log --all` can surface the fix commit. The
official SWE-bench images, built with the current harness (remote removed, tags newer than the base commit deleted,
reflog expired, `gc --prune`), are clean; this dir selects them with
`-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`. Reported upstream as
[UKGovernmentBEIS/inspect_evals#2632](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2632).
(2) `inspect_evals` declares `swebench>=3.0.15` but its scorer needs swebench < 5
([UKGovernmentBEIS/inspect_evals#2545](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2545)); the client
venv pins `swebench==4.1.0` (same scorer code).
