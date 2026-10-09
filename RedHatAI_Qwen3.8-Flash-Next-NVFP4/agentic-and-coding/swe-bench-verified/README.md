# RedHatAI/Qwen3.8-Flash-Next-NVFP4 — SWE-bench Verified

| | |
|---|---|
| Task | `inspect_evals/swe_bench` (500 instances, `princeton-nlp/SWE-bench_Verified` test split, revision `c104f840cc`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per instance (official SWE-bench images, Docker Hub), no internet, test-based grading |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |
| Task source | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench (eval version 5-C) |

**Protocol note.** The Qwen/Qwen3.8-Flash-Next card does not report SWE-bench Verified. `inspect_evals` solves each
instance with its default ReAct agent (`python`, `bash_session`, `text_editor`, `submit`; 210 s tool timeout) with
`--message-limit 200` and a `--time-limit 14400` guard, no internet (the task's default), graded by the official
`swebench` harness (FAIL_TO_PASS and PASS_TO_PASS must all pass). One epoch (pass@1). The sandbox images are the
official SWE-bench images (`-T image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`)
instead of `inspect_evals`' default Epoch AI images, which keep every release tag of the project and so expose the
reference fix offline (see Patches). This differs from the other SWE-bench Verified dirs of this repo, which used
the Epoch AI images, so the numbers are not comparable. Checkpoint: NVFP4 (W4A4) MoE experts, compressed-tensors
format (LLM Compressor), revision `863f54a`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtypes, reasoning/tool parsers, text-only mode), taken from the official recipe (Blackwell, features
  tool_calling, reasoning, text_only).
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and dataset pin, epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs
  (0-3, port 8000), one TP=4 replica (so each sample's prefix cache stays on one replica), the local
  checkpoint path and the Docker socket, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

`local_orchestrator.sh` activates `~/.venvs/vllm-nightly-ple-fix` for the server and `~/.venvs/inspect-swebench`
for the Inspect client; create them as below (edit the paths in `local_orchestrator.sh` if you keep them
elsewhere). `server_requirements.txt` and `client_requirements.txt` are the exact `pip freeze` of the environments
that produced the results below (Python 3.12, host details in their headers).

```bash
# server: the same vLLM nightly (0.30.1rc1.dev418+g50319916a) binaries plus the Qwen4Exp compressed-tensors
# fix from the fork branch (see Patches for upstream)
uv venv ~/.venvs/vllm-nightly-ple-fix --python 3.12 && source ~/.venvs/vllm-nightly-ple-fix/bin/activate
uv pip install -r server_requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/50319916a3c72ec1358f9e29c7c282d316d86f78 \
  --extra-index-url https://download.pytorch.org/whl/cu130
git clone -b fix/qwen4-exp-ple-compressed-tensors https://github.com/eldarkurtic/vllm.git && cd vllm
git checkout 77e1e136bf884d7e126af5a7561c7ce39f37cd7c   # the fix commit that produced the results; the branch has since merged newer main
uv pip install "cmake>=3.26.1" ninja "setuptools>=77.0.3,<81" "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" wheel jinja2
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=50319916a3c72ec1358f9e29c7c282d316d86f78 \
  uv pip install --no-deps --no-build-isolation -e .
# client: Inspect + inspect_evals
uv venv ~/.venvs/inspect-swebench --python 3.12 && source ~/.venvs/inspect-swebench/bin/activate
uv pip install -r client_requirements.txt   # or: uv pip install "inspect-ai==0.3.263" "inspect-evals[swe_bench]==0.20.0" "swebench==4.1.0" "openai==3.14.0"
```

Docker with the compose plugin (v2.x or later) is required. The results below were produced with a rootless Docker
29 daemon (containerd image store on local NVMe, 512 network pools, runc with `NoNewKeyring`);
`local_orchestrator.sh` points `DOCKER_HOST` at its socket. The 500 instance images (official SWE-bench images
`swebench/sweb.eval.x86_64.*`, about 5 GB each before layer sharing) are pulled, not built, from Docker Hub (100
anonymous pulls per hour per IP); pull them ahead of time at that rate, or through `mirror.gcr.io` after checking
that it serves the same manifest digest as Docker Hub.

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`); unset it to let vLLM fetch
`RedHatAI/Qwen3.8-Flash-Next-NVFP4` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running
(never in the repo); the dataset is downloaded from the Hugging Face Hub at the pinned revision.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=56, max_sandboxes=56, time_limit=14400s, message_limit=200, client_timeout=7200, TP=4 DP=1, epochs=1; 500 tasks in 5 runs of 62 / 64 / 65 / 91 / 218 tasks; errored samples retried once with `inspect eval-retry` (completed samples kept)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.794 (396 of 499 scored tasks) |
| stderr (binomial) | 0.0181 |
| resolved rate counting sample errors as failures | 0.792 (396 of 500) |
| samples | 500 tasks, 1 epoch, in 5 runs: 62 + 64 + 65 + 91 + 218 |
| time-limit hits (14400 s) | 0 |
| message-limit hits (200 messages) | 101 |
| turns truncated at max_tokens | 0 |
| tool calls that hit the tool timeout | 78 of 34983 (0.22%) |
| sample errors (excluded from the score) | 1 (scikit-learn__scikit-learn-14710) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 2,681,311 | 34,619 | 24,533 |
| median | 2,131,186 | 29,763 | 20,586 |
| max | 8,881,735 | 139,604 | 94,794 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 61, mean
64, max 99. Wall time per sample: median 1,951 s, mean 2,295 s, max 9,063 s. Summed run time about 12.4 h across
the runs on 4x B200 (one TP=4 replica, 56 concurrent samples), with the other model's runs on the other four GPUs
of the host at the same time.

Per repository (resolved / tasks): django 197/231, sympy 56/75, sphinx-doc 30/44, matplotlib 26/34, scikit-learn 27/31, astropy 17/22, pydata 18/22, pytest-dev 17/19, pylint-dev 5/10, psf 1/8, mwaskom 1/2, pallets 1/1.

## Patches for upstream

(1) Solution leak in the default images: `inspect_evals`' default Epoch AI images
(`ghcr.io/epoch-research/swe-bench.eval.x86_64.<id>`) keep every release tag of the project, up to 2024 releases,
reaching thousands of commits after the instance's base commit, plus 200-570 unreachable commit objects. The
reference fix is readable offline: in django__django-11099, `git show 3.0:django/contrib/auth/validators.py` shows
the fixed regex and `git log --all` finds the fix commit itself; in astropy__astropy-12907, `git show
v5.1:astropy/modeling/separable.py` shows the fixed line. The official SWE-bench images, built with the current
harness (remote removed, tags newer than the base commit deleted, reflog expired, `gc --prune`), are clean. Every
image used here was checked before its instance ran: no commit newer than the base commit reachable from any ref,
no unreachable commits. All refs are ancestors of `HEAD` except in the matplotlib images, which keep the release
tags of maintenance branches dated before the base commit (e.g. v3.6.1-v3.6.3 for a January 2023 base); the harness
deletes only tags newer than the base commit, and a fix merged to `main` after the base commit is not in them. The
official images carry one extra local commit on top of the base commit (a file-mode change by the image build);
grading resets test files to the base commit and is unaffected. Reported upstream as
[UKGovernmentBEIS/inspect_evals#2632](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2632): switch
`DEFAULT_IMAGE_TEMPLATE` to the official images, or ask Epoch AI to rebuild theirs with the current harness. (2)
`inspect_evals` declares `swebench>=3.0.15` but its scorer needs swebench < 5
(https://github.com/UKGovernmentBEIS/inspect_evals/issues/2545); fixed on branch
`fix/swe-bench-cap-swebench-below-5` of https://github.com/eldarkurtic/inspect_evals; the client venv pins swebench
4.1.0 (same scorer code). Host-side: the 34 matplotlib images contain files owned by UIDs beyond rootless Docker's
65,536 subordinate UIDs; they were loaded with those owners remapped to root, with layers rewritten from the
registry (the processes in the sandbox run as root).

vLLM (nightly and `main` as of 2026-09-30) cannot load this checkpoint:
`Qwen4ExpPLEEmbeddingMethod.from_quant_config` raises `NotImplementedError: Qwen4Exp PLE embedding does not support
quantization config CompressedTensorsConfig` for any compressed-tensors checkpoint, even when no config group
targets the n-gram table. The fix (the table is loaded unquantized unless a compressed-tensors config group targets
it) is branch
[`fix/qwen4-exp-ple-compressed-tensors`](https://github.com/eldarkurtic/vllm/tree/fix/qwen4-exp-ple-compressed-tensors)
of https://github.com/eldarkurtic/vllm, based on the nightly commit `50319916a3` (fix commit `77e1e136bf`); the
server venv installs it editable on top of that nightly's precompiled binaries (see Setup). Fixed upstream since:
[vllm-project/vllm#59431](https://github.com/vllm-project/vllm/pull/59431) (merged 2026-09-30, after the nightly
used here) makes the same change; nightlies that include it should load this checkpoint without the fork (not
verified here).
