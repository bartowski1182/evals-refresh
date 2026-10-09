# RedHatAI/Qwen3.8-27B-MXFP4 — SWE-bench Verified

| | |
|---|---|
| Task | `inspect_evals/swe_bench` (500 instances, `princeton-nlp/SWE-bench_Verified` test split, revision `c104f840cc`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per instance (Epoch AI, ghcr.io), no internet, test-based grading |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; it has no MXFP4 variant) |
| Task source | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench (eval version 5-C) |

**Protocol note.** This is Qwen/Qwen3.8-27B with the linear layers' weights and activations quantized to MXFP4 (W4A4,
GPTQ with AWQ smoothing, LLM Compressor, compressed-tensors format), checkpoint revision `651eed99386d`. Task, agent
and limits are those of the base model's run (`../../../Qwen_Qwen3.8-27B/agentic-and-coding/swe-bench-verified`):
`inspect_evals`'s default ReAct agent (`python`, `bash_session`, `text_editor`, `submit`; 210 s tool timeout),
`--message-limit 200`, a `--time-limit 14400` wall-clock guard, 56 samples in flight, one epoch (pass@1 over the 500
instances), grading by the official `swebench` harness (FAIL_TO_PASS and PASS_TO_PASS must all pass), and the base
model's sampling, reasoning effort, per-turn token budget and Inspect environment. The serving setup differs: AMD
MI355X with vLLM 0.31.0 for ROCm, four single-GPU replicas behind one endpoint (TP=1, DP=4) instead of one TP=8
replica on 8x H100 with vLLM 0.29.0. The prefix cache still serves about 85% of the prompt tokens across the four
replicas, and each sample advances at 2 to 3 agent turns per minute with 56 in flight (the base run: about 3), so the
200-message limit is reached well inside the 4-hour time limit. The checkpoint also declares a static per-tensor FP8
KV-cache scheme (calibrated `k_scale`/`v_scale`), which vLLM applies by default; `serve.sh` overrides it with
`--kv-cache-dtype bfloat16`, so the KV cache is BF16 as for the base model and for `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`,
and only the MXFP4 quantization differs between the two checkpoints. MXFP4 GEMMs run natively on gfx950 (vLLM's
`AiterMxfp4LinearKernel`); attention runs on vLLM's Triton backend (`--attention-backend TRITON_ATTN`) instead of its
ROCm default, which decodes about 14x slower at long context. The 34 matplotlib instances' images contain files owned
by UIDs above 65535, which rootless Docker cannot extract; they run from copies with those files owned by root
(contents unchanged), as for the base model. The official leaderboard numbers come from other scaffolds (SWE-agent,
OpenHands, Claude Code, ...) and are not directly comparable; neither model card reports SWE-bench Verified.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers, KV-cache dtype, attention backend).
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, per-turn max tokens, message limit, concurrency, sandbox count, time limit).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` inside the vLLM container and of the
  Inspect venv that produced the results below (the `inspect_evals` venv with the `swe_bench` extra, see Run).
- `local_orchestrator.sh` — runs everything in your local dev environment: sets GPUs, points Docker at the
  rootless daemon, starts `serve.sh` inside the vLLM ROCm container in the background, activates the Inspect venv,
  runs `run_eval.sh`, tears down. **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

`local_orchestrator.sh` runs `serve.sh` in the official vLLM ROCm image under rootless podman (it needs read-write
access to `/dev/kfd` and `/dev/dri/renderD*`), the Inspect client from a virtualenv you create as below (at the path
the orchestrator activates, or edit it there), and the sandboxes on a rootless Docker daemon.
`server_requirements.txt` is the `pip freeze` of that image; `client_requirements.txt` is the exact `pip freeze` of
the client environment that produced the results below, the same packages as the base model's SWE-bench client
environment.

```bash
# server: vLLM 0.31.0 for ROCm 7.2, from the official image pinned by digest
podman pull docker.io/vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f
# client: Inspect with the swe_bench extra (Python 3.12)
uv venv --python 3.12 --seed ~/.venvs/inspect-swe && source ~/.venvs/inspect-swe/bin/activate
pip install -r client_requirements.txt          # what produced the results; or inspect_evals with the swebench
# cap straight from the fork branch (see Patches):
# pip install "inspect-evals[swe_bench] @ git+https://github.com/eldarkurtic/inspect_evals@fix/swe-bench-cap-swebench-below-5" openai
# weights: into the HF hub cache that local_orchestrator.sh mounts into the container (HF_HUB_CACHE, default ~/hf_hub_cache)
HF_HUB_CACHE=~/hf_hub_cache hf download RedHatAI/Qwen3.8-27B-MXFP4 --revision 651eed99386dae5645cd4949024e8ebd163a533b
```

Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited. The server runs with `HF_HUB_OFFLINE=1` and needs the weights downloaded as above.

Docker (with the compose plugin) must be reachable by your user. `local_orchestrator.sh` exports `DOCKER_HOST` for a
rootless daemon (`unix:///run/user/<uid>/docker.sock`); with a rootful daemon drop that export. Rootless Docker needs
a runc runtime with `NoNewKeyring` (one kernel key per container otherwise exhausts a user's quota of 200) and, with
the usual 65536 subordinate IDs, cannot extract the matplotlib images unmodified; the host-side measures are in
`local_env_notes.md`.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

Requires Docker with roughly 150 GB free in its image store. Inspect pulls one prebuilt image per instance from
`ghcr.io/epoch-research` as containers start (500 images sharing most of their layers; anonymous pulls work).
Pre-pull with `docker pull` to avoid stalling the first samples. Containers run with `network_mode: none`.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), message_limit=200, time_limit=14400s, client_timeout=7200, tool_timeout=210s, TP=1 DP=4, epochs=1; 500 instances in one run at 56 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.758 (378 of 499 scored instances) |
| stderr (binomial) | 0.0192 |
| samples | 500 (500 instances x 1 epoch, in 1 run) |
| message-limit hits (200) | 121 |
| time-limit hits (14400 s) | 0 |
| samples that called `submit()` | 2 |
| turns truncated at max_tokens | 2 |
| sample errors (excluded from the score) | 1 (scikit-learn__scikit-learn-14710: after the agent's 200 messages, the official harness's test run exceeded Inspect's 7200 s scoring limit; the same instance errored for both MXFP4 checkpoints) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 3,507,570 | 44,524 | 34,531 |
| median | 3,121,894 | 38,304 | 28,874 |
| max | 10,051,236 | 178,337 | 168,278 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns
per sample: median 69, mean 69, max 99. Wall time per sample: median 2,036 s, mean 2,272 s, max 10,681 s. Total run time about 6 h 36 min on 4x MI355X.

Per repository (resolved / instances): django 182/231, sympy 56/75, sphinx 34/44, matplotlib 22/34, scikit-learn 28/32, astropy 16/22, xarray 16/22, pytest 16/19, pylint 5/10, requests 1/8, seaborn 1/2, flask 1/1.

## Patches for upstream

`inspect_evals` declares `swebench>=3.0.15` but its scorer imports `MAP_REPO_VERSION_TO_SPECS` and
`swebench.harness.test_spec`, which swebench 5.0 removed, so a fresh install resolves 5.x and every sample errors at
scoring time (swebench 4.1.0 works). Reported as https://github.com/UKGovernmentBEIS/inspect_evals/issues/2545; the
fix caps the `swe_bench` extra at `swebench<5` on branch `fix/swe-bench-cap-swebench-below-5` of
https://github.com/eldarkurtic/inspect_evals, which gives the same environment (swebench 4.1.0). The results below were produced with the base model's
client environment (`client_requirements.txt`: inspect_evals 0.20.0, swebench pinned to 4.1.0), the same scorer code.
Once the fix is released, install the release instead.
