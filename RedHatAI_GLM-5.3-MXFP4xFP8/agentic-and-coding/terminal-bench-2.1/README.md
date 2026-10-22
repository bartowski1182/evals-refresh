# RedHatAI/GLM-5.3-MXFP4xFP8 — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task (Docker Hub), bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3 (base model; its Hopper section is identical to this checkpoint's model-card command — the recipes site has no MXFP4xFP8 entry) |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The zai-org/GLM-5.3 card reports 88.2 on Terminal-Bench 2.1 with the Claude Code 2.1.207 harness
(temp 1.0, top_p 1.0, max_new_tokens 65536, 6 h timeout), measured on the FP8 `zai-org/GLM-5.3` checkpoint. This run
uses Inspect's generic ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context
compaction), the checkpoint's own sampling defaults (temp 1.0, top_p 0.95), reasoning effort `max`, a single
`--time-limit 7200` for every task, `override_cpus=0`, one epoch (pass@1), so the numbers are not comparable to the
card or to Terminus 2 leaderboard entries. The tasks have public network by design. Checkpoint: mixed MXFP4 (routed
experts and dense MLPs, W4A8 with per-group dynamic FP8 activations) / FP8 (attention, shared experts, MTP layer),
compressed-tensors format (LLM Compressor), ~4.5 bits per parameter, revision `13c63cd`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, KV-cache dtype,
  MTP speculative decoding, reasoning/tool parsers), taken from the model card's serving command (= the base
  recipe's single-node Hopper section), with one documented departure: `--max-model-len 262144` instead of the
  checkpoint's 1M default, because the DSA sparse-indexer warm-up OOMs on H200 at 1M (see the comment in the file).
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and dataset pin, epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
  Concurrency is 32 in flight rather than all 89 tasks at once, to stay inside the model's ~1.19M-token KV pool
  (see the comment in the file).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs
  (all 8, port 8000, one TP=8 replica so each sample's prefix cache stays on one replica), the HF cache and the
  Docker socket, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below (Python 3.12,
host details in their headers).

```bash
# server: vLLM 0.32.0 (the first release that serves this MXFP4/FP8 checkpoint on Hopper, per the model card)
python3.12 -m venv /data/playground/dsikka/venvs/vllm-glm53 && source /data/playground/dsikka/venvs/vllm-glm53/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.32.0
# client: Inspect + inspect_harbor
python3.12 -m venv /data/playground/dsikka/venvs/inspect-harbor && source /data/playground/dsikka/venvs/inspect-harbor/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "inspect-harbor==1.0.0" httpx2 && pip install "openai==3.14.0"
```

Model weights are pre-downloaded into the HF cache (`HF_HOME=/data/playground/dsikka/hf` in
`local_orchestrator.sh`; about 415 GB) and served with `HF_HUB_OFFLINE=1`. Nothing here is gated; export `HF_TOKEN`
in your shell before downloading (never in the repo) for higher Hub rate limits. The Harbor task files are fetched
from GitHub at the pinned commit on first use.

Docker with the compose plugin (v2.x or later) and buildx >= 0.17 is required. The results below were produced with
a rootless Docker daemon (containerd image store on `/data/playground/dsikka`, 512 network pools, runc with
`NoNewKeyring`); `local_orchestrator.sh` points `DOCKER_HOST` at its socket. Inspect builds a task's image before
its first sample if it is missing (serially, before any sample starts). If you pre-built the `hb__<hash>` images,
pass `--sandbox-prebuilt` to skip that step. Docker Hub limits anonymous pulls per IP (100/hour); the task images
pull base layers from it, so pace pre-builds or use `mirror.gcr.io` (see local_env_notes.md).

### Run the eval

```bash
./local_orchestrator.sh                  # add --sandbox-prebuilt if the task images are pre-built
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

Phase 2 of the reported results (the rerun of time-limit hits and the leaked sample at a higher limit) is
reproduced by forwarding Inspect's sample selection through the orchestrator:

```bash
TIME_LIMIT=14400 ./local_orchestrator.sh --sample-id <comma-separated list of the 13 sample ids>
```

## Results

Two phases, at the user's request (documented in full): phase 1 ran all 89 tasks at `--time-limit 7200`;
phase 2 reran the 12 samples that hit that limit (all failed) plus the one sample that passed by leaking
(see below) at `--time-limit 14400`, selected with `--sample-id`. Each task counts once — the rerun result
replaces the phase-1 result for those 13 samples; the 4 tasks that passed despite hitting the phase-1 limit
keep their passes.

With model generation config (both phases):
`max_connections=32, max_tokens=65536, top_p=0.95, temperature=1.0, reasoning_effort='max'`
(one epoch = pass@1, `-M client_timeout=7200`, 32 concurrent; only `--time-limit` differs between phases)
the following scores and token stats are obtained:

| metric | phase 1 (7200 s, 89 tasks) | phase 2 (14400 s, 13 rerun tasks) | combined |
|---|---|---|---|
| accuracy (pass@1) | 0.787 (70 of 89) as-run; 0.775 (69) leak-adjusted | 0.769 (10 of 13) | **0.888 (79 of 89)** |
| stderr | 0.043 | 0.117 | 0.033 |
| sample errors | 0 | 0 | 0 |
| time-limit hits | 16 (12 failed → rerun; 4 passed anyway) | 0 | — |
| per-turn `max_tokens` truncations | 11 | 9 | — |
| wall time | 4 h 03 m at 32 concurrent | 2 h 37 m at 13 concurrent | — |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| phase 1: mean / median / max | 2,295,317 / 767,216 / 50,514,598 | 74,771 / 48,797 / 871,793 | 63,244 / 39,315 / 827,641 |
| phase 2: mean / median / max | 7,883,685 / 5,947,800 / 34,118,263 | 236,095 / 191,196 / 800,676 | 202,101 / 167,839 / 768,920 |

Run totals — phase 1: 204,283,238 input / 6,654,599 output / 5,628,732 reasoning; phase 2:
102,487,905 input / 3,069,233 output / 2,627,309 reasoning. Input is cumulative over the whole session
(the ReAct loop resends the full history every turn), not peak context; reasoning is counted separately
from output. Phase-2 token means are higher because only the longest-running tasks were rerun.

**Leak (repo §11).** In phase 1, `terminal-bench/mteb-retrieve` (passed, 1.0) web-searched its
instruction's distinctive phrases, located the public HF mirror `harborframework/terminal-bench-2.1`,
downloaded its own `solution/solve.sh` and submitted a verbatim copy. The same-domain problem (the task's
own reference solution needs `huggingface.co` for its embedding model) makes network blocking impossible
without breaking the task, so the fix is upstream (issue drafted for `harbor-framework/terminal-bench-2-1`:
exclude `solution/` from the mirror). In the phase-2 rerun the sample made no mirror access and failed
cleanly, and it is counted as a failure in the combined figure. No other sample in either phase obtained
task files: three further canary-GUID hits in phase 1 were the canary embedded in files the tasks ship
inside their own containers (working files, and one task's deliberately visible tests), available to every
model equally.

Failed tasks in the combined protocol (10 of 89): `dna-insert`, `filter-js-from-html`,
`llm-inference-batching-scheduler`, `make-doom-for-mips`, `mteb-retrieve`, `protein-assembly`,
`pytorch-model-cli`, `qemu-alpine-ssh`, `qemu-startup`, `sanitize-git-repo` (all `terminal-bench/`).

## Patches for upstream

No dataset patch is needed. Known issues in the tooling, documented here for a future run: (1) Inspect 0.3.273's
`vllm` provider learns the served context window from the server's `/v1/models` on a sample's first model
call, but the ReAct agent's compaction resolves its threshold (90% of the window) at sample start, so samples that
start before the first call fall back to Inspect's 128,000 default. Pinning the window needs `set_model_info()` in
Python (no CLI option), or an Inspect fix that resolves the threshold after registration. (2) Trajectories whose own
messages exceed the threshold after compaction end with `RuntimeError: Compaction insufficient` and are not graded.
(3) The tasks have public network by design (several verifiers `apt-get install` at test time), and nothing stops
an agent from fetching the task's own files, including its reference solution, from the public
`harbor-framework/terminal-bench-2-1` repository or mirrors; such samples are flagged in Results.
