# RedHatAI/Qwen3.8-27B-MXFP4 — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task, bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (base model; the recipe has no MXFP4 variant) |
| Model | https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4 (revision `651eed9938`): W4A4 MXFP4 (compressed-tensors) |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The task, agent and generation settings are those of
`Qwen_Qwen3.8-27B/agentic-and-coding/terminal-bench-2.1`:
- Inspect's generic ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context
  compaction), not the Terminus 2 agent of the leaderboard and the Qwen model card.
- One uniform `--time-limit 7200` instead of Harbor's per-task agent timeouts.
- Containers without Harbor's CPU cap (`override_cpus=0`) and with `inspect_harbor`'s 6 GB memory floor.
- Per-turn max_tokens 65536, sampling from the checkpoint's `generation_config.json` (identical to the base model's),
  reasoning effort xhigh, avg@3.

Four things differ from the base run:
- Concurrency: at most 89 samples in flight per model (epoch 1 with all 89 at once, then epochs 2-3 with 89 in
  flight), where the base ran all 267 samples at once on one server.
- The KV cache is served in bf16 (`--kv-cache-dtype bfloat16`), although the checkpoint declares an FP8 KV-cache scheme,
  so this run and `amd_Qwen3.8-27B-Quark-AWQ-MXFP4` differ only in the MXFP4 weights/activations.
- The serving stack: 4x AMD MI355X (TP=1, DP=4), vLLM 0.31.0 ROCm build, AITER W4A4 MXFP4 GEMM, Triton attention.
  The base ran on 8x H100 (TP=2, DP=4) with vLLM 0.29.0.
- The host: this run and the AMD model's ran side by side, so 178 sandboxes shared 128 CPU cores. The base had 267
  sandboxes on 384 CPUs. Tool execution (builds, training) dominates wall time and the limit is wall-clock, so
  time-limit hits are not directly comparable.

The numbers are therefore not directly comparable to the model card (73.0 with Terminus 2), and only roughly to the
base run.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here: parallelism, context length and
  reasoning/tool parsers from the base model's recipe, plus the KV-cache dtype, attention backend and load strategy.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task and package digest,
  epochs, sampling params, per-turn max tokens, concurrency, sandbox count, time limit, CPU override).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM container and of the `inspect-harbor`
  venv that produced the results below, with the image digest and host in the headers.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM ROCm
  container (rootless podman), points Inspect's sandboxes at rootless Docker, waits for the server, activates the
  Inspect venv, runs `run_eval.sh`, tears down. **This is the only file you should modify** to match the environment
  you run in.

## Run

### Setup (once per host)

The server runs from the official vLLM ROCm image, so it needs no venv. `local_orchestrator.sh` expects:
- rootless podman and `numactl` for the server, with `/dev/kfd` and `/dev/dri` accessible to your user;
- Docker with the compose plugin for the sandboxes. The orchestrator exports `DOCKER_HOST` for a rootless daemon
  (`unix:///run/user/<uid>/docker.sock`, image store under `~/.local/share/docker`); with a rootful daemon, drop that
  export.

Rootless Docker needs a runc runtime with `NoNewKeyring` in `daemon.json`; otherwise each container takes one kernel key
and exhausts a user's quota of 200. It cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`.

Create the client environment as below. The commands use the path that `local_orchestrator.sh` activates; if you keep
it elsewhere, change the path there. `client_requirements.txt` is the exact `pip freeze` that produced the results
below.

```bash
# server: vLLM 0.31.0 ROCm image (digest in server_requirements.txt)
podman pull docker.io/vllm/vllm-openai-rocm:v0.31.0
# client: Inspect + inspect-harbor
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install --no-deps -r client_requirements.txt   # or:
# pip install "inspect-harbor==0.7.5" httpx2 "openai==3.14.0"
```

`inspect-harbor` pulls in `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible providers need
`openai>=3`. The freeze therefore contains `openai==3.14.0`, and pip refuses to resolve it, hence `--no-deps`. The
resulting `pip check` warning is harmless because Inspect's agent path never imports `litellm`. `httpx2` is imported by
`inspect-ai` but not declared.

Model weights are fetched from the Hugging Face Hub on first use into `~/.cache/huggingface` (or `HF_HOME`), which the
orchestrator mounts into the container. Nothing here is gated; export `HF_TOKEN` (never in the repo) to avoid the Hub's
anonymous rate limits.

### Run the eval

```bash
./local_orchestrator.sh                         # all 8 GPUs (TP=1, DP=8)
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`. The results below come from
two runs with the same config: `GPUS=0,1,2,3 DP=4 PORT=8000 EPOCHS=1 ./local_orchestrator.sh` (epoch 1), then the same with
`EPOCHS=2 MAX_CONNECTIONS=89 MAX_SANDBOXES=89` (epochs 2-3), each with the AMD model's run on GPUs 4-7 at the same time.

Requires Docker with roughly 70 GB free in its image store. Harbor downloads the 89 task definitions into
`~/.cache/harbor/tasks/packages/terminal-bench/`. Inspect then pulls one prebuilt image per task
(`alexgshaw/<task>:20251031` on Docker Hub; nothing is built). Pre-pulling them one by one with `docker pull` avoids
stalls on Docker Hub's anonymous rate limit.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), time_limit=7200s, client_timeout=7200, override_cpus=0, epochs=3 (mean; runs of 1 epoch at max_connections=267, all 89 in flight, and 2 epochs at max_connections=max_sandboxes=89); kv_cache_dtype=bfloat16, attention_backend=TRITON_ATTN, vLLM 0.31.0 (ROCm), TP=1 DP=4 on 4x MI355X`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (avg@3, all 267 samples; sample errors count as failures) | 0.554 |
| stderr | 0.043 |
| per-epoch pass rate | 0.506 / 0.584 / 0.573 |
| samples | 267 (89 tasks x 3 epochs, in 2 runs: 89 + 178) |
| tasks passing 3/3, 2/3, 1/3, 0/3 | 30 / 23 / 12 / 24 |
| time-limit hits (7200 s) | 95 (14 of them still passed the verifier) |
| samples that called `submit()` | 172; the verifier scores the container state either way |
| turns truncated at max_tokens | 16 |
| context compactions | 25 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 2,484,635 | 68,277 | 56,436 |
| median | 1,141,056 | 56,592 | 46,315 |
| max | 17,856,708 | 270,264 | 262,474 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 29, mean 40, max 191. Wall time per sample: median 3,355 s, mean 3,822 s, max 7,615 s.
Total run time 2 h 03 min (epoch 1) + 3 h 27 min (epochs 2-3) on 4x MI355X.

Always-failing tasks (0/3): chess-best-move, circuit-fibsqrt, dna-assembly, extract-moves-from-video, feal-
differential-cryptanalysis, filter-js-from-html, gcode-to-text, gpt2-codegolf, install-windows-3.11, make-doom-for-
mips, make-mips-interpreter, mteb-retrieve, path-tracing-reverse, protein-assembly, qemu-alpine-ssh, qemu-startup,
query-optimize, raman-fitting, regex-chess, schemelike-metacircular-eval, train-fasttext, video-processing, winning-
avg-corewars, write-compressor.

## Patches for upstream

None.
