# RedHatAI/GLM-5.3-Flash-NVFP4 — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task (Docker Hub), bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3-Flash |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The GLM-5.3-Flash card reports Terminal-Bench 2.1 = 84.3 with a Claude Code 2.1.207 scaffold
(temperature 1.0, top_p 1, 6 h timeout). `inspect_harbor` instead solves every task with Inspect's generic ReAct
agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction), so the number
below is **not comparable** to that card figure or to the Terminus 2 leaderboard. Harbor's per-task agent timeouts
(600 s to 12000 s, most 900 s) are not applied; a single `--time-limit 7200` is used for every task. Containers run
without Harbor's per-task CPU cap (`override_cpus=0`) and with `inspect_harbor`'s 6 GB memory floor; the tasks have
public network by design. One epoch (pass@1). Checkpoint: NVFP4 (mixed precision — W4A4 on most MoE experts, FP8 on
the last expert layer and the MTP module), compressed-tensors format (LLM Compressor), base `zai-org/GLM-5.3-Flash`,
revision `18d55bf`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtype, MTP speculative decoding, reasoning/tool parsers, text-only mode), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here (task and dataset pin,
  epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
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

# client: Inspect + inspect_harbor (stock 1.0.0)
uv venv ~/.venvs/glm-inspect-harbor --python 3.12 && source ~/.venvs/glm-inspect-harbor/bin/activate
uv pip install -r client_requirements.txt   # or: uv pip install "inspect-ai==0.3.273" "inspect-harbor==1.0.0" httpx2 "openai==3.14.0"
```

Docker with the compose plugin (v2.x or later) is required. The results below were produced with the host's system
(rootful) Docker 25 daemon (overlay2 image store on local NVMe); the invoking user is in the `docker` group, so
`local_orchestrator.sh` leaves `DOCKER_HOST` at its default. Inspect pulls a task's prebuilt image before its first
sample if it is missing (serially, before any sample starts; ~15 min for the 89 images).

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`); unset it to let vLLM fetch
`RedHatAI/GLM-5.3-Flash-NVFP4` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running (never in
the repo); the Harbor task files are fetched from GitHub at the pinned commit.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k unset, reasoning_effort=max, max_tokens=65536 (per turn), max_connections=89, max_sandboxes=89, time_limit=7200s, client_timeout=7200, override_cpus=0, TP=8 DP=1 with MTP (num_speculative_tokens=5), kv_cache_dtype=fp8, epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (pass@1, 1 epoch) | **0.775 (69 of 89)** |
| stderr | 0.044 |
| samples | 89 (89 tasks × 1 epoch) |
| sample errors | 0 |
| time-limit hits (7200 s) | 5 (extract-moves-from-video, make-doom-for-mips, make-mips-interpreter, regex-chess, train-fasttext) |
| turns truncated at max_tokens | 0 |
| samples that fetched their own task files (solution/tests) | 0 (audited) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 8,878,298 | 123,242 | 101,684 |
| median | 1,798,080 | 55,716 | 46,902 |
| max | 355,135,856 | 909,876 | 831,242 |

Input tokens are cumulative over agent turns (history is resent each turn). Assistant turns per sample: median 34,
mean 77, max 2,071. Total run time 2:01:20 on 8× B200 (one TP=8 replica, 89 concurrent samples). No solution leaks:
every outbound request was a legitimate task dependency (upstream project source, datasets, fonts, models); none
fetched the Terminal-Bench benchmark's own task files.

Failed tasks (scored 0): bn-fit-modify, code-from-image, dna-assembly, dna-insert, extract-elf,
extract-moves-from-video, filter-js-from-html, gcode-to-text, llm-inference-batching-scheduler, make-doom-for-mips,
make-mips-interpreter, mteb-retrieve, pytorch-model-cli, qemu-alpine-ssh, qemu-startup, raman-fitting, sam-cell-seg,
sanitize-git-repo, train-fasttext, video-processing.

## Patches for upstream

None (dataset/task). Known tooling behavior, documented for a future run: Inspect's `vllm` provider does not learn
the served context window (262,144) in time for the ReAct agent's compaction threshold, which resolves at sample
start, so samples fall back to Inspect's 128,000 default context window for compaction decisions. This is identical
across models evaluated this way (the served model name is not in Inspect's registry) and does not affect
comparability between them; it only makes compaction more conservative than the model's true 262k window. No
model-specific vLLM fork is needed: the stock nightly (commit faf417b75) loads and serves this checkpoint.
