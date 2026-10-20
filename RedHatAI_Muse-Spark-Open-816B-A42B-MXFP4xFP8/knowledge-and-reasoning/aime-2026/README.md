# RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 — AIME 2026

| | |
|---|---|
| Task | `inspect_evals/aime2026` (30 problems (AIME I + II), integer answers) |
| Category | knowledge-and-reasoning (subclass: math-reasoning) |
| Exec tier | `none` — no Docker, no sandbox, no tools |
| vLLM recipe | https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B |

Answers are matched numerically after stripping `\boxed{}`; the prompt asks for `ANSWER: $ANSWER` on the last line.
Dataset pinned by inspect_evals 0.20.0 at revision `79037aebdb`. One epoch (pass@1), the repo default: with 30
problems a single epoch has a standard error of several points. The Muse Spark Open card does not report AIME 2026,
so there is no reference number.
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
  (task, epochs, sampling params, reasoning effort, max tokens, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM image and of the Inspect venv that
  produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not
  present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM podman
  image (GPUs 0-7, port 8000, one TP=8 replica, the local checkpoint and the task dir mounted), waits for the server,
  activates the client venv, runs `run_eval.sh`, stops the container. **This is the only file you should modify** to match the environment you run in.

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
# client: Inspect
uv venv ~/.venvs/inspect --python 3.12 && source ~/.venvs/inspect/bin/activate
uv pip install "inspect-ai==0.3.263" "inspect-evals==0.20.0" "openai==3.14.0"
```

If the image's vLLM is an editable install of a source checkout, `local_orchestrator.sh` mounts that checkout (`VLLM_SRC`) at
`/workspace/vllm`, where the image expects it.

Whichever image you use, check `logs/serve.log` at start-up: the expert layers must load with
`CompressedTensorsW4A8Mxfp4MoEMethod` (MXFP4 weights, FP8 activations). A vLLM without vllm-project/vllm#59815 loads
the checkpoint without error but ignores its FP8 activation scheme and runs the experts with BF16 activations.

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`, 453 GB; pre-download it with `hf download`
and keep it on local disk); set `MODEL_PATH` to the HF id to let vLLM fetch it. Export `HF_TOKEN` in your shell before
running (never in the repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 5`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5), with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=1.0, top_k=64, reasoning_effort=xhigh, max_tokens=131072, max_connections=128, client_timeout=7200, TP=8 DP=1 (expert parallel), epochs=1`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

## Patches for upstream

None.
