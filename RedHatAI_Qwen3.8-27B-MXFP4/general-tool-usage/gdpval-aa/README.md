# RedHatAI/Qwen3.8-27B-MXFP4 — GDPval-AA

| | |
|---|---|
| Task | `inspect_evals/gdpval` (220 tasks, the open GDPval gold set, HF `openai/gdpval` revision `a3848a2a…` pinned by inspect_evals 0.20.0) |
| Category | general-tool-usage (subclass: pro-work) |
| Exec tier | `agent` — one Docker sandbox per sample (inspect_evals's GDPval image: LibreOffice plus the paper's package list, about 10 GB), bash/python tools, input files staged in, deliverables collected from `deliverable_files/` |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B (the base model's; its MXFP4 variant serves this checkpoint with the same flags) |
| Task source | https://huggingface.co/datasets/openai/gdpval |

**Protocol note.** No model card reports GDPval. Artificial Analysis's GDPval-AA runs these 220 tasks with its own
agent (shell and web browsing) and reports an Elo from blind pairwise LLM-judge comparisons between models, which
cannot be reproduced for a single model. `inspect_evals/gdpval` produces the deliverables with Inspect's default tool
loop (`bash`, `python`, 180 s tool timeout, no code-interpreter or search API) and records the final message; it has
no grader of its own. Scoring is a separate phase, to be planned with a local judge identical across the three models
(AGENTS.md, section 8) and described under Results. One epoch (pass@1); one TP=8 replica so the prefix cache stays
local. This dir is one of the three identical copies of the task that make up the Qwen3.8-27B comparison set (`Qwen_Qwen3.8-27B`, `RedHatAI_Qwen3.8-27B-MXFP4`, `RedHatAI_Qwen3.8-27B-MXFP4xFP8`): same task, agent, prompt, sampling, limits, concurrency, epochs (one, pass@1), vLLM release and serving topology, so the three numbers are like for like; only the checkpoint differs. The numbers are not comparable to the GDPval-AA leaderboard or to OpenAI's GDPval leaderboard.

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
  (task, epochs, sampling params, per-turn max tokens, message and time limits, concurrency, sandbox count); identical in the three dirs apart from `MODEL`.
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` inside the vLLM image and of the Inspect venv
  that produce the results, with the host (GPUs, driver, CUDA, Python) in the header. Written after the final run; not present yet.
- `local_orchestrator.sh` — runs everything in your local dev environment: starts `serve.sh` inside the vLLM podman image (GPUs 0-7, port 8000, one TP=8 replica, the HF cache and the task dir mounted), points Docker at the rootless daemon, waits for the server, activates the client venv, runs `run_eval.sh`, stops the container.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Pull the server image and create the client environment as below. The commands use the image reference and the venv
path that `local_orchestrator.sh` uses; if you keep them elsewhere, change them there. Once the results are in,
`server_requirements.txt` (`pip freeze` inside the image) and `client_requirements.txt` hold the exact packages that
produced them (host details in their headers); create the client environment from its file instead. Use the same image
and client environments for the three dirs of the comparison set.

```bash
# server: the vLLM 0.32.0 podman image, docker.io/vllm/vllm-openai:v0.32.0 pinned by digest (IMAGE in
# local_orchestrator.sh), the same image for the three dirs. 0.32.0 is the first release with the dense MXFP4-weight /
# FP8-activation scheme the MXFP4xFP8 checkpoint needs; the humming-kernels dependency of its CUDA wheels provides the
# only MXFP4 GEMM of that release that takes FP8 activations (CUDA, SM75+), so the comparison set needs an NVIDIA host.
# GPU access from the container goes through the NVIDIA Container Toolkit's CDI spec (generated once per host, as root).
podman pull docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
# client: Inspect + inspect_evals with the gdpval extra
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install "inspect-ai==0.3.263" "inspect-evals[gdpval]==0.20.0" "openai==3.14.0"
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

Build the sandbox image once before the first run (`docker build -t gdpval -f <site-packages>/inspect_evals/gdpval/Dockerfile <that dir>`,
about 10 minutes); Inspect rebuilds it from the same Dockerfile with cached layers. Each run writes the consolidated
deliverables (one folder per task plus a parquet table) to
`<site-packages>/inspect_evals/gdpval/gdpval_hf_upload/<timestamp>_gdpval/`, the layout OpenAI's grader expects.
`local_orchestrator.sh` serves the checkpoint from the Hugging Face Hub (`MODEL`) through the host's HF cache (`HF_HOME`,
mounted into the container): download it there once with `hf download` and export `HF_HUB_OFFLINE=1`, or set
`MODEL_PATH` there to serve a local copy instead. Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub downloads are rate-limited.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

## Results

Not run yet. `run_eval.sh` records the configuration below; fill the tables from the final `.eval` log (AGENTS.md,
section 5), with sample errors counted as failures.

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), message_limit=200, time_limit=7200s, client_timeout=7200, tool_timeout=180s, kv_cache_dtype=bfloat16, TP=8 DP=1, epochs=1, 56 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| | |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| | | | |

Scoring phase: not planned yet.

## Patches for upstream

None.
