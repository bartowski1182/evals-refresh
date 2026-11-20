> **Superseded.** This directory was the first pass for the DSpark speculator. The benchmark was
> re-scoped to `RedHatAI/GLM-5.3-speculator.dflash2-preview`, whose results are in
> `RedHatAI_GLM-5.3-speculator.dflash2-preview/agentic-and-coding/terminal-bench-2.1/`. No full run was
> completed from this directory: only two 2-task smoke runs (both 2/2 passed, canary healthy), and those were
> on vLLM 0.32.0 with the PIECEWISE workaround for vLLM #63587. The scripts here are left as the DSpark
> variant of the same protocol.

# RedHatAI/GLM-5.3-speculator.dspark — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task (Docker Hub image named in the task's `task.toml`), bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/zai-org/GLM-5.3 |
| Target model | https://huggingface.co/zai-org/GLM-5.3 (FP8, 78-layer `GlmMoeDsaForCausalLM`, ~756 GB) |
| Draft (speculator) | https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark (DSpark, ~2.5 GB, 3-layer Qwen3-style backbone, `num_speculative_tokens=8`) |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

This directory measures draft-model acceptance for DSpark on real agent trajectories. It is not a leaderboard
entry. `serve.sh` runs the target with the DSpark draft model and with `--per-request-spec-decode-metrics
detailed`. `run_eval.sh` passes Inspect's `--log-model-api`, so the log keeps the raw response of every model
call. `analyze_spec_decode.py` reads those responses offline.

The task, agent and generation settings match the sibling
`RedHatAI_GLM-5.3-MXFP4xFP8/agentic-and-coding/terminal-bench-2.1` directory (same model family, same task pin):

- Inspect's ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context
  compaction), not the Claude Code 2.1.207 harness used by the GLM-5.3 model card (88.2) or the Terminus 2 agent
  used by the leaderboard.
- One uniform `--time-limit 7200` instead of Harbor's per-task agent timeouts.
- Containers run without Harbor's CPU cap (`override_cpus=0`) and with `inspect_harbor`'s 6 GB memory floor.
- Per-turn `max_tokens` 65536, sampling from the checkpoint's `generation_config.json` (temp 1.0, top_p 0.95; no
  top_k, unlike Qwen) and reasoning effort `max` (GLM-5.3's highest and default level).
- Epochs: 1 (`pass@1`).

The only difference from that sibling directory is the serving stack: the full-precision FP8 `zai-org/GLM-5.3`
checkpoint (~756 GB) instead of the MXFP4xFP8 quantized one, one TP=8 replica, and DSpark speculative decoding
against `RedHatAI/GLM-5.3-speculator.dspark` instead of MTP. Speculative decoding changes what generation costs,
not the task, the scaffold or the sampling. This directory reports the acceptance numbers, with the score next
to them.

## Files

- `serve.sh` runs the target with vLLM and the DSpark draft model. The vLLM serve arguments live here: parallelism,
  context length, KV-cache dtype, parsers, `--speculative-config` and `--per-request-spec-decode-metrics`.
- `run_eval.sh` runs the Inspect task against that server. The Inspect parameters live here: task and package
  digest, epochs, sampling, per-turn max tokens, concurrency, sandbox count, time limit and `--log-model-api`.
- `analyze_spec_decode.py` collates the raw responses out of the `.eval` logs into a per-call CSV and computes
  the acceptance analysis (MAL, per-position rates, thinking against response, slices, outliers). It is the one
  file beyond the standard set, and the deliverable of this directory.
- `server_requirements.txt` and `client_requirements.txt` are `pip freeze` of the vLLM and Inspect venvs that
  produced the results below. The client venv is the `inspect-harbor` one (see Run).
- `local_orchestrator.sh` runs everything on one host. It checks the GPU reservation, activates the venvs, sets
  the HF cache, starts `serve.sh` in the background, runs `run_eval.sh` and tears down. Modify this file only,
  to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh`
activates; if you keep the environments elsewhere, change the paths in both places.

```bash
# server: vLLM 0.32.0, the first release with --per-request-spec-decode-metrics and DSpark support
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.32.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "inspect-harbor==1.0.0" "inspect-ai==0.3.283" httpx2 "openai==3.14.0"
```

`inspect-harbor` pulls in `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible providers need
`openai>=3`. Installing `openai==3.14.0` last overrides the pin. pip prints a conflict warning, which is
harmless here because Inspect's agent path never imports `litellm`. `httpx2` is imported by `inspect-ai` but
not declared.

Both models are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`). The target is
~756 GB in 141 safetensors shards; the speculator is ~2.5 GB. Both are public and un-gated. Export `HF_TOKEN` in
your shell before running (never in the repo) to avoid anonymous rate limits. huggingface_hub 2.x keeps large
files in a content-addressed store under `<HF_HOME>/hub/blobs/` and leaves each model's `blobs/<sha>` entries as
symlinks into it, so `du -sh` on a model directory under-reports. Check the files with
`ls -laL <snapshot>/*.safetensors` instead. On this host the cache is on the large `/data` volume
(`HF_HOME=/data/reed-meyerson/hf`), because `/home` does not have ~1 TB free. `local_orchestrator.sh` serves with
`HF_HUB_OFFLINE=1`.

flashinfer 0.7.0.post1 (installed by vLLM) has no matching `flashinfer-cubin` wheel, so the DSA sparse-MLA
kernels are JIT-compiled on first start: `serve.sh` exports `CUDA_HOME=/usr/local/cuda` so the JIT path is
available. The compiled kernels are cached under `~/.cache/flashinfer`, so later starts are fast.
`VLLM_ENGINE_READY_TIMEOUT_S=3600` in `local_orchestrator.sh` raises vLLM's 600 s engine start-up budget, which
loading a ~756 GB checkpoint does not fit in.

Docker (with the compose plugin, v2.x or later) must be reachable by your user. On this host the daemon is
**rootful and shared** (other tenants' containers are running on it), so `local_orchestrator.sh` leaves
`DOCKER_HOST` unset; the image store lives under `/var/lib/docker`. The compose plugin is installed user-locally
in `~/.docker/cli-plugins/docker-compose`. The eval needs roughly 70 GB free in the Docker image store.

The 89 task images are Docker Hub images named in each task's `task.toml`. Docker Hub limits anonymous pulls per
IP, so pre-pull them ahead of the run: `docker pull mirror.gcr.io/<repo>:<tag> && docker tag mirror.gcr.io/<repo>:<tag>
<repo>:<tag> && docker rmi mirror.gcr.io/<repo>:<tag>` (the `mirror.gcr.io` pull-through cache is not rate limited
and needs no login). Harbor downloads the 89 task definitions into `~/.cache/inspect_harbor/` on first use.

### Run the eval

`local_orchestrator.sh` refuses to start without a `canhazgpu` reservation, because this class of host is
shared and no run may touch the GPUs unreserved:

```bash
canhazgpu run --gpus 8 --note "TB-2.1 GLM-5.3 DSpark" -- ./local_orchestrator.sh
inspect view --log-dir ./logs
```

Smoke-test first with 2 tasks and 2 concurrent samples:
`canhazgpu run --gpus 8 --note "TB-2.1 GLM-5.3 DSpark smoke" -- bash -c 'EPOCHS=1 MAX_CONNECTIONS=2 MAX_SANDBOXES=2 ./local_orchestrator.sh --limit 2'`

Concurrency is 32 in flight, not all 89 tasks at once: Terminal-Bench histories reach ~30-60k tokens, and 32 x
262,144 stays inside the KV pool (see "GPU KV cache size" in `logs/serve.log`), while the run still proceeds in
~3 waves. Keep `MAX_SANDBOXES` equal to `MAX_CONNECTIONS`.

### Analyse the run

```bash
# from the client venv, because collate needs inspect_ai; analyze alone uses the standard library
python analyze_spec_decode.py logs -o spec_decode_report
# or in two steps, to reuse the CSV across analyses
python analyze_spec_decode.py collate logs -o calls.csv
python analyze_spec_decode.py analyze calls.csv -o spec_decode_report
```

## Speculative decoding instrumentation

The acceptance metrics come from vLLM 0.32.0's experimental per-request reporting, captured by Inspect.

1. `serve.sh` passes `--speculative-config '{"method":"dspark","model":"RedHatAI/GLM-5.3-speculator.dspark",
   "num_speculative_tokens":8}'` to enable the draft model. That is the deployment command on the speculator's
   model card, and 8 tokens per verify step is the speculator's own `block_size` / `speculative_tokens`. The draft
   model consumes auxiliary hidden states from target layers 2, 20, 39, 58 and 75 (1-based), which its config
   carries as `aux_hidden_state_layer_ids`; vLLM resolves the DSpark draft architecture to `Qwen3DSparkModel`
   automatically from the speculator's `config.json`. `serve.sh` also passes `--per-request-spec-decode-metrics
   detailed`, which adds a `metrics.speculative_decoding` object to every chat-completion response. The object
   holds the mean acceptance length, a histogram of how many verify steps accepted exactly j draft tokens, the
   step and token counts, and the ordered per-step arrays that only the `detailed` level emits. Only
   single-sequence requests (`n == 1`) are populated, and Inspect always sends `n == 1`.
2. `run_eval.sh` passes `--log-model-api`, which makes Inspect keep the raw request and response of every model
   call under `ModelEvent.call`. Without the flag Inspect keeps only the first few calls per model plus errors.
   The OpenAI-compatible provider stores the response through `completion.model_dump()`, so the non-standard
   `metrics` field survives.
3. `analyze_spec_decode.py` reads those responses. `collate` walks `sample.events` rather than
   `sample.messages`, because compaction drops early turns from the messages. It numbers the `model` events to
   get a 1-based turn number, and writes one row per call holding the task, epoch, turn, stop reason, token
   usage (input, cache-read, output, reasoning) and every acceptance field. `analyze` then reports:
   - MAL, defined as `1 + accepted_draft_tokens / num_spec_steps`, where the leading 1 is the always-accepted
     bonus token. It reports a pooled value over all steps (micro), where every verify step weighs the same, and
     the unweighted mean of the per-call values (macro), where every request weighs the same, and the per-call
     distribution.
   - Draft acceptance rate, defined as `accepted_draft_tokens / num_draft_tokens`.
   - Per-position acceptance, as two rates. The overall rate is the probability that the draft token at position
     j is accepted, which is the tail of the histogram, `Σ_{i>j} hist[i] / steps`; the overall rates plus 1
     reproduce MAL. The conditional rate is the probability that position j is accepted given that position j-1
     was, `overall[j] / overall[j-1]`, the per-position hazard rate. Both rates are sliced by turn number,
     context length, output length, pass/fail and stop reason. The "exactly j" marginal probabilities stay in
     `report.json` and `acceptance_by_position.csv`.
   - Thinking against response, in token space and without a tokenizer. A completion is one contiguous token
     sequence, and each verify step emits its accepted drafts plus one bonus token, so step spans follow from
     `per_step_accepted` (the `Σ(accepted+1)` sum equals `completion_tokens`). The reasoning block is a prefix
     of the completion, and its token length is `usage.completion_tokens_details.reasoning_tokens`. MAL and both
     per-position rates are therefore reported per region. A step is attributed to the region that holds the
     midpoint of its accepted tokens, and the report states how many steps straddled the boundary.
   - MAL by turn number, by context-length and output-length bins, and by task.
   - Outlier checks, computed two ways: drop the calls above the p95 output-token count, and drop whole samples
     whose total output exceeds its p95.
   - Within-response drift, as the MAL of each decile of position within a response, and MAL for passing against
     failing samples.

   No timing or throughput metric is reported. Tokens per second and wall time depend on the host, the
   concurrency and the serving topology, so they describe only the run that produced them.

Outputs are `report.md`, `report.json` and one CSV per slice family. vLLM labels the per-request field
experimental, so its shape may change between releases.

## Deviations from the prompt's suggested flags

- `--tokenizer-mode glm5_next` does not exist in vLLM 0.32.0 (choices are `auto`, `cohere`, `deepseek_v32`,
  `deepseek_v4`, `hf`, `inkling`, `kimi_k3`, `mistral`). The checkpoint's `tokenizer_config.json` specifies
  `tokenizer_class = TokenizersBackend` with a GLM-5 `chat_template.jinja`, which the default `auto` resolves
  (verified: it renders `[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>…`). `serve.sh` sets
  `--tokenizer-mode auto` explicitly and the Rust frontend is left at its default (`VLLM_USE_RUST_FRONTEND`
  unset).
- `--disable-flashinfer-autotune` (used by the GLM-5.3-Flash sibling dir) does not exist in 0.32.0 either; the
  equivalent is `--no-enable-flashinfer-autotune`, and the default (auto) is used here.
- `--kv-cache-dtype fp8_e4m3` is written as `--kv-cache-dtype fp8`, which vLLM maps to `fp8_e4m3` and which is
  the spelling used by the GLM-5.3 siblings.
- `analog to the Qwen dir`, `--reasoning-parser glm47` + `--tool-call-parser glm47` + `--enable-auto-tool-choice`
  are GLM-5.x's parsers in 0.32.0 (`glm47_moe_reasoning_parser`, `glm47_moe_tool_parser`).

## Patches for upstream

None are needed for the dataset. Two tooling notes, both inherited from the sibling directories:

1. **Context compaction can fail, at either threshold, and the window it uses is racy.** The `vllm` provider
   learns the served `max_model_len` from `/v1/models` during the first `generate()`, but the ReAct agent's
   compaction resolves its threshold (0.9 x window) on the first `compact_fn`, which can run before that call.
   `CompactionEdit` only clears old tool results and thinking blocks, keeping the last 3 tool-use pairs and all
   conversation text, so a trajectory whose own messages exceed the threshold cannot be reduced and the sample
   dies with `RuntimeError: Compaction insufficient`. Inspect records that as a sample error, not a failure.
2. **`--per-request-spec-decode-metrics` is experimental** in vLLM 0.32.0. The response shape may change, and
   only single-sequence (`n == 1`) requests are populated.

<!-- RESULTS -->
