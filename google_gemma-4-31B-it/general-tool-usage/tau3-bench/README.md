# google/gemma-4-31B-it — tau3-bench

| | |
|---|---|
| Task | `inspect_harbor/sierra_research_tau3_bench` (375 tasks: airline 50, retail 114, telecom 114, banking_knowledge 97; Harbor hub package digest `sha256:a57304f6…`; the tasks are generated from the adapter fork that carries the image fix, see Patches) |
| Category | general-tool-usage (subclass: tool-use) |
| Exec tier | `agent` — two Docker containers per sample (task image plus the tau2-bench MCP runtime sidecar), domain tools and the user conversation through MCP, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/Google/gemma-4-31B-it |
| Task source | https://hub.harborframework.com/datasets/sierra-research/tau3-bench |

**Protocol note.** Neither model card reports tau3-bench. The task set is tau2-bench >= 1.0 with the
banking_knowledge domain (the voice modality of tau3 is not part of the package). Inspect's generic ReAct agent
(`bash`, `python`, `update_plan`, plus the task's MCP tools: `start_conversation`, `send_message_to_user`,
`end_conversation` and the domain tools) drives the conversation; the user simulator and the natural-language
assertion judge (91 tasks) are LLM calls made by the runtime and the verifier through an OpenAI-compatible
endpoint, both Qwen/Qwen3.8-27B here (a fixed simulator as on the tau-bench leaderboards; the leaderboard uses
gpt-5.2 at low reasoning effort). A second vLLM server (GPUs 4-7, TP=4) serves Qwen/Qwen3.8-27B as the user simulator, the model under test runs on GPUs 0-3 (TP=4). One epoch (pass^1) instead of the repo's usual avg@3; a uniform
`--time-limit 3600` (Harbor's agent budget) as a runaway guard. The numbers are not comparable to taubench.com;
the native Inspect re-implementation of the same task set is in `../tau2-bench`.

## Files

- `assets/generate_config.json` — Inspect generate config that cannot be given as a CLI flag; here it
  carries the per-request `chat_template_kwargs` that enable thinking.
- `assets/tool_chat_template_gemma4.jinja` — vLLM's Gemma 4 chat template (from the recipe), used by `serve.sh`.
- `assets/serve_user_model.sh` — vLLM server for the Qwen/Qwen3.8-27B user simulator on a second port (recipe flags), started by `local_orchestrator.sh`.
- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (dataset copy, epochs, sampling params, per-turn max tokens, concurrency, sandbox count, time limit,
  simulator/judge endpoint variables).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below. The client venv is the `inspect-harbor` one.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs and TP/DP, points Docker at the rootless daemon, exports the host address containers can
  reach, starts the server(s) in the background, runs `run_eval.sh`, tears down. **This is the only file you
  should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and `client_requirements.txt`
are the exact `pip freeze` of the environments that produced the results below (Python 3.12, host details in their
headers): installing from them reproduces those environments; the commented alternative gives an equivalent fresh
install with the same pinned versions.

```bash
# server: vLLM 0.29.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.29.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install -r client_requirements.txt          # what produced the results (a local 0.7.6-based checkout); or the current setup,
# inspect_harbor with MCP-server support straight from the fork branch of PR #186 (see Patches):
# pip install "inspect-harbor @ git+https://github.com/eldarkurtic/inspect_harbor@feat/mcp-servers-and-compose-sidecars" mcp httpx2 "openai==3.14.0"
```

`inspect-harbor` pulls in `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible providers need
`openai>=3`; installing `openai==3.14.0` last overrides the pin (pip prints a conflict warning that is harmless
here because Inspect's agent path never imports `litellm`). `httpx2` is imported by `inspect-ai` but not declared.

Model weights and datasets are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

The task set is generated from the adapter fork that carries the image fix of harbor#3401 (see Patches), with
tau2-bench checked out at the commit the images pin; `run_eval.sh` reads it from `DATASET_PATH`:

```bash
git clone -b fix/tau3-bench-image-deps https://github.com/eldarkurtic/harbor.git ~/github/eldarkurtic/harbor
git clone https://github.com/sierra-research/tau2-bench.git ~/github/eldarkurtic/tau2-bench   # next to the harbor clone
git -C ~/github/eldarkurtic/tau2-bench checkout b7ea9074c1cba482b30687fecdb5c8425fd6f619
cd ~/github/eldarkurtic/harbor/adapters/tau3-bench && uv run tau3-bench --output-dir ~/.cache/harbor/local/tau3-bench-cc9e4b7c --overwrite
```

Docker (with the compose plugin) must be reachable by your user. `local_orchestrator.sh` exports `DOCKER_HOST` for
a rootless daemon (`unix:///run/user/<uid>/docker.sock`, image store under `~/.local/share/docker`); with a
rootful daemon drop that export. Check the image-store disk space quoted below before the first run. Rootless
Docker needs a runc runtime with `NoNewKeyring` in `daemon.json` (one kernel key per container otherwise
exhausts a user's quota of 200) and cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

Requires Docker and the inspect_harbor branch described under Patches. Each task builds two images (about
1.9 GB each, layers shared across tasks).

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=64, chat_template_kwargs={enable_thinking: true}, max_tokens=65536 (per turn), time_limit=3600s, client_timeout=7200, tool_timeout=300s, TP=4 DP=1 on GPUs 0-3, epochs=1; user simulator and assertion judge openai/Qwen/Qwen3.8-27B on a second server (TP=4, GPUs 4-7; simulator at reasoning effort low, as the package defaults); 375 tasks in one run at 56 samples in flight`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| pass^1 over all domains (task-weighted mean, 375 tasks) | 0.592 (222 of 375) |
| stderr (binomial) | 0.0254 |
| pass^1 airline | 0.840 (42 of 50) |
| pass^1 retail | 0.728 (83 of 114) |
| pass^1 telecom | 0.658 (75 of 114) |
| pass^1 banking_knowledge | 0.227 (22 of 97) |
| samples | 375 of 375 tasks, 1 epoch |
| time-limit hits (3600 s) | 20 |
| turns truncated at max_tokens | 7 |
| sample errors (excluded from the score) | 0 |

| agent tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 807,087 | 6,819 | 4,295 |
| median | 200,664 | 4,359 | 3,125 |
| max | 10,985,075 | 73,959 | 65,892 |

Agent tokens cover the model under test only; the user simulator and the assertion judge are called by the task
containers and are not in the Inspect log. Input tokens are cumulative over agent turns (history is resent each
turn). Agent turns per sample: median 17, mean 39, max 504. Wall time per sample: median 236 s,
mean 706 s, max 3,616 s. Total run time 1 h 58 min on 8x H100.

## Patches for upstream

Two upstream defects had to be fixed to run this package faithfully:

- `inspect_harbor` (up to 1.0.0) ignores `[[environment.mcp_servers]]` and leaves the main service of a task's
  docker-compose.yaml incomplete, so the agent gets no domain tools and the sidecar's `/logs/agent` is not shared
  with the verifier. Fix: https://github.com/meridianlabs-ai/inspect_harbor/pull/186 (branch
  `feat/mcp-servers-and-compose-sidecars` of https://github.com/eldarkurtic/inspect_harbor), which wires each MCP
  server as Inspect MCP tools through a stdio bridge started inside the service that hosts it, completes the main
  service from the task's Dockerfile and `[environment.env]`, marks it as Inspect's default sandbox, and turns
  Harbor's log-directory bind mounts into named volumes shared by all services. The client venv installs that branch
  (see Setup); once released, pin the release.
- The task images clone tau2-bench's moving `main` at build time and install it without `websockets`, which
  `import tau2` now needs, so the verifier and the reference solution crash in the main container. Fix:
  https://github.com/harbor-framework/harbor/pull/3401 (branch `fix/tau3-bench-image-deps` of
  https://github.com/eldarkurtic/harbor): the adapter's Dockerfile templates pin the tau2-bench checkout
  (`TAU2_BENCH_REF`, commit `b7ea9074`) and add `websockets`. The task set is generated from that branch with the
  adapter (see Setup); once merged and the hub package republished, switch back to
  `inspect_harbor/sierra_research_tau3_bench -T ref=<digest>`.

The results below were produced with a local checkout of inspect_harbor 0.7.6 carrying the same MCP change and a
patched local copy of the hub package with the same image changes; the setup above was verified afterwards with
`--solver inspect_harbor/oracle` on retail tasks. Task content, tools and verifiers are unchanged by either fix.
