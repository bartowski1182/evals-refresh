# Qwen/Qwen3.8-27B — GDPval-AA

| | |
|---|---|
| Task | `inspect_evals/gdpval` (220 tasks, the open GDPval gold set, HF `openai/gdpval` revision `a3848a2a…` pinned by inspect_evals 0.20.0) |
| Category | general-tool-usage (subclass: pro-work) |
| Exec tier | `agent` — one Docker sandbox per sample (inspect_evals's GDPval image: LibreOffice plus the paper's package list, about 10 GB), bash/python tools, input files staged in, deliverables collected from `deliverable_files/` |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://huggingface.co/datasets/openai/gdpval |

**Protocol note.** The Qwen model card does not report GDPval. Artificial Analysis's GDPval-AA runs these 220 tasks with its own agent (shell and web
browsing) and reports an Elo from blind pairwise LLM-judge comparisons between models, which cannot be reproduced
for a single model. `inspect_evals/gdpval` produces the deliverables with Inspect's default tool loop (`bash`,
`python`, 180 s tool timeout, no code-interpreter or search API) and records the final message; it has no grader
of its own. Scoring is a separate phase, described under Results. 98 of the 220 tasks have no input files; the
others come with spreadsheets, PDFs and Word documents, and 25 include images, audio or video that a text-only
model can only inspect through tools. One epoch (pass@1) instead of the repo's usual avg@3. All in-flight samples
share one vLLM server (one TP=8 replica so the prefix cache stays local). The numbers are therefore not comparable
to the GDPval-AA leaderboard or to OpenAI's GDPval leaderboard.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, per-turn max tokens, message and time limits, concurrency, sandbox count).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs and TP/DP, points Docker at the rootless daemon, starts `serve.sh` in the background,
  runs `run_eval.sh`, tears down. **This is the only file you should modify** to match the
  environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below (Python 3.12,
host details in their headers): installing from them reproduces those environments; the commented alternative gives an
equivalent fresh install with the same pinned versions.

```bash
# server: vLLM 0.29.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.29.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "inspect-evals[gdpval]==0.20.0" openai
```

Model weights and datasets are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

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

Requires Docker. Build the sandbox image once before
the first run (`docker build -t gdpval -f <site-packages>/inspect_evals/gdpval/Dockerfile <that dir>`, about
10 minutes); Inspect rebuilds it from the same Dockerfile with cached layers. Each run writes the consolidated
deliverables (one folder per task plus a parquet table) to `<site-packages>/inspect_evals/gdpval/gdpval_hf_upload/<timestamp>_gdpval/`,
the layout OpenAI's grader expects.

## Results

Generation phase, with model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), message_limit=200, time_limit=7200s, client_timeout=7200, tool_timeout=180s, TP=8 DP=1, epochs=1; 220 tasks in one run at 56 samples in flight`

the following run stats are obtained (scores are in the scoring phase below):

| metric | value |
|---|---|
| samples | 220 of 220 tasks, 1 epoch |
| samples that wrote at least one deliverable file | 192 (the others answered in the final message only, which the task allows) |
| deliverable files per sample | median 2, max 86 |
| time-limit hits (7200 s) | 17 |
| message-limit hits (200) | 4 |
| turns truncated at max_tokens | 4 |
| context compactions | 0 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 1,608,388 | 59,015 | 36,470 |
| median | 818,256 | 53,544 | 33,164 |
| max | 10,114,146 | 172,012 | 93,416 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 21, mean 29, max 99. Wall time per sample: median 2,983 s,
mean 3,407 s, max 7,201 s. Total run time 4 h 16 min on 8x H100.

Samples with a deliverable file / tasks per sector: Health Care and Social Assistance 23/25, Government 25/25, Professional, Scientific, and Technical Services 23/25, Real Estate and Rental and Leasing 23/25, Manufacturing 21/25, Finance and Insurance 20/25, Wholesale Trade 24/25, Information 16/25, Retail Trade 17/20.

Scoring phase: pending.

## Patches for upstream

None.
