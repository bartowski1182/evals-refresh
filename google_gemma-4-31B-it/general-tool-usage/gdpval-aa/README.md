# google/gemma-4-31B-it — GDPval-AA

| | |
|---|---|
| Task | `inspect_evals/gdpval` (220 tasks, the open GDPval gold set, HF `openai/gdpval` revision `a3848a2a…` pinned by inspect_evals 0.20.0) |
| Category | general-tool-usage (subclass: pro-work) |
| Exec tier | `agent` — one Docker sandbox per sample (inspect_evals's GDPval image: LibreOffice plus the paper's package list, about 10 GB), bash/python tools, input files staged in, deliverables collected from `deliverable_files/` |
| vLLM recipe | https://recipes.vllm.ai/Google/gemma-4-31B-it |
| Task source | https://huggingface.co/datasets/openai/gdpval |

**Protocol note.** The Gemma model card does not report GDPval. Artificial Analysis's GDPval-AA runs these 220 tasks with its own agent (shell and web
browsing) and reports an Elo from blind pairwise LLM-judge comparisons between models, which cannot be reproduced
for a single model. `inspect_evals/gdpval` produces the deliverables with Inspect's default tool loop (`bash`,
`python`, 180 s tool timeout, no code-interpreter or search API) and records the final message; it has no grader
of its own. Scoring is a separate phase, described under Results. 98 of the 220 tasks have no input files; the
others come with spreadsheets, PDFs and Word documents, and 25 include images, audio or video that a text-only
model can only inspect through tools. One epoch (pass@1) instead of the repo's usual avg@3. All in-flight samples
share one vLLM server (one TP=8 replica so the prefix cache stays local). The numbers are therefore not comparable
to the GDPval-AA leaderboard or to OpenAI's GDPval leaderboard.

## Files

- `assets/generate_config.json` — Inspect generate config that cannot be given as a CLI flag; here it
  carries the per-request `chat_template_kwargs` that enable thinking.
- `assets/tool_chat_template_gemma4.jinja` — vLLM's Gemma 4 chat template (from the recipe), used by `serve.sh`.
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

`temperature=1.0, top_p=0.95, top_k=64, chat_template_kwargs={enable_thinking: true}, max_tokens=65536 (per turn), message_limit=200, time_limit=7200s, client_timeout=7200, tool_timeout=180s, TP=8 DP=1, epochs=1; 220 tasks in one run at 56 samples in flight`

the following run stats are obtained (scores are in the scoring phase below):

| metric | value |
|---|---|
| samples | 220 of 220 tasks, 1 epoch |
| samples that wrote at least one deliverable file | 220 |
| deliverable files per sample | median 1, max 12 |
| time-limit hits (7200 s) | 0 |
| message-limit hits (200) | 0 |
| turns truncated at max_tokens | 0 |
| context compactions | 0 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 70,575 | 8,651 | 3,268 |
| median | 41,546 | 7,744 | 2,966 |
| max | 742,921 | 33,139 | 13,009 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 9, mean 10, max 40. Wall time per sample: median 154 s,
mean 175 s, max 648 s. Total run time 0 h 15 min on 8x H100.

Samples with a deliverable file / tasks per sector: Health Care and Social Assistance 25/25, Government 25/25, Professional, Scientific, and Technical Services 25/25, Real Estate and Rental and Leasing 25/25, Manufacturing 25/25, Finance and Insurance 25/25, Wholesale Trade 25/25, Information 25/25, Retail Trade 20/20.

Scoring phase: pending.

## Patches for upstream

None.
