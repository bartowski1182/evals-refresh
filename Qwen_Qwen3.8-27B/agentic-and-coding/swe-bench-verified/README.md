# Qwen/Qwen3.8-27B — SWE-bench Verified

| | |
|---|---|
| Task | `inspect_evals/swe_bench` (500 instances, `princeton-nlp/SWE-bench_Verified` test split, revision `c104f840cc`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker image per instance (Epoch AI, ghcr.io), no internet, test-based grading |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/swe_bench (eval version 5-C) |

**Protocol note.** The model card does not report SWE-bench Verified (it reports SWE-bench Pro with the Claude Code harness). `inspect_evals` solves each instance with its default ReAct agent
(`python`, `bash_session`, `text_editor`, `submit`; 210 s tool timeout) with `--message-limit 200` (about 100 agent
turns; the package default of 30 stops most samples after 13 turns) and a `--time-limit 14400` wall-clock guard (56 samples in flight, what the server's 4M-token KV cache holds with a warm prefix cache once histories average 45k tokens).
One epoch (pass@1 over 500 instances) instead of the repo's usual avg@3. Grading delegates to the official
`swebench` harness (FAIL_TO_PASS and PASS_TO_PASS must all pass). The official leaderboard numbers come
from other scaffolds (SWE-agent, OpenHands, Claude Code, ...) and are not directly comparable. All
in-flight samples share one vLLM server, so wall-clock limits bind harder than against an API endpoint.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task, epochs, sampling params, per-turn max tokens, message limit, concurrency, sandbox count, time limit).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below. The client venv is the `inspect_evals` one with the
  `swe_bench` extra (see Run).
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs, points Docker at the rootless daemon, starts `serve.sh` in the background,
  runs `run_eval.sh`, tears down. **This is the only file you should modify** to match the
  environment you run in.

## Run

### Setup (once per host)

`local_orchestrator.sh` activates two virtualenvs, `~/.venvs/vllm` for the server and `~/.venvs/inspect` for the
Inspect client; create them as below (edit the paths in `local_orchestrator.sh` if you keep them elsewhere).
`server_requirements.txt` and `client_requirements.txt` are the exact `pip freeze` of the environments that
produced the results below (Python 3.12, host details in their headers): installing from them reproduces those
environments; the commented alternative gives an equivalent fresh install with the same pinned versions.

```bash
# server: vLLM 0.29.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.29.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect && source ~/.venvs/inspect/bin/activate
pip install -r client_requirements.txt          # what produced the results; or the current setup, inspect_evals with the
# swebench cap straight from the fork branch (see Patches):
# pip install "inspect-evals[swe_bench] @ git+https://github.com/eldarkurtic/inspect_evals@fix/swe-bench-cap-swebench-below-5" openai
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

Requires Docker with roughly 150 GB free in its image store. **What to expect on the first run:**
Inspect pulls one prebuilt image per instance from `ghcr.io/epoch-research` as containers start
(500 images, 1.2 GB compressed each on average but sharing 33 GB of unique layers; anonymous pulls
work). Pre-pull with `docker pull` to avoid stalling the first samples. Later runs reuse the images.
Containers run with `network_mode: none`.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), message_limit=200, time_limit=14400s, client_timeout=7200, tool_timeout=210s, TP=8 DP=1, epochs=1; 468 instances in one run (concurrency launched at 96 and retuned live to 64 / 40 / 48 / 56 samples) plus the 32 matplotlib instances in a second run at 56`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| resolved rate (pass@1, 1 epoch) | 0.772 (386 of 500 scored instances) |
| stderr (binomial) | 0.0188 |
| samples | 500 (500 instances x 1 epoch, in 2 runs: 468 + 32 instances) |
| message-limit hits (200) | 96 |
| time-limit hits (14400 s) | 0 |
| samples that called `submit()` | 1 |
| turns truncated at max_tokens | 7 |
| sample errors (excluded from the score) | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 3,211,160 | 44,193 | 34,101 |
| median | 2,653,292 | 39,338 | 28,390 |
| max | 11,816,143 | 164,788 | 158,276 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 63, mean 66, max 99. Wall time per sample: median 2,314 s,
mean 3,167 s, max 13,035 s. Total run time about 7 h 55 min on 8x H100 across the runs.

Per repository (resolved / instances): django 183/231, sympy 57/75, sphinx 31/44, matplotlib 26/34, scikit-learn 28/32, astropy 16/22, xarray 19/22, pytest 17/19, pylint 5/10, requests 1/8, seaborn 2/2, flask 1/1.

The 32 matplotlib instances ran separately: their Epoch images contain files owned by UIDs beyond this user's
subuid range, which rootless Docker cannot extract (Docker 29 no longer has the overlay2 `ignore_chown_errors`
option), so copies with those UIDs mapped to root in the one offending layer were loaded instead. `logs/` holds
both `.eval` files.


## Patches for upstream

**Known issue (results affected).** These runs used `inspect_evals`' default Epoch AI images, which keep every
release tag of the project, so the reference fix is readable offline (for example `git show
3.0:django/contrib/auth/validators.py` in django__django-11099). The `*_Qwen3.8-Flash-Next*` SWE-bench Verified
dirs use the official SWE-bench images instead (`-T
image_name_template=swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest`), which are clean. Reported
upstream as [UKGovernmentBEIS/inspect_evals#2632](https://github.com/UKGovernmentBEIS/inspect_evals/issues/2632)
(`inspect_evals` would switch `DEFAULT_IMAGE_TEMPLATE`).

`inspect_evals` declares `swebench>=3.0.15` but its scorer imports `MAP_REPO_VERSION_TO_SPECS` and
`swebench.harness.test_spec`, which swebench 5.0 removed, so a fresh install resolves 5.x and every sample errors at
scoring time (swebench 4.1.0 works). Reported as https://github.com/UKGovernmentBEIS/inspect_evals/issues/2545; the
fix caps the `swe_bench` extra at `swebench<5` on branch `fix/swe-bench-cap-swebench-below-5` of
https://github.com/eldarkurtic/inspect_evals, which the client venv installs from (see Setup; it resolves swebench
4.1.0). The results below were produced with inspect_evals 0.20.0 and swebench pinned by hand to 4.1.0, the same
scorer code. Once the fix is released, install the release instead.
