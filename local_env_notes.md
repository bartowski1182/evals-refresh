# Environment notes

Requirements and pitfalls of the hosts these evals run on, written to apply to any Linux GPU server (this file is
referenced and imported from `AGENTS.md`). Nothing in it should be needed to understand the task directories or
their READMEs.

Facts about one particular machine (its paths, user names, installed versions, other tenants) do not belong here.
Keep them in `local_env_notes.local.md` next to this file, which git ignores, and read that file too if it exists.

## Serving

- `local_orchestrator.sh` activates the vLLM venv, starts `serve.sh` in the background, polls `/health`, activates
  the Inspect venv, runs `run_eval.sh "$@"`, and kills the server. Dirs that source `tools/vllm_server.sh` pick the
  server environment at run time instead: `VLLM_VENV=<venv dir>` or `VLLM_IMAGE=<podman image>` (not
  both; with neither, the orchestrator's `DEFAULT_VLLM_IMAGE`), and `INSPECT_VENV=<venv dir>` for the client. The module's podman mode is the NVIDIA
  pattern below; ROCm hosts (other devices, `numactl`) are not covered yet. `vllm_freeze` prints `pip freeze` of the
  selected environment for `server_requirements.txt`.
- Serving convention: `TP=2`, `DP=4` (four replicas behind one port on all 8 GPUs) for ~30B bf16 models on 80 GB
  GPUs. Adjust per model size; a 27–31B bf16 model needs ~26–31 GB per GPU at TP=2. On 288 GB GPUs (MI355X), a 27B
  MXFP4 checkpoint (about 19 GB) runs at `TP=1`, one replica per GPU.
- Keep checkpoints and the HF caches (`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`) on fast local disks. Network
  volumes (NFS, cloud block storage) can cap uncached reads near 100 MB/s, so loading a 100+ GB checkpoint takes tens
  of minutes. `serve.sh` inherits the cache variables from the shell.
- The HF Hub rate-limits anonymous API calls per IP address (429), and many engines starting at once can hit it and
  abort model loading. Pre-download the weights and serve with `HF_HUB_OFFLINE=1`.
- Qwen4Exp (Qwen3.8-Flash-Next) keeps its 51B n-gram table in pinned host memory. While a server loads, every DP
  worker holds a private host copy (about 136 GB each for the BF16 table), so two DP=4 servers starting together need
  about 1.1 TB of RAM and evict the page cache. Start the servers one after the other.
- On a shared host, check `nvidia-smi` (`amd-smi` on AMD) before claiming GPUs. Stop processes by exact PID:
  `pgrep -f`/`pkill -f` patterns can match other users' processes or your own shell.
- `amd-smi` can number GPUs differently from HIP (`HIP_VISIBLE_DEVICES`). Once, with HIP devices 0 and 1 free,
  `amd-smi` showed GPUs 0 and 3 idle. Find the free HIP devices from the `HIP_VISIBLE_DEVICES` of the running servers
  (e.g. `podman inspect` on their containers), not from the `amd-smi` index.
- Multi-turn agent tasks need every turn of a sample on one prefix cache. When one large-TP replica is not possible
  (see TP>1 under AMD below), split the tasks into shards (`-T dataset_task_names=[...]`) and serve each shard from its
  own DP=1 server (own `PORT` and `LOG_DIR`). DeepSWE (~120k-token histories) on DP=4 behind one port cached each
  history on several replicas, which caused evictions: model calls took ~85 s and 40% of samples hit the 14400 s time
  limit. Sharded, it got ~97% prefix-cache hits and ~20 s per call at 4-7 samples per GPU. Terminal-Bench 2.1 (shorter
  histories, tool-bound) did not need this.
- Muse Spark Open (`meta-models/Muse-Spark-Open-816B-A42B` and its quantized copies) is not in a vLLM release or nightly
  wheel: the model lands with vllm-project/vllm#61879 (draft branch `muse-spark-support`, depends on
  flashinfer-ai/flashinfer#6544), and the recipe points at the image `vllm/vllm-openai:muse-spark-open` (pin it by
  digest). Build the branch's `docker/Dockerfile` (target `vllm-openai`) into a local image, or install the branch into
  a venv with `VLLM_USE_PRECOMPILED=1 uv pip install -e .` (precompiled kernels of the nearest main commit). The
  RedHatAI MXFP4xFP8 checkpoint also needs the compressed-tensors
  MXFP4-weight / FP8-activation MoE path (vllm-project/vllm#59815, merged 2026-10-14, included in the PR branch); the
  start-up log must show `CompressedTensorsW4A8Mxfp4MoEMethod`, otherwise the experts run with BF16 activations. The
  checkpoint is 453 GB: one TP=8 replica with expert parallelism on an 8-GPU H200/B200/B300 node, nothing smaller.
- GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, arch `glm5_next`: hybrid KDA linear attention + NoPE sparse MLA, 320B/18B,
  about 306 GiB of FP8 weights) is in vLLM's model registry from release 0.32.0 (the recipe, older than the release,
  still asks for a nightly or the `vllm/vllm-openai:glm53-flash` image) and needs FlashInfer 0.6.17 or newer for the
  sparse-MLA kernels. On Hopper its KV cache must be BF16: the recipe's `--kv-cache-dtype fp8` is a Blackwell-only
  override ("Hopper runs BF16 KV"), and the Hopper override is `--no-enable-flashinfer-autotune`. The recipe sets
  `VLLM_ENGINE_READY_TIMEOUT_S=3600`. The checkpoint's 1M default window hits the sparse-indexer warm-up OOM of AGENTS.md
  section 9 on H200; cap `--max-model-len`.
- Dense MXFP4-weight / FP8-activation checkpoints (compressed-tensors, e.g. `RedHatAI/Qwen3.8-27B-MXFP4xFP8`) need vLLM
  0.32.0 or later (the `CompressedTensorsW4A8Mxfp4` scheme) and the `humming-kernels` package that the CUDA wheels
  depend on: in 0.32.0 Humming is the only MXFP4 linear kernel that takes FP8 activations, on CUDA SM75+. On ROCm
  (AITER) or without Humming the scheme falls back to MXFP4 activations with a one-line warning and the run is a W4A4
  run (AGENTS.md, section 9). The serve log line `Using HummingMxFp4LinearKernel for MXFP4 GEMM` confirms the path.

- Serving from a podman image on NVIDIA (when the vLLM you need is a branch build): give the container the GPUs
  through the NVIDIA Container Toolkit's CDI spec (`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` once per
  host, then `--device nvidia.com/gpu=all`; `CUDA_VISIBLE_DEVICES` selects GPUs inside), `--ipc host` for NCCL and the
  TP workers, `--pids-limit -1`, `--init`, `--security-opt label=disable` under SELinux, the port published on
  `127.0.0.1`, and the task dir and checkpoint mounted at their host paths so `serve.sh` runs unchanged. Mount a host
  directory on `/root/.cache` to keep the torch.compile and FlashInfer JIT caches across runs, and raise
  `VLLM_ENGINE_READY_TIMEOUT_S` for the first start. Take `server_requirements.txt` from `pip freeze` inside the image.
  The same pattern serves a vLLM release: pull `docker.io/vllm/vllm-openai:v<version>` and pin it by digest (query
  `registry-1.docker.io/v2/vllm/vllm-openai/manifests/v<version>` with an anonymous pull token; the
  `Docker-Content-Digest` header is the digest). To serve a Hub id from inside the container, mount the host's HF cache
  (`HF_HOME`) on `/root/.cache/huggingface` and pass `HF_TOKEN` and `HF_HUB_OFFLINE` through; `podman run -e VAR`
  without a value forwards VAR only when it is set on the host, so unset overrides fall back to `serve.sh`'s defaults.
  A server that the task sandboxes (Docker) must reach, such as the tau3-bench user simulator, has to be published on
  all interfaces (`-p PORT:PORT`), not on 127.0.0.1: the sandboxes call the host's primary IP. With rootless podman and
  rootless Docker on one host, podman serves the models and Docker runs the sandboxes; they do not share images.

## Serving on AMD (ROCm)

- vLLM's ROCm wheels (https://wheels.vllm.ai/rocm/) need glibc 2.39 or later; RHEL 9 ships 2.34. On such hosts
  `local_orchestrator.sh` runs `serve.sh` inside the `vllm/vllm-openai-rocm` image (pin it by digest) with the task dir
  mounted, and the Inspect client still runs from a venv on the host. Under podman:
  - The user needs read-write access to `/dev/kfd` and `/dev/dri/renderD*` (world-writable, or through groups passed
    with `--group-add keep-groups`).
  - Pass `--pids-limit -1`. With DP=8, vLLM starts 16 processes (8 API servers, 8 engines), each with a thread per
    CPU, which exhausts podman's default of 2048 tasks per container.
  - Publish only the server port (no `--network host`), so two servers on disjoint GPUs cannot clash on vLLM's
    internal ports. Publish it on `127.0.0.1`, because `localhost` resolves to `::1` first.
  - Pass `--init`: vLLM as PID 1 ignores SIGTERM, so `podman stop` cannot stop it gracefully.
  - With SELinux enforcing, pass `--security-opt label=disable`; otherwise the container cannot list the directories
    of the bind-mounted HF cache.
  - Mount a host directory on the container's `/root/.cache/vllm` to keep the torch.compile cache: the first start of
    a new config compiles for about 10 min, past the default `VLLM_ENGINE_READY_TIMEOUT_S` of 600 s (raise it, e.g. to
    1800), and a warm start takes 3-4 min.
  - TP>1 can fail in rootless containers: RCCL's P2P transport fails with `hipIpcGetMemHandle failed`, even with
    `--cap-add SYS_PTRACE`. Scale out with DP instead.
- NUMA auto-balancing (`kernel.numa_balancing=1`; AMD's ROCm tuning guides say to disable it) stalls vLLM start-up for
  10+ minutes on a multi-socket host: tiny host-to-GPU copies take minutes, with threads stuck in mmap. Without root,
  start the container under `numactl --interleave=all`; the kernel does not balance processes that have an explicit
  memory policy, and the container's processes inherit it.
- Copying memory-mapped safetensors to the GPU can cost about 1 s per tensor whatever its size (about 330 s for an
  18 GB checkpoint). `--safetensors-load-strategy eager` reads the files into RAM first and loads in about 5 s.
- Attention backend (vLLM 0.31.0, Qwen3.8 hybrid attention): the ROCm default `ROCM_ATTN` decodes at about 5 tok/s per
  stream at 100k context, which pushes long reasoning traces past Inspect's 7200 s `client_timeout` (the request is
  retried from scratch). `--attention-backend TRITON_ATTN` does about 77 tok/s there. `ROCM_AITER_UNIFIED_ATTN` is faster
  still (about 115) but produced garbage for `RedHatAI/Qwen3.8-27B-MXFP4` with `--kv-cache-dtype bfloat16` (correct with
  its FP8 KV and for the AMD checkpoint); `ROCM_AITER_FA` stalls at 16 concurrent streams. `VLLM_ROCM_USE_AITER=1` adds
  2-7% and swaps more kernels; not used. Throughput benchmarks with random prompts do not see broken outputs: check
  greedy generations before adopting a kernel setting. With `TRITON_ATTN`, vLLM 0.31.0 decodes the 27B MXFP4 6-22%
  faster than 0.29.0 on one MI355X.

## Docker for sandboxed and agentic tasks

- Check that `docker` is the Docker CLI: some hosts alias it to `podman` in interactive shells. Shell aliases do not
  apply to the `docker` that Inspect runs, so call the real binary (e.g. `/usr/bin/docker`) in your own commands.
- Disk: the image store needs room for the task images. SWE-bench Pro takes about 4 GB per task (about 3 TB for 731
  tasks), SWE-bench Verified about 5 GB per instance before layer sharing, DeepSWE about 122 GB for 113 tasks. Check
  free space on the Docker data-root before any sandboxed task, and move the data-root to a large local disk.
- podman stages pulled layers in `/var/tmp`. If that is on a small root disk, pull with
  `TMPDIR=<dir on the large disk> podman pull ...` (or set `image_copy_tmp_dir` in `containers.conf`).
- Without root, run a rootless Docker daemon of your own user:
  - Install the static Docker binaries with the rootless extras and `slirp4netns`, and the compose plugin (v2 or
    later) in `~/.docker/cli-plugins`. Then run
    `DOCKERD_ROOTLESS_ROOTLESSKIT_NET=slirp4netns dockerd-rootless-setuptool.sh install --force`, which installs a
    systemd user unit. `loginctl enable-linger $USER` keeps the daemon running after you log out. Without lingering,
    the daemon and rootless podman containers stop when your last login session ends; run long evals from a session
    that stays open (e.g. tmux).
  - The setup tool refuses to install while a rootful daemon is reachable. In that case, or without lingering, start
    the daemon by hand in a delegated systemd user scope; otherwise runc cannot create container cgroups:
    `XDG_RUNTIME_DIR=/run/user/$(id -u) nohup systemd-run --user --scope -p Delegate=yes --unit=docker-rootless dockerd-rootless.sh > <log> 2>&1 &`.
  - In `~/.config/docker/daemon.json`:
    - Put `data-root` on a large local disk.
    - Use the containerd snapshotter: the overlay2 driver refuses to run rootless when SELinux is enabled, even in
      permissive mode.
    - Set `default-address-pools` for many networks, e.g. two /16 pools split into /24 (512 networks). Multi-service
      tasks (tau3-bench) create one compose network per sample, and Docker's default pools (about 30 networks) run
      out near 40 concurrent samples.
    - Give the default runc runtime `NoNewKeyring`: define a runtime (e.g. `runc-nokeyring`) with
      `"runtimeType": "io.containerd.runc.v2"` and `"options": {"NoNewKeyring": true}`, and make it `default-runtime`.
      Otherwise each container takes a kernel keyring, and the kernel allows 200 keys per user.
  - Delegate the cpu, memory and pids cgroup controllers to the user so container limits work. Without the cpu
    controller, Docker rejects CPU limits ("NanoCPUs can not be set"), hence Harbor's `-T override_cpus=0`.
  - `local_orchestrator.sh` exports `DOCKER_HOST=unix:///run/user/$UID/docker.sock` and puts the rootless binaries
    first in `PATH`.
  - The daemon's default `bridge` network gets the first /24 of the address pools (about 250 addresses). The forked
    inspect_harbor moves SWE-bench Pro containers onto it for the verifier phase, so at most about 250 samples can be
    verifying at once.
- Rootless Docker maps only the user's subordinate UIDs (often 65,536). Some images contain files owned by higher
  UIDs and fail to extract: the SWE-bench Verified matplotlib images (32 of the 34) and the SWE-bench Pro
  openlibrary/protonmail base images. Either raise the `/etc/subuid` range (root), or load copies whose layers are
  rewritten to map those owners to root. Without root, pull the image into a separate rootless podman store that ignores
  chown errors (`podman --root <dir> --runroot <dir> --storage-driver overlay --storage-opt overlay.ignore_chown_errors=true`),
  `podman save` it as a docker-archive, rewrite the owners above the range to root in its layers (file contents
  unchanged), and `docker load` the result under the original name. With the containerd image store, name a loaded
  image with its full `docker.io/<repo>:<tag>` reference; it does not resolve unnormalized names.
- inotify instances: `fs.inotify.max_user_instances` is 128 on many distributions, and rootless containers share one
  per-user budget.
  - Each container's containerd-shim holds one instance, so with more than about 120 containers running, programs in
    the containers that need one fail with `EMFILE: too many open files` (Node `fs.watch`/chokidar, Go fsnotify,
    webpack, testem). Seen: tutanota's build-server tests in SWE-bench Pro, which stop `npm test` before the suites the
    verifier checks, and one DeepSWE verifier.
  - Other users of the budget: qutebrowser tasks in SWE-bench Pro leave a session `dbus-daemon` behind after every
    test run under `xvfb-run` (80-90 instances by the end of a run), and every `tail -F`.
  - Fix (root): `sysctl -w fs.inotify.max_user_instances=8192`, persisted in `/etc/sysctl.d/`. Without root, keep the
    number of containers well below the limit, run the inotify-sensitive tasks (tutanota) with few containers alive,
    and watch logs by polling file sizes instead of `tail -F`. Count the instances in use with
    `ls -l /proc/*/fd 2>/dev/null | grep -c inotify` over your own processes.
- When many sandboxes are torn down at once, some containers end up `Dead` with "removal already in progress", and
  `docker system df` fails with snapshot races. The daemon finishes those removals by itself; `docker ps` keeps
  working.

## Getting the task images

- Docker Hub limits anonymous pulls per IP address (100 per hour when these evals ran), and everyone behind the same
  IP shares the budget. Pre-pull or pre-build the task images ahead of the run with a loop that paces pulls against the
  remaining budget, rather than letting Inspect pull on demand. Inspect skips building an image that already exists
  (inspect_harbor names them `hb__<environment hash>`).
- `mirror.gcr.io` (Google's Docker Hub pull-through cache) serves the SWE-bench and SWE-bench Pro images without that
  limit and without a login. `docker pull mirror.gcr.io/<repo>:<tag>`, then `docker tag` it to `docker.io/<repo>:<tag>`
  and drop the mirror tag. Docker Hub also counts manifest HEAD requests against the limit (a 429 on
  `HEAD .../manifests/latest` was seen on 2026-10-21), so comparing the mirror's digest with the Hub's is not free;
  the SWE-bench image tags are immutable in practice. Four pulls in parallel through the mirror were fine.
- SWE-bench Verified's default images (Epoch AI's `ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance>`) pull
  anonymously from `ghcr.io`, outside Docker Hub's limit.
- SWE-bench Pro task images are built `FROM jefzda/sweap-images:<instance>` on Docker Hub. Build them in batches
  (`docker build -t <name> <task>/environment`) and delete the images of finished batches when disk is short. Building
  a few in parallel as their base images arrive saves Inspect from building all 731 serially before the first sample;
  take each `hb__<environment hash>` name from the task's own sandbox config.
- The DeepSWE task images come from ECR public, which throttles bursts; pull them one at a time.

## Running long evals

- Launch long runs with `setsid nohup ...` so they survive a Claude Code restart or a dropped SSH session (but see
  lingering above).
- `inspect eval-retry <log>` resumes an interrupted eval. It first writes a `-recovered.eval` copy of the log; the new
  log it writes carries all samples (finished ones kept, unfinished and errored ones rerun).
- A running eval can be retuned live: `inspect ctl config <task_id> --max-samples N --max-sandboxes N --key docker N
  --key vllm/<model> N` (each shard is its own process).

## Client environments beyond the three in `AGENTS.md`

Environments are local to each user and server: create them from the task dir's `client_requirements.txt` (its README
Setup gives the commands), name and place them as you like, and point `local_orchestrator.sh` at them.

- SWE-bench Verified dirs that pin inspect-ai 0.3.263, inspect-evals 0.20.0 and swebench 4.1.0 need an environment of
  their own: inspect_evals' scorer breaks under swebench 5 (see those READMEs' Patches).
- DeepSWE and SWE-bench Pro dirs that run the `eldarkurtic/inspect_harbor@fix/enforce-agent-network-mode` fork
  (AGENTS.md section 11) need an environment of their own. Install the fork by commit with
  `uv pip install --no-deps "inspect_harbor @ git+https://github.com/eldarkurtic/inspect_harbor@<commit>"`: its
  pyproject's uv `exclude-newer = "7 days"` breaks a resolving install, and a local-directory install records no
  commit in `pip freeze`.
- The superseded earlier DeepSWE runs of the Qwen3.8-27B dirs (before the comparison set of 2026-10-21) used stock
  inspect-harbor 1.0.0, for its verifier fixes, without the network fix. Don't reuse that setup for new DeepSWE runs.
- A `client_requirements.txt` that contains litellm next to `openai==3.14.0` installs only with
  `pip install --no-deps -r client_requirements.txt`; a resolving install fails on litellm's `openai<3` pin.
