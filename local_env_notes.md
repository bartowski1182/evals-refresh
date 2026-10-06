# Environment notes

Requirements and pitfalls of the hosts these evals run on, written to apply to any Linux GPU server (this file is
referenced and imported from `AGENTS.md`). Nothing in it should be needed to understand the task directories or
their READMEs.

Facts about one particular machine (its paths, user names, installed versions, other tenants) do not belong here.
Keep them in `local_env_notes.local.md` next to this file, which git ignores, and read that file too if it exists.

## Serving

- `local_orchestrator.sh` activates the vLLM venv, starts `serve.sh` in the background, polls `/health`, activates
  the Inspect venv, runs `run_eval.sh "$@"`, and kills the server.
- Serving convention: `TP=2`, `DP=4` (four replicas behind one port on all 8 GPUs) for ~30B bf16 models on 80 GB
  GPUs. Adjust per model size; a 27–31B bf16 model needs ~26–31 GB per GPU at TP=2.
- Keep checkpoints and the HF caches (`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`) on fast local disks. Network
  volumes (NFS, cloud block storage) can cap uncached reads near 100 MB/s, so loading a 100+ GB checkpoint takes tens
  of minutes. `serve.sh` inherits the cache variables from the shell.
- Qwen4Exp (Qwen3.8-Flash-Next) keeps its 51B n-gram table in pinned host memory. While a server loads, every DP
  worker holds a private host copy (about 136 GB each for the BF16 table), so two DP=4 servers starting together need
  about 1.1 TB of RAM and evict the page cache. Start the servers one after the other.
- On a shared host, check `nvidia-smi` before claiming GPUs. Stop processes by exact PID: `pgrep -f`/`pkill -f`
  patterns can match other users' processes or your own shell.

## Docker for sandboxed and agentic tasks

- Check that `docker` is the Docker CLI: some hosts alias it to `podman` in interactive shells. Shell aliases do not
  apply to the `docker` that Inspect runs, so call the real binary (e.g. `/usr/bin/docker`) in your own commands.
- Disk: the image store needs room for the task images. SWE-bench Pro takes about 4 GB per task (about 3 TB for 731
  tasks), SWE-bench Verified about 5 GB per instance before layer sharing, DeepSWE about 122 GB for 113 tasks. Check
  free space on the Docker data-root before any sandboxed task, and move the data-root to a large local disk.
- Without root, run a rootless Docker daemon of your own user:
  - Install the static Docker binaries with the rootless extras and `slirp4netns`, and the compose plugin (v2 or
    later) in `~/.docker/cli-plugins`. Then run
    `DOCKERD_ROOTLESS_ROOTLESSKIT_NET=slirp4netns dockerd-rootless-setuptool.sh install --force`, which installs a
    systemd user unit. `loginctl enable-linger $USER` keeps the daemon running after you log out.
  - In `~/.config/docker/daemon.json`:
    - Put `data-root` on a large local disk.
    - Use the containerd snapshotter: the overlay2 driver refuses to run rootless when SELinux is enabled, even in
      permissive mode.
    - Set `default-address-pools` for many networks, e.g. two /16 pools split into /24 (512 networks). Multi-service
      tasks (tau3-bench) create one compose network per sample, and Docker's default pools (about 30 networks) run
      out near 40 concurrent samples.
    - Give the default runc runtime `NoNewKeyring`. Otherwise each container takes a kernel keyring, and the kernel
      allows 200 keys per user.
  - Delegate the cpu, memory and pids cgroup controllers to the user so container limits work.
  - `local_orchestrator.sh` exports `DOCKER_HOST=unix:///run/user/$UID/docker.sock` and puts the rootless binaries
    first in `PATH`.
  - The daemon's default `bridge` network gets the first /24 of the address pools (about 250 addresses). The forked
    inspect_harbor moves SWE-bench Pro containers onto it for the verifier phase, so at most about 250 samples can be
    verifying at once.
- Rootless Docker maps only the user's subordinate UIDs (often 65,536). Some images contain files owned by higher
  UIDs and fail to extract: the SWE-bench Verified matplotlib images and the SWE-bench Pro openlibrary/protonmail base
  images. Either raise the `/etc/subuid` range (root), or load copies whose layers are rewritten to map those owners to
  root. With the containerd image store, name a loaded image with its full `docker.io/<repo>:<tag>` reference; it does
  not resolve unnormalized names.
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
  limit. Take the manifest digest from Docker Hub with a HEAD request (it does not count against the limit), check that
  the mirror serves the same digest, `docker pull mirror.gcr.io/<repo>@<digest>`, then `docker tag` it to
  `docker.io/<repo>:<tag>`.
- SWE-bench Pro task images are built `FROM jefzda/sweap-images:<instance>` on Docker Hub. Build them in batches
  (`docker build -t <name> <task>/environment`) and delete the images of finished batches when disk is short.
- The DeepSWE task images come from ECR public, which throttles bursts; pull them one at a time.

## Client venvs beyond the three in `AGENTS.md`

- `~/.venvs/inspect-swebench` (inspect-ai 0.3.263, inspect-evals 0.20.0, swebench 4.1.0) for the SWE-bench Verified
  dirs that pin those versions.
- `~/.venvs/inspect-harbor-fix` for the DeepSWE and SWE-bench Pro dirs that need the
  `eldarkurtic/inspect_harbor@fix/enforce-agent-network-mode` fork. Install the fork by commit with
  `uv pip install --no-deps "inspect_harbor @ git+...@<commit>"`: its pyproject's uv `exclude-newer = "7 days"` breaks
  a resolving install, and a local-directory install records no commit in `pip freeze`.
