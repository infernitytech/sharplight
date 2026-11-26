# Agent and coordinator scripts

Machine setup and merge tooling used by the coordinator and the worktree agents. These scripts are not product code and
are not part of any package. Every path comes from an environment variable with a default under `$HOME`, so they work
on any Linux machine; `env.sh` lists the variables.

| script | purpose |
|---|---|
| `env.sh` | Source it in every shell (`. tools/agent/env.sh`). Sets the path variables (`SL_DATA`, `SL_LOCAL`, `SL_REPO`, `SL_VERIFY`, `SL_WORKTREES`, `SL_WORK`, `SL_MODELS`, `SL_MODELS_LOCAL`, `SL_TEST_HOME`, `SL_STATE`, `LLAMA_ROOT`), the .NET SDK, `TMPDIR` on local disk, `HF_TOKEN` from a private file, and a `dotnet` wrapper that limits concurrent `build`/`pack`/`publish` to `SL_BUILD_SLOTS`. |
| `verify.sh <tag> [suite...]` | Under the verify lock (`$SL_STATE/.sl-verify.lock`): checks out `develop` of `$SL_REPO` detached in `$SL_VERIFY`, validates the workflow YAML, runs the style checkers, builds `Sharplight.slnx` in Release, then runs the given suites. Prints `RESULT-<tag> done <commit>`. Logs: `$SL_WORK/logs/verify-<tag>/`. |
| `verify-push.sh <tag> [suite...]` | `verify.sh`, then pushes `develop` to `origin` only if the build passed, every suite ended with `exit=0`, the privacy scan of the new commits is clean, all authors and committers are GitHub noreply addresses, and no tracked file name is invalid on Windows. Also pushes to the `backup` remote when one exists. |
| `run-suites.sh <repo> <logdir> [suite...]` | Runs test projects niced, at most `SUITE_JOBS` at a time; GPU suites one class per process, pinned round-robin over `SUITE_GPUS`. Suite names drop the `Sharplight.` prefix and keep `.Tests` (`Core.Tests`, `Vulkan.Tests`, `Import.Onnx.Conformance`). |
| `sl-test <TestProject> [xunit args]` | Runs one built test project with `HOME=$SL_TEST_HOME`, so `~/models` resolves to the local model mirror. |
| `sync-local-models.sh` | Mirrors `$SL_MODELS` into `$SL_MODELS_LOCAL` (tiny-* copied, everything else symlinked) and links `$SL_TEST_HOME/models`. |
| `cleanup.sh` | Prunes stale worktrees and temp files older than 6 hours; prints free disk and RAM. |
| `bootstrap-local.sh` | Rebuilds a machine's local state (apt packages, SDK 11.0.100, minisign/nfpm, shell profile, gh/git/Claude settings from a private folder, engine clone + verify worktree, Claude memory from `docs/handoff/memory`, uv venvs from pinned freezes, dev PostgreSQL for the server tests, Playwright, model mirror, optional llama.cpp and real models, optional feed runner). `--check` reports only; `--prepare-relaunch` backs up branches, memory and settings before a machine move. Pins (versions + sha256) sit at the top. |

## Typical use

```sh
export SL_DATA=/data/sharplight SL_LOCAL=/fast/sharplight   # once per machine, e.g. in ~/.bashrc; defaults are under $HOME
git clone https://github.com/infernitytech/sharplight.git "$SL_LOCAL/sl-repo"
sudo -E "$SL_LOCAL/sl-repo/tools/agent/bootstrap-local.sh"  # or --check first
. "$SL_LOCAL/sl-repo/tools/agent/env.sh"

# agent worktree
git -C "$SL_REPO" worktree add "$SL_WORKTREES/<name>" -b <branch> develop

# coordinator: merge a green branch, then verify and push
git -C "$SL_REPO" merge --no-ff <branch>
GPUS=0 tools/agent/verify-push.sh <tag> Core.Tests Cuda.Tests Vulkan.Tests
```

## Locks

- `$SL_STATE/.sl-verify.lock`: held by `verify.sh` for checkout + build + suites. Any process that opens a GPU context
  (tests, servers, benchmarks) runs inside this lock or `$SL_STATE/.sl-bench.lock`, and stops before releasing it.
- Builds inside a lock use `-nodeReuse:false -p:UseSharedCompilation=false` and close the lock descriptor (`9>&-`); a
  reused MSBuild node or compiler server otherwise keeps the lock after the script exits.
- `$SL_STATE/.sl-build-slots/`: the `dotnet` wrapper from `env.sh`.

## Secrets

The scripts never contain or print secrets. `HF_TOKEN` comes from `SL_HF_TOKEN_FILE` (mode 600, outside every repo).
GitHub login, git config and Claude settings are backed up to `SL_SECRETS` (mode 700) by `--prepare-relaunch`. Tokens
for GitHub runners, R2 and signing keys are provided by the owner; never create or fetch them yourself.
Extra private words for the privacy scan (for example the owner's private name) go into `SL_PRIVACY_EXTRA` in the local
environment, never into a repo.
