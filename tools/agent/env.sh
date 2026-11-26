# Sharplight development environment for agents and the coordinator. Source it in every shell: `. tools/agent/env.sh`
# (or from ~/.bashrc). Every path is an environment variable with a default; set the variables before sourcing to change them.
#
#   SL_DATA        persistent data root (repo backups, tools, models, logs)           default: $HOME/sharplight-data
#   SL_LOCAL       fast local disk for clones, worktrees, venvs, model mirror, locks  default: $HOME
#   SL_TOOLS       toolchains (dotnet, vulkansdk, llama.cpp references)               default: $SL_DATA/tools
#   SL_WORK        build output, logs, temp (never /tmp)                             default: $SL_DATA/work
#   SL_MODELS      master model store (~/models points here)                          default: $SL_DATA/models
#   SL_MODELS_LOCAL  local mirror read by the tests (tools/agent/sync-local-models.sh) default: $SL_LOCAL/models-local
#   SL_TEST_HOME   HOME for test runs; its `models` links to SL_MODELS_LOCAL          default: $SL_LOCAL/test-home
#   SL_REPO        engine working clone (merges happen here)                          default: $SL_LOCAL/sl-repo
#   SL_VERIFY      detached verify worktree of develop                                default: $SL_LOCAL/sl-verify
#   SL_WORKTREES   agent worktrees                                                    default: $SL_LOCAL/sl-worktrees
#   SL_STATE       lock files (.sl-verify.lock, .sl-bench.lock, .sl-build-slots/)     default: $SL_LOCAL
#   SL_BUILD_SLOTS concurrent dotnet build|pack|publish on this machine               default: 2
#   SL_HF_TOKEN_FILE  shell file that exports HF_TOKEN (mode 600, outside every repo) default: $SL_TOOLS/hf-token.sh
#   LLAMA_ROOT     llama.cpp reference binaries                                       default: $SL_TOOLS/llama

export SL_DATA=${SL_DATA:-$HOME/sharplight-data}
export SL_LOCAL=${SL_LOCAL:-$HOME}
export SL_TOOLS=${SL_TOOLS:-$SL_DATA/tools}
export SL_WORK=${SL_WORK:-$SL_DATA/work}
export SL_MODELS=${SL_MODELS:-$SL_DATA/models}
export SL_MODELS_LOCAL=${SL_MODELS_LOCAL:-$SL_LOCAL/models-local}
export SL_TEST_HOME=${SL_TEST_HOME:-$SL_LOCAL/test-home}
export SL_REPO=${SL_REPO:-$SL_LOCAL/sl-repo}
export SL_VERIFY=${SL_VERIFY:-$SL_LOCAL/sl-verify}
export SL_WORKTREES=${SL_WORKTREES:-$SL_LOCAL/sl-worktrees}
export SL_STATE=${SL_STATE:-$SL_LOCAL}
export SL_BUILD_SLOTS=${SL_BUILD_SLOTS:-2}
export SL_HF_TOKEN_FILE=${SL_HF_TOKEN_FILE:-$SL_TOOLS/hf-token.sh}
export LLAMA_ROOT=${LLAMA_ROOT:-$SL_TOOLS/llama}
# Kept for older scripts and docs that read it.
export SHARPLIGHT_WORK=$SL_WORK

# .NET SDK: a local copy is preferred (a network volume can be slow), then the persistent one, then whatever is installed.
if [ -x "$SL_LOCAL/dotnet-local/dotnet" ]; then
    export DOTNET_ROOT=$SL_LOCAL/dotnet-local
elif [ -x "$SL_TOOLS/dotnet/dotnet" ]; then
    export DOTNET_ROOT=$SL_TOOLS/dotnet
fi
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
if [ -n "${DOTNET_ROOT:-}" ]; then
    export PATH="$DOTNET_ROOT:$HOME/.dotnet/tools:$PATH"
fi

# ~/models is the path tests and docs use.
if [ -d "$SL_MODELS" ] && [ ! -e "$HOME/models" ]; then
    ln -sfn "$SL_MODELS" "$HOME/models"
fi
if [ -f "$SL_TOOLS/vulkansdk/current/setup-env.sh" ]; then
    . "$SL_TOOLS/vulkansdk/current/setup-env.sh"
fi

# Temp files on local disk, never /tmp (owner rule 2026-11-15).
export TMPDIR=${SL_TMPDIR:-$SL_LOCAL/work-tmp}
mkdir -p "$TMPDIR" "$SL_WORK" 2>/dev/null

# Hugging Face token (read-only, provided by the owner). Sourced, never printed or committed.
if [ -f "$SL_HF_TOKEN_FILE" ]; then
    . "$SL_HF_TOKEN_FILE"
fi

# Machine-wide limit on concurrent dotnet builds (several agents share a machine; the owner's session must stay responsive).
# `dotnet build|pack|publish` waits for one of SL_BUILD_SLOTS flock slots; every other dotnet command runs directly.
_sl_with_build_slot() {
    local dir="$SL_STATE/.sl-build-slots" slot fd rc
    mkdir -p "$dir"
    while :; do
        for slot in $(seq 1 "${SL_BUILD_SLOTS:-2}"); do
            exec {fd}>"$dir/$slot"
            if flock -n "$fd"; then
                "$@"
                rc=$?
                flock -u "$fd"
                exec {fd}>&-
                return $rc
            fi
            exec {fd}>&-
        done
        sleep 3
    done
}
dotnet() {
    case "${1:-}" in
        build|pack|publish) _sl_with_build_slot command dotnet "$@" ;;
        *) command dotnet "$@" ;;
    esac
}
export -f _sl_with_build_slot dotnet
