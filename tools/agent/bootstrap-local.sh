#!/usr/bin/env bash
# Rebuilds the local state of a Sharplight development machine (SDK, tools, clones, venvs, dev PostgreSQL, browsers, model mirror)
# and, before a machine move or wipe, backs up what lives only on local disk. Machine setup, not product code; see tools/agent/README.md.
#
#   bootstrap-local.sh --prepare-relaunch [--dry-run]   BEFORE a move/wipe: push branches to backup, back up memory + settings, list dirty trees
#   bootstrap-local.sh                                  AFTER: full bootstrap (idempotent, safe to re-run)
#   bootstrap-local.sh --check                          report what is present or missing, change nothing
#
# Options: --only a,b,c   run only these steps        --skip a,b   skip steps        --list   print the step names
#          --all-venvs    also build the optional venvs (audio-gen, omnivoice, qtts, vibevoice, speech-r5, ...)
#          --models       also download the frequently used real models (prints sizes first; --models-list = sizes only)
#          --models-big   with --models: also the big ones (Qwen3-8B, ...)
#          --llama        also fetch the llama.cpp Vulkan release for references ($SL_LOCAL/llama-local)
#          --no-verify    skip the final verification
# Steps: apt dotnet tools shell secrets repo verifywt enginesrc memory venvs pg playwright testhome dirs serverclones runner [llama] [models]
#
# Paths come from the same variables as tools/agent/env.sh (SL_DATA, SL_LOCAL, SL_REPO, ...; defaults under $HOME). Extra:
#   SL_ENGINE_BACKUP   persistent backup remote of the engine repo (optional)       default: $SL_DATA/sharplight
#   SL_SERVER_BACKUP   clone of the private server repo (dev PostgreSQL scripts)    default: $SL_DATA/sharplight-server
#   SL_SECRETS         private folder (mode 700) for gh auth, git config, Claude settings; never inside a repo
#                                                                                    default: $SL_TOOLS/secrets
#   SL_MEMORY_BACKUP   folder with backed-up Claude memory notes                    default: $SL_WORK/memory-backup
#   SL_RUNNER_DIR, SL_RUNNER_USER   self-hosted feed runner install dir and user (step "runner"; skipped when unset)
set -uo pipefail
# ---------------------------------------------------------------- pins (change here only)
SL_DATA=${SL_DATA:-$HOME/sharplight-data}
SL_LOCAL=${SL_LOCAL:-$HOME}
SL_TOOLS=${SL_TOOLS:-$SL_DATA/tools}
SL_WORK=${SL_WORK:-$SL_DATA/work}
SL_MODELS=${SL_MODELS:-$SL_DATA/models}
SL_MODELS_LOCAL=${SL_MODELS_LOCAL:-$SL_LOCAL/models-local}
SL_TEST_HOME=${SL_TEST_HOME:-$SL_LOCAL/test-home}
SL_REPO=${SL_REPO:-$SL_LOCAL/sl-repo}
SL_VERIFY=${SL_VERIFY:-$SL_LOCAL/sl-verify}
SL_WT=${SL_WORKTREES:-$SL_LOCAL/sl-worktrees}
SL_STATE=${SL_STATE:-$SL_LOCAL}
ENGINE_SRC=${SL_ENGINE_SRC:-$SL_LOCAL/sl-engine-src}
ENGINE_BACKUP=${SL_ENGINE_BACKUP:-$SL_DATA/sharplight}
SERVER_BACKUP=${SL_SERVER_BACKUP:-$SL_DATA/sharplight-server}
GITHUB_ENGINE=https://github.com/infernitytech/sharplight.git
MEMORY_BACKUP=${SL_MEMORY_BACKUP:-$SL_WORK/memory-backup}
SECRETS=${SL_SECRETS:-$SL_TOOLS/secrets}  # created by --prepare-relaunch (mode 700): gh auth, git config, claude settings
VENV_REQS=${SL_VENV_REQS:-$SL_TOOLS/venv-reqs}  # pip freeze per venv (small text files; --prepare-relaunch refreshes them)
VENVS=$SL_LOCAL/venvs
TOOLS_LOCAL=$SL_LOCAL/tools-local
DOTNET_LOCAL=$SL_LOCAL/dotnet-local
SCRATCH=$SL_LOCAL/sl-tmp
DOTNET_VERSION=11.0.100
DOTNET_URL=https://builds.dotnet.microsoft.com/dotnet/Sdk/$DOTNET_VERSION/dotnet-sdk-$DOTNET_VERSION-linux-x64.tar.gz
DOTNET_SHA256=6fdb8822f47ccdba28f507dcd5d92e632d5b99cd27908d6838f0cc575be33baf
MINISIGN_VERSION=0.12
MINISIGN_URL=https://github.com/jedisct1/minisign/releases/download/$MINISIGN_VERSION/minisign-$MINISIGN_VERSION-linux.tar.gz
MINISIGN_SHA256=9a599b48ba6eb7b1e80f12f36b94ceca7c00b7a5173c95c3efc88d9822957e73
NFPM_VERSION=2.49.0
NFPM_URL=https://github.com/goreleaser/nfpm/releases/download/v$NFPM_VERSION/nfpm_${NFPM_VERSION}_Linux_x86_64.tar.gz
NFPM_SHA256=2afaa5ffb5e3493402e31e7c67a5a1fb42133e3f66fc315ca0051934125561c7
LLAMA_BUILD=b12483
LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-bin-ubuntu-vulkan-x64.tar.gz
LLAMA_SHA256=0c2a5e15ea2ab8790ec6e77e7bd374ce5e13e7b7fb25955019793b9ff8b314d8
PLAYWRIGHT_VERSION=1.64.1
PW_BROWSERS=$SL_LOCAL/pw-browsers
PG_ROOT=/opt/sl-pg
PG_PORT=55432
UV_CACHE=$SL_LOCAL/uvcache
TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
APT_PACKAGES="libegl1 libgl1 libvulkan1 libnss3 libnspr4 libxcb-cursor0 xvfb xdotool x11-utils stress-ng jq rsync inotify-tools lsof
  unzip zip zstd ffmpeg imagemagick build-essential cmake ninja-build pkgconf git gh curl wget tmux qemu-user libc6-arm64-cross ruby-full
  python3.12-venv libssl-dev libffi-dev libjpeg-dev libpng-dev"
# venv name -> requirement source. Important ones first; the rest only with --all-venvs.
IMPORTANT_VENVS="ref vllm"
OPTIONAL_VENVS="audio-gen omnivoice qtts vibevoice speech-r5 sdxl-refiner da3-extra loadgen pdf"
# Real models for --models: "<hub repo>|<target under $SL_MODELS_LOCAL>|<include patterns, comma separated or empty>"
MODELS_SMALL="Qwen/Qwen3-0.6B|hf/Qwen3-0.6B|
Qwen/Qwen3-1.7B|hf/Qwen3-1.7B|
Qwen/Qwen2.5-0.5B-Instruct|hf/Qwen2.5-0.5B-Instruct|
Qwen/Qwen2.5-1.5B-Instruct|hf/Qwen2.5-1.5B-Instruct|
meta-llama/Llama-3.2-1B-Instruct|hf/Llama-3.2-1B-Instruct|
HuggingFaceTB/SmolLM2-135M|hf/SmolLM2-135M|
sentence-transformers/all-MiniLM-L6-v2|hf/all-MiniLM-L6-v2|
BAAI/bge-small-en-v1.5|hf/bge-small-en-v1.5|
Qwen/Qwen3-Embedding-0.6B|hf/Qwen3-Embedding-0.6B|
Qwen/Qwen3-Reranker-0.6B|hf/Qwen3-Reranker-0.6B|
Qwen/Qwen3-0.6B-GGUF|gguf/Qwen3-0.6B|Qwen3-0.6B-Q8_0.gguf
Qwen/Qwen3-4B-GGUF|gguf/Qwen3-4B|Qwen3-4B-Q4_K_M.gguf"
MODELS_BIG="Qwen/Qwen3-8B|hf/Qwen3-8B|
Qwen/Qwen3-Embedding-4B|hf/Qwen3-Embedding-4B|
Qwen/Qwen3-Reranker-4B|hf/Qwen3-Reranker-4B|"
SLM_CONVERT="gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf|qwen3-0.6b-q8_0.slm
gguf/Qwen3-4B/Qwen3-4B-Q4_K_M.gguf|qwen3-4b-q4_k_m.slm"

# ---------------------------------------------------------------- args
MODE=bootstrap DRY=0 ONLY="" SKIP="" ALL_VENVS=0 MODELS=0 MODELS_BIGF=0 MODELS_LIST=0 LLAMA=0 VERIFY=1
while [ $# -gt 0 ]; do
  case "$1" in
    --prepare-relaunch) MODE=prepare ;;
    --check) MODE=check ;;
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    --skip) SKIP="$2"; shift ;;
    --all-venvs) ALL_VENVS=1 ;;
    --models) MODELS=1 ;;
    --models-big) MODELS_BIGF=1 ;;
    --models-list) MODELS=1; MODELS_LIST=1 ;;
    --llama) LLAMA=1 ;;
    --no-verify) VERIFY=0 ;;
    --list) sed -n '2,/^set /p' "$0" | grep '^# Steps' ; exit 0 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- helpers
export DEBIAN_FRONTEND=noninteractive
MISSING=()
say()  { printf '\n=== %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }
have() { command -v "$1" >/dev/null 2>&1; }
miss() { MISSING+=("$1"); printf '    MISSING: %s\n' "$1"; }
ok()   { printf '    ok: %s\n' "$*"; }
run()  { # run unless dry-run
  if [ "$DRY" = 1 ]; then printf '    [dry-run] %s\n' "$*"; else "$@"; fi
}
want() { # step enabled?
  local s=$1
  if [ -n "$ONLY" ]; then case ",$ONLY," in *",$s,"*) return 0 ;; *) return 1 ;; esac; fi
  case ",$SKIP," in *",$s,"*) return 1 ;; esac
  return 0
}
sha_ok() { [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }
download() { # url dest sha256
  local url=$1 dest=$2 sum=$3
  curl -fL --retry 3 -o "$dest" "$url" || return 1
  if ! sha_ok "$dest" "$sum"; then echo "checksum mismatch for $url" >&2; rm -f "$dest"; return 1; fi
}
scratch() { mkdir -p $SCRATCH/bootstrap && mktemp -d $SCRATCH/bootstrap/XXXXXX; }
repo_key() { printf '%s' "$1" | sed 's#[^a-zA-Z0-9]#-#g'; }
dirty_report() { # dir -> prints porcelain lines, returns 1 if dirty
  local d=$1 n s
  n=$(git -C "$d" status --porcelain 2>/dev/null | wc -l)
  s=$(git -C "$d" stash list 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ] || [ "$s" -gt 0 ]; then
    printf '    DIRTY %s: %s changed/untracked paths, %s stashes\n' "$d" "$n" "$s"
    git -C "$d" status --porcelain 2>/dev/null | head -15 | sed 's/^/        /'
    [ "$n" -gt 15 ] && printf '        ... (%s more)\n' $((n - 15))
    return 1
  fi
  return 0
}

STEP_TIMES=()
timed() { # name function
  local t0 t1
  t0=$(date +%s)
  say "$1"
  "$2"
  t1=$(date +%s)
  STEP_TIMES+=("$(printf '%-14s %5ss' "$1" $((t1 - t0)))")
}

# ================================================================ --prepare-relaunch
prepare_engine() {
  local b n=0
  info "engine: $SL_REPO (remote backup = $ENGINE_BACKUP)"
  if [ ! -d "$SL_REPO/.git" ]; then warn "$SL_REPO missing"; return; fi
  while read -r b; do
    [ -n "$b" ] || continue
    n=$((n + 1))
    local ahead
    ahead=$(git -C "$SL_REPO" rev-list --count "refs/remotes/backup/$b..refs/heads/$b" 2>/dev/null || echo new)
    if [ "$DRY" = 1 ]; then
      info "[dry-run] would push $b to backup ($ahead commits not on backup; as last fetched)"
      git -C "$SL_REPO" push --dry-run backup "refs/heads/$b:refs/heads/$b" 2>&1 | sed 's/^/        /' | head -4
    else
      git -C "$SL_REPO" push -q backup "refs/heads/$b:refs/heads/$b" 2>&1 | sed 's/^/        /' \
        && info "pushed $b" || warn "push of $b to backup failed (non-fast-forward? resolve by hand, never force)"
    fi
  done < <(git -C "$SL_REPO" for-each-ref --format='%(refname:short)' refs/heads)
  info "$n local branches in $SL_REPO"
  # worktrees under $SL_WT: classify by their common git dir
  local d common
  for d in "$SL_WT"/*/; do
    d=${d%/}
    [ -e "$d/.git" ] || { [ -d "$d" ] && info "skip $d (not a git worktree)"; continue; }
    common=$(git -C "$d" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || continue
    local br; br=$(git -C "$d" branch --show-current)
    case "$common" in
      "$SL_REPO/.git") info "worktree $d: branch ${br:-detached} belongs to $SL_REPO (covered above)" ;;
      "$ENGINE_BACKUP/.git") info "worktree $d: branch ${br:-detached} lives in $ENGINE_BACKUP/.git (shared object store, already persistent)" ;;
      *) warn "worktree $d: git dir $common is not a known repo; push by hand" ;;
    esac
  done
}

prepare_server() {
  local c clone name
  [ -d "$SERVER_BACKUP/.git" ] || { warn "$SERVER_BACKUP missing"; return; }
  local cur; cur=$(git -C "$SERVER_BACKUP" branch --show-current)
  local checked_out; checked_out=$(git -C "$SERVER_BACKUP" worktree list --porcelain | sed -n 's#^branch refs/heads/##p')
  for clone in $SL_LOCAL/sl-srv-*/; do
    clone=${clone%/}
    [ -d "$clone/.git" ] || continue
    name=${clone#$SL_LOCAL/sl-srv-}
    info "server clone $clone -> $SERVER_BACKUP"
    local b
    while read -r b; do
      [ -n "$b" ] || continue
      local csha ssha target
      csha=$(git -C "$clone" rev-parse "refs/heads/$b")
      ssha=$(git -C "$SERVER_BACKUP" rev-parse -q --verify "refs/heads/$b" || true)
      if [ -z "$ssha" ]; then target=$b; action="create"
      elif [ "$ssha" = "$csha" ]; then info "  $b: already identical"; continue
      elif git -C "$SERVER_BACKUP" merge-base --is-ancestor "$csha" "$ssha" 2>/dev/null; then info "  $b: already contained in $SERVER_BACKUP (clone is behind)"; continue
      elif git -C "$clone" merge-base --is-ancestor "$ssha" "$csha" 2>/dev/null; then
        if printf '%s\n' "$checked_out" | grep -qx "$b"; then target="relaunch/$name-${b//\//-}"; action="checked out in $SERVER_BACKUP, use new name"
        else target=$b; action="fast-forward"; fi
      else target="relaunch/$name-${b//\//-}"; action="diverged, use new name"; fi
      if [ "$DRY" = 1 ]; then info "  [dry-run] $b -> $target ($action)"
      else
        if git -C "$SERVER_BACKUP" fetch -q "$clone" "refs/heads/$b:refs/heads/$target" 2>&1 | sed 's/^/        /'; then info "  $b -> $target ($action)"
        else warn "  $b -> $target failed"; fi
      fi
    done < <(git -C "$clone" for-each-ref --format='%(refname:short)' refs/heads)
  done
  : "$cur"
}

prepare_memory() {
  local d
  mkdir -p "$MEMORY_BACKUP"
  for d in $HOME/.claude/projects/*/memory; do
    [ -d "$d" ] || continue
    info "memory $d -> $MEMORY_BACKUP (newer files only, nothing deleted)"
    if [ "$DRY" = 1 ]; then rsync -rtun --itemize-changes "$d"/ "$MEMORY_BACKUP"/ | sed 's/^/        /' | head -10
    else rsync -rtu "$d"/ "$MEMORY_BACKUP"/; fi
  done
  # reverse direction: nothing is overwritten that is newer locally
  info "memory backup has $(ls "$MEMORY_BACKUP"/*.md 2>/dev/null | wc -l) notes"
}

prepare_secrets() {
  info "secrets and settings that live only on local disk -> $SECRETS (mode 700, never commit)"
  local pairs=(
    "$HOME/.config/gh/hosts.yml|gh-hosts.yml"
    "$HOME/.gitconfig|gitconfig"
    "$HOME/.claude/settings.json|claude-settings.json"
    "$HOME/.bashrc|bashrc")
  local p src dst
  [ "$DRY" = 1 ] || { mkdir -p "$SECRETS"; chmod 700 "$SECRETS"; }
  for p in "${pairs[@]}"; do
    src=${p%%|*}; dst=$SECRETS/${p##*|}
    [ -e "$src" ] || { info "  $src not present, skipped"; continue; }
    if [ "$DRY" = 1 ]; then info "  [dry-run] $src -> $dst"; else
      rm -rf "$dst"; cp -a "$src" "$dst"; chmod -R go-rwx "$dst"; info "  $src -> $dst"; fi
  done
  # venv requirement snapshots (pip freeze) so a rebuild is pinned
  local v
  [ "$DRY" = 1 ] || mkdir -p "$VENV_REQS"
  for v in $VENVS/*/; do
    v=${v%/}
    [ -x "$v/bin/python" ] || continue
    if [ "$DRY" = 1 ]; then info "  [dry-run] freeze $(basename "$v") -> $VENV_REQS/$(basename "$v").txt"
    else uv pip freeze --python "$v/bin/python" 2>/dev/null | grep -v '^-e' > "$VENV_REQS/$(basename "$v").txt"; fi
  done
  info "venv requirement files: $(ls "$VENV_REQS" 2>/dev/null | wc -l) in $VENV_REQS"
}

prepare_dirty() {
  local d any=0
  for d in "$SL_REPO" "$SL_VERIFY" "$SL_WT"/*/ $SL_LOCAL/sl-srv-*/; do
    d=${d%/}
    [ -e "$d/.git" ] || continue
    dirty_report "$d" || any=1
  done
  [ "$any" = 0 ] && info "no uncommitted changes in any worktree or clone" \
    || info "ABOVE: uncommitted changes are NOT backed up; commit them (or copy the files) before the relaunch"
  info "local-only data worth a thought (not backed up here): $SL_LOCAL/slm (re-creatable with --models), $SL_MODELS_LOCAL (re-downloadable), $SCRATCH, published server/Studio test builds"
}

do_prepare() {
  [ "$DRY" = 1 ] && say "DRY RUN: nothing is pushed or written"
  timed "engine-push" prepare_engine
  timed "server-push" prepare_server
  timed "memory" prepare_memory
  timed "secrets" prepare_secrets
  timed "uncommitted" prepare_dirty
  say "prepare done"; printf '%s\n' "${STEP_TIMES[@]}"
}

# ================================================================ steps: check_X (report) and do_X (apply)
# ---- apt
check_apt() { local p bad=(); for p in $APT_PACKAGES; do dpkg -s "$p" >/dev/null 2>&1 || bad+=("$p"); done
  [ ${#bad[@]} = 0 ] && ok "apt packages" || miss "apt packages: ${bad[*]}"; }
do_apt() {
  local p need=()
  for p in $APT_PACKAGES; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
  if [ ${#need[@]} = 0 ]; then ok "all apt packages installed"; return; fi
  apt-get update -qq && apt-get install -y -qq --no-install-recommends "${need[@]}" || warn "apt install failed for some packages"
  have uv || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh; }
}

# ---- dotnet
check_dotnet() {
  if [ -x $DOTNET_LOCAL/dotnet ] && [ "$($DOTNET_LOCAL/dotnet --version 2>/dev/null)" = "$DOTNET_VERSION" ]; then ok "dotnet $DOTNET_VERSION in $DOTNET_LOCAL"
  else miss "$DOTNET_LOCAL ($DOTNET_VERSION)"; fi
}
do_dotnet() {
  [ -x $DOTNET_LOCAL/dotnet ] && [ "$($DOTNET_LOCAL/dotnet --version 2>/dev/null)" = "$DOTNET_VERSION" ] && { ok "dotnet present"; return; }
  local t; t=$(scratch)
  if download "$DOTNET_URL" "$t/sdk.tgz" "$DOTNET_SHA256"; then
    rm -rf $DOTNET_LOCAL.new && mkdir -p $DOTNET_LOCAL.new && tar -xzf "$t/sdk.tgz" -C $DOTNET_LOCAL.new \
      && rm -rf $DOTNET_LOCAL && mv $DOTNET_LOCAL.new $DOTNET_LOCAL
  else
    warn "official download failed; copying $SL_TOOLS/dotnet (slow network FS)"
    rm -rf $DOTNET_LOCAL && cp -a "$SL_TOOLS/dotnet" $DOTNET_LOCAL
  fi
  rm -rf "$t"
  $DOTNET_LOCAL/dotnet --version
}

# ---- tools-local
check_tools() {
  local r=0
  [ -x $TOOLS_LOCAL/minisign ] || { miss "$TOOLS_LOCAL/minisign"; r=1; }
  [ -x $TOOLS_LOCAL/nfpm ] || { miss "$TOOLS_LOCAL/nfpm"; r=1; }
  [ "$r" = 0 ] && ok "minisign $MINISIGN_VERSION, nfpm $NFPM_VERSION"
}
do_tools() {
  mkdir -p $TOOLS_LOCAL
  local t; t=$(scratch)
  if [ ! -x $TOOLS_LOCAL/minisign ]; then
    download "$MINISIGN_URL" "$t/m.tgz" "$MINISIGN_SHA256" && mkdir "$t/m" && tar -xzf "$t/m.tgz" -C "$t/m" \
      && install -m755 "$t/m/minisign-linux/x86_64/minisign" $TOOLS_LOCAL/minisign || warn "minisign install failed"
  fi
  if [ ! -x $TOOLS_LOCAL/nfpm ]; then
    download "$NFPM_URL" "$t/n.tgz" "$NFPM_SHA256" && mkdir "$t/n" && tar -xzf "$t/n.tgz" -C "$t/n" \
      && install -m755 "$t/n/nfpm" $TOOLS_LOCAL/nfpm || warn "nfpm install failed"
  fi
  rm -rf "$t"
}

# ---- shell (~/.bashrc sources env.sh)
check_shell() { grep -q 'tools/agent/env.sh' "$HOME/.bashrc" 2>/dev/null && ok "~/.bashrc sources env.sh" || miss "~/.bashrc does not source $SL_REPO/tools/agent/env.sh"; }
do_shell() {
  grep -q 'tools/agent/env.sh' "$HOME/.bashrc" 2>/dev/null && return
  {
    echo 'export PATH="$HOME/.local/bin:$PATH"'
    # the path variables are written out so that every new shell gets the same layout
    for v in SL_DATA SL_LOCAL SL_TOOLS SL_WORK SL_MODELS SL_MODELS_LOCAL SL_TEST_HOME SL_REPO SL_VERIFY SL_STATE; do echo "export $v=\"${!v}\""; done
    echo "export SL_WORKTREES=\"$SL_WT\""
    echo "[ -f \"$SL_REPO/tools/agent/env.sh\" ] && . \"$SL_REPO/tools/agent/env.sh\""
  } >> "$HOME/.bashrc"
  info "appended the path variables and env.sh to ~/.bashrc"
}

# ---- secrets (gh auth, git global config, claude settings) from $SECRETS
check_secrets() {
  gh auth status >/dev/null 2>&1 && ok "gh authenticated" || miss "gh auth (restore from $SECRETS or run 'gh auth login')"
  [ -f $HOME/.gitconfig ] && ok "~/.gitconfig" || miss "~/.gitconfig"
  [ -f $HOME/.claude/settings.json ] && ok "claude settings" || miss "$HOME/.claude/settings.json"
}
do_secrets() {
  if [ ! -d "$SECRETS" ]; then warn "$SECRETS missing: --prepare-relaunch was not run; log in with 'gh auth login' by hand"; fi
  if [ -f "$SECRETS/gh-hosts.yml" ] && [ ! -f $HOME/.config/gh/hosts.yml ]; then
    mkdir -p $HOME/.config/gh && install -m600 "$SECRETS/gh-hosts.yml" $HOME/.config/gh/hosts.yml; fi
  [ -f $HOME/.gitconfig ] || { [ -f "$SECRETS/gitconfig" ] && cp "$SECRETS/gitconfig" $HOME/.gitconfig; }
  gh auth status >/dev/null 2>&1 && gh auth setup-git 2>/dev/null
  if [ ! -f $HOME/.claude/settings.json ] && [ -f "$SECRETS/claude-settings.json" ]; then
    mkdir -p $HOME/.claude && cp "$SECRETS/claude-settings.json" $HOME/.claude/settings.json; fi
  gh auth status >/dev/null 2>&1 || warn "gh is not authenticated: run 'gh auth login' (pushes to GitHub need it; clone of the public repo does not)"
}

# ---- engine clone
check_repo() {
  if [ -d "$SL_REPO/.git" ]; then
    ok "$SL_REPO on $(git -C "$SL_REPO" branch --show-current), remotes: $(git -C "$SL_REPO" remote | tr '\n' ' ')"
    git -C "$SL_REPO" remote | grep -qx backup || miss "remote backup"
    [ -n "$(git -C "$SL_REPO" config user.email)" ] || miss "git user.email in $SL_REPO"
  else miss "$SL_REPO"; fi
}
do_repo() {
  if [ ! -d "$SL_REPO/.git" ]; then
    git clone "$GITHUB_ENGINE" "$SL_REPO" || { warn "clone failed"; return; }
  fi
  if [ -d "$ENGINE_BACKUP/.git" ] || [ -f "$ENGINE_BACKUP/HEAD" ]; then
    git -C "$SL_REPO" remote | grep -qx backup || git -C "$SL_REPO" remote add backup "$ENGINE_BACKUP"
  fi
  git -C "$SL_REPO" remote set-url origin "$GITHUB_ENGINE"
  git -C "$SL_REPO" fetch -q origin
  git -C "$SL_REPO" remote | grep -qx backup && git -C "$SL_REPO" fetch -q backup 2>&1 | tail -2
  # develop: from origin, then fast-forward to backup if backup is ahead (branches pushed only to backup before the wipe)
  if [ "$(git -C "$SL_REPO" branch --show-current)" != develop ]; then
    git -C "$SL_REPO" show-ref -q refs/heads/develop || git -C "$SL_REPO" branch develop "origin/develop" 2>/dev/null || git -C "$SL_REPO" branch develop backup/develop
    git -C "$SL_REPO" checkout -q develop
  fi
  git -C "$SL_REPO" branch -q --set-upstream-to=origin/develop develop 2>/dev/null
  if git -C "$SL_REPO" show-ref -q refs/remotes/backup/develop \
     && ! git -C "$SL_REPO" merge-base --is-ancestor backup/develop develop 2>/dev/null \
     && git -C "$SL_REPO" merge-base --is-ancestor develop backup/develop 2>/dev/null; then
    git -C "$SL_REPO" merge -q --ff-only backup/develop && info "develop fast-forwarded to backup/develop ($(git -C "$SL_REPO" rev-parse --short develop)): backup was ahead of origin"
  fi
  # other branches that exist only on backup (wip/*, agent branches from before the wipe): create local branches, never touch existing ones
  local b
  while read -r b; do
    b=${b#backup/}
    [ "$b" = HEAD ] && continue
    git -C "$SL_REPO" show-ref -q "refs/heads/$b" || git -C "$SL_REPO" branch -q "$b" "backup/$b" 2>/dev/null
  done < <(git -C "$SL_REPO" for-each-ref --format='%(refname:short)' refs/remotes/backup)
  # commits use the GitHub noreply identity only (docs/handoff/memory/public-repo-privacy.md)
  git -C "$SL_REPO" config user.name "${SL_GIT_NAME:-bglueck}"
  git -C "$SL_REPO" config user.email "${SL_GIT_EMAIL:-826745+BernhardGlueck@users.noreply.github.com}"
  git -C "$SL_REPO" config core.fsmonitor false
  info "develop at $(git -C "$SL_REPO" log -1 --format='%h %s' develop | cut -c1-90)"
}

# ---- verify worktree (detached)
check_verifywt() { [ -e "$SL_VERIFY/.git" ] && ok "$SL_VERIFY ($(git -C "$SL_VERIFY" rev-parse --short HEAD))" || miss "$SL_VERIFY"; }
do_verifywt() {
  [ -e "$SL_VERIFY/.git" ] && { ok "present"; return; }
  [ -d "$SL_REPO/.git" ] || { warn "needs $SL_REPO"; return; }
  git -C "$SL_REPO" worktree prune
  git -C "$SL_REPO" worktree add --detach "$SL_VERIFY" develop
}

# ---- engine source snapshot for the server build (git-less; the server build points SharplightSourceRoot at it)
check_enginesrc() {
  [ -f "$ENGINE_SRC/Sharplight.slnx" ] && ok "$ENGINE_SRC (from $(cat "$ENGINE_SRC/.sl-source-rev" 2>/dev/null | cut -c1-8))" || miss "$ENGINE_SRC"
}
do_enginesrc() {
  [ -d "$SL_REPO/.git" ] || { warn "needs $SL_REPO"; return; }
  local rev; rev=$(git -C "$SL_REPO" rev-parse develop)
  if [ -f "$ENGINE_SRC/Sharplight.slnx" ] && [ "$(cat "$ENGINE_SRC/.sl-source-rev" 2>/dev/null)" = "$rev" ]; then ok "snapshot is current"; return; fi
  mkdir -p "$ENGINE_SRC"
  # extracted over the existing snapshot (keeps bin/obj of earlier server builds); tracked files only, no .git
  git -C "$SL_REPO" archive develop | tar -x -C "$ENGINE_SRC" && echo "$rev" > "$ENGINE_SRC/.sl-source-rev"
  info "snapshot of develop $(git -C "$SL_REPO" rev-parse --short develop)"
}

# ---- Claude memory
memory_dirs() { local r; for r in "$SL_REPO" "$SL_VERIFY" "$ENGINE_BACKUP"; do echo "$HOME/.claude/projects/$(repo_key "$r")/memory"; done; }
check_memory() {
  local d n; n=$(ls "$SL_REPO"/docs/handoff/memory/*.md 2>/dev/null | wc -l)
  for d in $(memory_dirs); do
    if [ "$(ls "$d"/*.md 2>/dev/null | wc -l)" -ge "$n" ] && [ "$n" -gt 0 ]; then ok "memory $d"
    else info "memory notes missing or incomplete in $d"; fi
  done
}
do_memory() {
  local d
  # the repo copy (docs/handoff/memory) first, then a newer local backup if there is one (rsync -u never overwrites newer files)
  for d in $(memory_dirs); do
    mkdir -p "$d"
    [ -d "$SL_REPO/docs/handoff/memory" ] && rsync -rtu "$SL_REPO/docs/handoff/memory"/ "$d"/
    [ -d "$MEMORY_BACKUP" ] && rsync -rtu "$MEMORY_BACKUP"/ "$d"/
  done
  info "restored memory notes into $(memory_dirs | wc -l) project keys"
}

# ---- venvs
venv_list() { echo $IMPORTANT_VENVS; [ "$ALL_VENVS" = 1 ] && echo $OPTIONAL_VENVS; }
check_venvs() {
  local v
  for v in $IMPORTANT_VENVS; do [ -x $VENVS/$v/bin/python ] && ok "venv $v" || miss "venv $v"; done
  for v in $OPTIONAL_VENVS; do [ -x $VENVS/$v/bin/python ] && ok "venv $v (optional)" || info "absent (optional): $v"; done
}
make_venv() {
  local v=$1 req=$VENV_REQS/$1.txt
  [ -x $VENVS/$v/bin/python ] && { ok "venv $v present"; return; }
  if [ ! -s "$req" ]; then warn "no requirement snapshot for $v ($req); create the venv by hand"; return; fi
  have uv || { warn "uv missing"; return; }
  export UV_CACHE_DIR=$UV_CACHE
  uv venv --python 3.12 "$VENVS/$v" -q || { warn "uv venv $v failed"; return; }
  info "installing $(wc -l < "$req") pinned packages into $v"
  # torch +cpu builds come from the PyTorch CPU index; everything else from PyPI
  uv pip install -q --python "$VENVS/$v/bin/python" -r "$req" \
    --extra-index-url "$TORCH_CPU_INDEX" --index-strategy unsafe-best-match || warn "some packages of $v failed to install"
}
do_venvs() { local v; mkdir -p $VENVS; for v in $(venv_list); do make_venv "$v"; done; }

# ---- dev PostgreSQL + PgBouncer (server repo scripts)
pg_psql() { LD_LIBRARY_PATH=$PG_ROOT/usr/lib/x86_64-linux-gnu $PG_ROOT/usr/lib/postgresql/17/bin/psql -h 127.0.0.1 -p $PG_PORT -U postgres -Atc "$1" 2>/dev/null; }
check_pg() {
  [ -x "$PG_ROOT/usr/lib/postgresql/17/bin/postgres" ] && ok "postgres installed in $PG_ROOT" || miss "postgres in $PG_ROOT"
  [ -x "$PG_ROOT/usr/sbin/pgbouncer" ] || [ -x "$PG_ROOT/usr/bin/pgbouncer" ] && ok "pgbouncer" || miss "pgbouncer in $PG_ROOT"
  [ "$(pg_psql 'select 1')" = 1 ] && ok "postgres running on $PG_PORT" || info "postgres not running on $PG_PORT (started by the bootstrap)"
}
do_pg() {
  local S=$SERVER_BACKUP/tools
  [ -x "$S/dev-postgres.sh" ] || { warn "$S/dev-postgres.sh missing"; return; }
  id ubuntu >/dev/null 2>&1 || useradd -m -u 1000 ubuntu 2>/dev/null || warn "user 'ubuntu' missing (PostgreSQL refuses to run as root)"
  [ -x "$PG_ROOT/usr/lib/postgresql/17/bin/postgres" ] || "$S/dev-postgres.sh" install
  { [ -x "$PG_ROOT/usr/sbin/pgbouncer" ] || [ -x "$PG_ROOT/usr/bin/pgbouncer" ]; } || "$S/dev-pgbouncer.sh" install
  [ "$(pg_psql 'select 1')" = 1 ] || "$S/dev-postgres.sh" start
}

# ---- Playwright browsers
check_playwright() { ls -d "$PW_BROWSERS"/chromium-* >/dev/null 2>&1 && ok "playwright chromium in $PW_BROWSERS" || miss "playwright browsers ($PW_BROWSERS)"; }
do_playwright() {
  ls -d "$PW_BROWSERS"/chromium-* >/dev/null 2>&1 && { ok present; return; }
  export PLAYWRIGHT_BROWSERS_PATH=$PW_BROWSERS UV_CACHE_DIR=$UV_CACHE
  uvx --from "playwright==$PLAYWRIGHT_VERSION" playwright install chromium chromium-headless-shell ffmpeg || warn "playwright install failed"
}

# ---- test-home model mirror (tiny models) via the existing script
check_testhome() {
  [ -L $SL_TEST_HOME/models ] && ok "$SL_TEST_HOME/models -> $(readlink $SL_TEST_HOME/models)" || miss "$SL_TEST_HOME/models"
  ls -d $SL_MODELS_LOCAL/*/tiny-* >/dev/null 2>&1 && ok "tiny models mirrored" || miss "tiny models in $SL_MODELS_LOCAL"
}
do_testhome() {
  [ -e "$HOME/models" ] || ln -sfn "$SL_MODELS" "$HOME/models"
  # copies the tiny models from the master store (small files, but slow when that store is a network volume)
  "$SL_REPO/tools/agent/sync-local-models.sh"
}

# ---- dirs and locks
check_dirs() {
  local d; for d in $SL_WT $SL_STATE/.sl-build-slots $SL_WORK/tmp $SCRATCH; do [ -d "$d" ] && ok "$d" || miss "$d"; done
  [ -e $SL_STATE/.sl-verify.lock ] && ok "verify lock file" || miss "$SL_STATE/.sl-verify.lock"
}
do_dirs() { mkdir -p $SL_WT $SL_STATE/.sl-build-slots $SL_WORK/tmp $SCRATCH; touch $SL_STATE/.sl-verify.lock; }

# ---- server clones ($SL_LOCAL/sl-srv-clean, $SL_LOCAL/sl-srv-ui) from $SL_SERVER_BACKUP
SERVER_CLONES="clean:cleanup/backend ui:cleanup/ui"
check_serverclones() {
  local p; for p in $SERVER_CLONES; do [ -d "$SL_LOCAL/sl-srv-${p%%:*}/.git" ] && ok "$SL_LOCAL/sl-srv-${p%%:*}" || info "absent (optional): $SL_LOCAL/sl-srv-${p%%:*}"; done
}
do_serverclones() {
  local p n b
  for p in $SERVER_CLONES; do
    n=${p%%:*}; b=${p#*:}
    [ -d "$SL_LOCAL/sl-srv-$n/.git" ] && continue
    git -C "$SERVER_BACKUP" show-ref -q "refs/heads/$b" || { info "branch $b not in $SERVER_BACKUP; skipping $SL_LOCAL/sl-srv-$n"; continue; }
    git clone -q -b "$b" "$SERVER_BACKUP" "$SL_LOCAL/sl-srv-$n" \
      && git -C "$SL_LOCAL/sl-srv-$n" config user.name "$(git -C "$SERVER_BACKUP" config user.name)" \
      && git -C "$SL_LOCAL/sl-srv-$n" config user.email "$(git -C "$SERVER_BACKUP" config user.email)" \
      && info "cloned $SL_LOCAL/sl-srv-$n on $b"
  done
}

# ---- self-hosted feed runner (optional; docs/model-feed.md). Registration needs a token from the owner: never create one.
check_runner() {
  [ -n "${SL_RUNNER_DIR:-}" ] || { info "SL_RUNNER_DIR not set (no feed runner on this machine)"; return; }
  [ -f "$SL_RUNNER_DIR/.runner" ] && ok "runner configured in $SL_RUNNER_DIR" || info "runner not registered in $SL_RUNNER_DIR (ask the owner for an org registration token)"
}
do_runner() {
  [ -n "${SL_RUNNER_DIR:-}" ] && [ -f "$SL_RUNNER_DIR/.runner" ] || { check_runner; return; }
  pgrep -u "${SL_RUNNER_USER:-$(id -un)}" -f Runner.Listener >/dev/null && { ok "runner running"; return; }
  su "${SL_RUNNER_USER:-$(id -un)}" -c "cd '$SL_RUNNER_DIR' && nohup ./run.sh > '$SL_RUNNER_DIR/runner.log' 2>&1 &"
  info "runner started from $SL_RUNNER_DIR"
}

# ---- llama.cpp release (optional)
check_llama() { [ -x $SL_LOCAL/llama-local/$LLAMA_BUILD/llama-$LLAMA_BUILD/llama-cli ] && ok "llama.cpp $LLAMA_BUILD" || info "absent (optional): $SL_LOCAL/llama-local/$LLAMA_BUILD"; }
do_llama() {
  local d=$SL_LOCAL/llama-local/$LLAMA_BUILD
  [ -x "$d/llama-$LLAMA_BUILD/llama-cli" ] && { ok present; return; }
  mkdir -p "$d" && download "$LLAMA_URL" "$d/llama.tar.gz" "$LLAMA_SHA256" && tar -xzf "$d/llama.tar.gz" -C "$d" && rm -f "$d/llama.tar.gz"
}

# ---- real models (optional)
models_table() { printf '%s\n' "$MODELS_SMALL"; [ "$MODELS_BIGF" = 1 ] && printf '%s\n' "$MODELS_BIG"; }
check_models() { local d; for d in hf/Qwen3-0.6B gguf/Qwen3-0.6B gguf/Qwen3-4B; do [ -d $SL_MODELS_LOCAL/$d ] && ok "model $d" || info "absent (optional --models): $d"; done
  [ -f $SL_LOCAL/slm/qwen3-0.6b-q8_0.slm ] && ok "$SL_LOCAL/slm qwen3-0.6b" || info "absent (optional --models): $SL_LOCAL/slm/qwen3-0.6b-q8_0.slm"; }
do_models() {
  local py=$VENVS/ref/bin/python
  [ -x $py ] || { warn "needs the ref venv"; return; }
  local table; table=$(models_table)
  local cat=$SL_REPO/catalog/verified-models.json
  MODELS_TABLE="$table" CATALOG="$cat" LIST_ONLY=$MODELS_LIST DEST=$SL_MODELS_LOCAL $py -I - <<'PY'
import json, os, sys
from huggingface_hub import HfApi, snapshot_download
pins = {}
try:
    for e in json.load(open(os.environ["CATALOG"]))["entries"]:
        for v in e.get("variants", []):
            if v.get("repository") and v.get("revision"):
                pins.setdefault(v["repository"], v["revision"])
except Exception as ex:
    print("    (catalog pins unavailable: %s)" % ex)
api = HfApi(token=os.environ.get("HF_TOKEN"))
rows = []
for line in os.environ["MODELS_TABLE"].splitlines():
    if not line.strip():
        continue
    repo, target, inc = (line.split("|") + ["", ""])[:3]
    pats = [p for p in inc.split(",") if p] or None
    rev = pins.get(repo)
    try:
        info = api.model_info(repo, revision=rev, files_metadata=True)
        files = [s for s in info.siblings if pats is None or any(__import__("fnmatch").fnmatch(s.rfilename, p) for p in pats)]
        size = sum(s.size or 0 for s in files)
    except Exception as ex:
        print("    %-45s UNAVAILABLE (%s)" % (repo, str(ex)[:60])); continue
    rows.append((repo, target, pats, rev, size))
    print("    %-45s %7.2f GB  rev %s  -> %s" % (repo, size / 1e9, (rev or "main")[:8], target))
print("    total %.1f GB" % (sum(r[4] for r in rows) / 1e9))
if os.environ["LIST_ONLY"] == "1":
    sys.exit(0)
for repo, target, pats, rev, size in rows:
    dest = os.path.join(os.environ["DEST"], target)
    if os.path.isdir(dest) and os.listdir(dest) and not os.path.exists(dest + "/.cache"):
        print("    present, skipped: %s" % target); continue
    print("    downloading %s ..." % repo, flush=True)
    snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=pats, token=os.environ.get("HF_TOKEN"))
PY
  [ "$MODELS_LIST" = 1 ] && return
  # converted .slm files for the server tests
  local row src dst
  mkdir -p $SL_LOCAL/slm
  while IFS='|' read -r src dst; do
    [ -n "$src" ] || continue
    [ -f "$SL_LOCAL/slm/$dst" ] && continue
    [ -f "$SL_MODELS_LOCAL/$src" ] || { info "skip $dst: $SL_MODELS_LOCAL/$src absent"; continue; }
    (cd "$SL_REPO" && nice -n 15 dotnet run -c Release --project samples/Sharplight.Convert -- convert "$SL_MODELS_LOCAL/$src" "$SL_LOCAL/slm/$dst") || warn "convert $dst failed"
  done <<< "$SLM_CONVERT"
}

# ================================================================ verification
verify() {
  . "$SL_REPO/tools/agent/env.sh" 2>/dev/null
  say "verification"
  printf '    dotnet:  '; dotnet --version 2>&1 | tail -1
  printf '    gh:      '; gh auth status 2>&1 | grep -m1 -E 'Logged in|not logged' || echo "unknown"
  printf '    psql:    '; pg_psql 'select version()' | cut -c1-40 || true
  printf '    git:     '; git -C "$SL_REPO" log -1 --format='%h %s' 2>/dev/null | cut -c1-70
  printf '    vulkan:  '; ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null | tr '\n' ' '; echo
  printf '    nvidia:  '; nvidia-smi -L 2>/dev/null | wc -l | tr -d '\n'; echo " GPUs"
  if [ -d "$SL_REPO" ]; then
    info "sharplight --devices (Vulkan + CUDA); builds the CLI first (a few minutes)"
    (cd "$SL_REPO" && nice -n 15 ionice -c3 dotnet run -c Release --project samples/Sharplight.Cli -- --devices 2>&1 | tail -15) || warn "devices listing failed"
    local proj=${VERIFY_TEST_PROJECT:-Sharplight.Core.Tests} cls=${VERIFY_TEST_CLASS:-*Tokenizer*}
    info "quick test: $proj -class '$cls' (override with VERIFY_TEST_PROJECT / VERIFY_TEST_CLASS)"
    (cd "$SL_REPO" && nice -n 15 ionice -c3 dotnet build "tests/$proj" -c Release -m:4 2>&1 | grep -E "error |Error\(s\)|Elapsed" | tail -3
     "$SL_REPO/tools/agent/sl-test" "$proj" -class "$cls" 2>&1 | tail -4) || warn "quick test failed"
  fi
}

# ================================================================ drivers
STEPS="apt dotnet tools shell secrets repo verifywt enginesrc memory venvs pg playwright testhome dirs serverclones runner"
[ "$LLAMA" = 1 ] && STEPS="$STEPS llama"
[ "$MODELS" = 1 ] && STEPS="$STEPS models"

if [ "$MODE" = prepare ]; then do_prepare; exit 0; fi

if [ "$MODE" = check ]; then
  say "CHECK (nothing is changed)"
  for s in apt dotnet tools shell secrets repo verifywt enginesrc memory venvs pg playwright testhome dirs serverclones runner llama models; do
    want "$s" || continue
    echo "-- $s"; "check_$s"
  done
  echo
  if [ ${#MISSING[@]} = 0 ]; then echo "RESULT: all present"; exit 0; fi
  echo "RESULT: ${#MISSING[@]} missing:"; printf '  - %s\n' "${MISSING[@]}"; exit 1
fi

[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
# the order matters: secrets (gh) before repo; repo before verifywt/enginesrc; dotnet/pg need nothing local
for s in $STEPS; do
  want "$s" || continue
  timed "$s" "do_$s"
done
if [ "$VERIFY" = 1 ] && [ -z "$ONLY" ]; then verify; fi
say "summary"
printf '%s\n' "${STEP_TIMES[@]}"
echo "--- final check"
if [ -n "$ONLY" ]; then exec "$0" --check --only "$ONLY"; fi
exec "$0" --check
