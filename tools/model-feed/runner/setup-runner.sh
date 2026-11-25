#!/usr/bin/env bash
# Sets up a self-hosted GitHub Actions runner for the large model-feed builds (docs/model-feed.md, "Self-hosted feed runner").
#
#   sudo RUNNER_URL=https://github.com/<org or owner/repo> RUNNER_TOKEN=<registration token> \
#     tools/model-feed/runner/setup-runner.sh [--user ghrunner] [--name feed-1] [--work /home/ghrunner/_work] \
#                                             [--feed-work /home/ghrunner/feed-work] [--version 2.329.0]
#
# The registration token comes from GitHub (repository or organization Settings -> Actions -> Runners -> New self-hosted runner); it
# expires after an hour. This script never fetches or creates tokens and changes no GitHub settings.
#
# What it does (run as root once; the runner itself never runs as root):
#   1. creates the non-root user (no sudo) and the work directories;
#   2. installs the packages the jobs need (jq, git, curl, python3, unzip, the AWS CLI v2 when missing); .NET is installed by
#      actions/setup-dotnet in each job into the runner's tool cache;
#   3. downloads the Actions runner, registers it with the labels self-hosted, linux, x64, sharplight-feed (ephemeral: no; one
#      job at a time) and installs it as a systemd service under that user;
#   4. installs a job-started hook that refuses every job except model-feed on develop (push or workflow_dispatch) of the
#      repository set in ALLOWED_REPOSITORY (default: sharplight), so a public repository's other workflows or pull requests from
#      forks cannot run on this machine even if a label matches.
set -euo pipefail

user=ghrunner
name="feed-$(hostname -s)"
work=""
feed_work=""
version=2.329.0
while [ $# -gt 0 ]; do
  case "$1" in
    --user) user=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --work) work=$2; shift 2 ;;
    --feed-work) feed_work=$2; shift 2 ;;
    --version) version=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

: "${RUNNER_URL:?set RUNNER_URL to https://github.com/<org> or https://github.com/<owner>/<repo>}"
: "${RUNNER_TOKEN:?set RUNNER_TOKEN to a registration token from Settings -> Actions -> Runners}"
allowed_repository=${ALLOWED_REPOSITORY:-sharplight}
if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (it creates the user and the systemd service); the runner itself runs as $user." >&2
  exit 1
fi

home="/home/$user"
work=${work:-$home/_work}
feed_work=${feed_work:-$home/feed-work}
runner_dir="$home/actions-runner"

# 1. User and directories.
if ! id "$user" > /dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "$user"
fi
install -d -o "$user" -g "$user" "$work" "$feed_work" "$runner_dir"

# 2. Packages.
if command -v apt-get > /dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get update -q
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q jq git curl python3 unzip ca-certificates libicu-dev
fi
if ! command -v aws > /dev/null; then
  tmp=$(mktemp -d)
  curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o "$tmp/awscli.zip"
  unzip -q "$tmp/awscli.zip" -d "$tmp"
  "$tmp/aws/install"
  rm -rf "$tmp"
fi

# 3. The runner.
if [ ! -x "$runner_dir/config.sh" ]; then
  curl -fsSL "https://github.com/actions/runner/releases/download/v$version/actions-runner-linux-x64-$version.tar.gz" \
    | sudo -u "$user" tar xz -C "$runner_dir"
fi

# 4. The job-started hook (runs before every job; a non-zero exit fails the job before any of its steps).
hook="$runner_dir/feed-job-guard.sh"
cat > "$hook" << EOF
#!/usr/bin/env bash
# Only the model feed's publication of $allowed_repository on develop runs on this machine (docs/model-feed.md, "Self-hosted feed runner").
repository_name="\${GITHUB_REPOSITORY#*/}"
case "\${GITHUB_EVENT_NAME:-}" in
  push|workflow_dispatch) ;;
  *) echo "feed runner: refusing a \${GITHUB_EVENT_NAME:-unknown} event" >&2; exit 1 ;;
esac
if [ "\$repository_name" != "$allowed_repository" ] || [ "\${GITHUB_REF:-}" != "refs/heads/develop" ]; then
  echo "feed runner: refusing \${GITHUB_REPOSITORY:-?} at \${GITHUB_REF:-?}" >&2
  exit 1
fi
case "\${GITHUB_WORKFLOW_REF:-}" in
  "\$GITHUB_REPOSITORY/.github/workflows/model-feed.yml@refs/heads/develop") ;;
  *) echo "feed runner: refusing workflow \${GITHUB_WORKFLOW_REF:-?}" >&2; exit 1 ;;
esac
EOF
chown "$user:$user" "$hook"
chmod 755 "$hook"
cat > "$runner_dir/.env" << EOF
ACTIONS_RUNNER_HOOK_JOB_STARTED=$hook
LANG=C.UTF-8
EOF
chown "$user:$user" "$runner_dir/.env"

if [ ! -f "$runner_dir/.runner" ]; then
  sudo -u "$user" "$runner_dir/config.sh" --unattended --url "$RUNNER_URL" --token "$RUNNER_TOKEN" --name "$name" \
    --labels sharplight-feed --work "$work" --replace
fi
(cd "$runner_dir" && ./svc.sh install "$user" && ./svc.sh start)

echo "Runner $name is installed as a service under $user (labels self-hosted, linux, x64, sharplight-feed)."
echo "Next (owner, on GitHub): allow the repository in the runner group (organization runners), set the repository variables"
echo "MODEL_FEED_SELF_HOSTED=true and MODEL_FEED_WORK_DIR=$feed_work (docs/model-feed.md, 'Self-hosted feed runner')."
