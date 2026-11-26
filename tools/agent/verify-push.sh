#!/usr/bin/env bash
# usage: verify-push.sh <tag> [suite...]     e.g. verify-push.sh r49 Core.Tests Cuda.Tests Vulkan.Tests
# Verifies develop of $SL_REPO (verify.sh: YAML, style, build, suites) and pushes develop to origin only when the build passed,
# every suite ended with exit=0, the privacy scan of the new commits is clean and no tracked file name is invalid on Windows.
# Then pushes to the `backup` remote when the clone has one.
#
# Privacy scan: added lines must not contain home paths or personal mail addresses. Put further private words (e.g. the owner's
# private name) into SL_PRIVACY_EXTRA as an extended-regex alternative list ("word1|word2"); keep that value out of every repo.
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
out=$("$here/verify.sh" "$@" 2>&1); echo "$out" | tail -8
echo "$out" | grep -q "RESULT-$1 done" || { echo "NOT PUSHED: yaml/style/build failed"; exit 1; }
echo "$out" | grep -E 'exit=[1-9]' && { echo "NOT PUSHED: suite failures"; exit 1; }
cd "$SL_REPO" || exit 1
git fetch -q origin develop
pattern='/(root|home)/|@g[m]ail'
[ -n "${SL_PRIVACY_EXTRA:-}" ] && pattern="$pattern|$SL_PRIVACY_EXTRA"
git diff origin/develop..develop | grep -nE "^\+.*($pattern)" && { echo "NOT PUSHED: privacy"; exit 1; }
git log origin/develop..develop --format='%ae%n%ce' | sort -u | grep -vE '@users\.noreply\.github\.com$' \
    && { echo "NOT PUSHED: commit author/committer is not a GitHub noreply address"; exit 1; }
git ls-files | grep -E '[<>:"|?*\\]' && { echo "NOT PUSHED: file names invalid on Windows"; exit 1; }
timeout 900 git push origin develop 2>&1 | tail -1
if git remote | grep -qx backup; then
    timeout 900 git push -q backup develop 2>&1 | tail -1
fi
