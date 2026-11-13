#!/bin/bash
. "$(dirname "$0")/models.sh"
for e in "${MODELS[@]}"; do for b in cuda13 vulkan; do
  echo "== $b ${e%%|*}"; "$(dirname "$0")/llama_single.sh" $b "${e%%|*}" "${e##*|}"
done; done
