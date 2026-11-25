#!/bin/bash
# ptxas 13.0 on PTX from cicc 13.2: lower the PTX ISA version to 9.0
args=()
for a in "$@"; do if [[ "$a" == *.ptx && -f "$a" ]]; then sed -i -E 's/^\.version 9\.[1-9]/.version 9.0/' "$a"; fi; args+=("$a"); done
[ -n "$PTXAS_LOG" ] && echo "ptxas13.0 $*" >> $PTXAS_LOG
exec $PTXAS_13_0 "${args[@]}"
