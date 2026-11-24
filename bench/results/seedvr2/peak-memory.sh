#!/bin/bash
# peak-memory.sh <log> <command...>: runs the command and records the peak device memory nvidia-smi reports for its process (100 ms samples;
# the largest entry: a Vulkan process is listed twice with the same memory).
log=$1; shift
"$@" &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
    used=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p && $2>m {m=$2} END {print m+0}')
    if [ -n "$used" ] && [ "$used" -gt "$peak" ]; then peak=$used; fi
    sleep 0.1
done
wait $pid
status=$?
echo "peak_mib $peak" >> "$log"
exit $status
