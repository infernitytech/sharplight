#!/bin/bash
# Runs "$@" and samples its major page faults, read bytes and RSS, the cgroup's file cache and major faults every 0.5 s.
# SAMPLE_LOG=<file>: also records the line count of the process's log (marks its phases).
# usage: sample_faults.sh <sample file> <cmd...>   (the command must exec into the measured process: use env, not a shell)
F=$1; shift
"$@" &
pid=$!
t0=$(date +%s.%N)
echo "t_s proc_majflt proc_read_mb proc_rss_mb cg_file_mb cg_anon_mb cg_pgmajfault log_lines" > "$F"
while kill -0 $pid 2>/dev/null; do
  st=$(awk '{print $12}' /proc/$pid/stat 2>/dev/null); rb=$(awk '/^read_bytes/{print int($2/1048576)}' /proc/$pid/io 2>/dev/null)
  rss=$(awk '/^VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null)
  cg=$(awk '/^file /{f=int($2/1048576)} /^anon /{a=int($2/1048576)} /^pgmajfault /{m=$2} END{print f, a, m}' /sys/fs/cgroup/memory.stat)
  t=$(awk -v a="$(date +%s.%N)" -v b="$t0" "BEGIN{printf \"%.1f\", a-b}")
  ll=NA; [ -n "$SAMPLE_LOG" ] && ll=$(wc -l < "$SAMPLE_LOG" 2>/dev/null)
  echo "$t ${st:-NA} ${rb:-NA} ${rss:-NA} $cg ${ll:-NA}" >> "$F"
  sleep 0.5
done
wait $pid
