#!/usr/bin/env python3
"""Removes machine-specific details from raw benchmark output before it is committed: the home directory becomes '~',
the user name is dropped, and the process lines of *.gpu.txt keep only pid, executable name and memory.
usage: sanitize_results.py <results dir>"""
import os, re, sys

home = os.path.expanduser("~")
user = os.path.basename(home)
root = sys.argv[1]
for dirpath, _, files in os.walk(root):
    for f in files:
        p = os.path.join(dirpath, f)
        try:
            s = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        t = s.replace(home, "~").replace(f"/{user}/", "/USER/")
        if f.endswith(".gpu.txt"):
            out = []
            for line in t.splitlines():
                m = re.match(r"^(\d+), ([^,]*)(, .*)?$", line)
                if m and not line.startswith("gpu"):
                    exe = os.path.basename(m.group(2).split(" ")[0])
                    line = f"{m.group(1)}, {exe}{m.group(3) or ''}"
                out.append(line)
            t = "\n".join(out) + "\n"
        if t != s:
            open(p, "w", encoding="utf-8").write(t)
