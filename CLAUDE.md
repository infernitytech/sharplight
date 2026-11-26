# Notes for Claude Code

Work continues from the 2026-11-26 handoff (development moved from one RunPod pod to several machines). At the start of
a session:

1. If `~/.claude/projects/<this clone's path with non-alphanumerics replaced by ->/memory/MEMORY.md` does not exist yet,
   run `docs/handoff/restore.sh` to install the project memory notes (goals, owner rules, decisions).
2. Read `docs/handoff/README.md`: long-term goals, owner rules, the state of `develop` / `main` / releases, the open
   branches, the work queue, the merge flow and feed publishing. Machine setup and merge scripts: `tools/agent/`.

Rules in short (details in docs/handoff/memory): code style per docs/code-style.md; format-agnostic core (format names
only in `Sharplight.Import.*`); only generic operations in Core; importers and graph generation stay outside Core;
everything storable in `.slm`; work in git worktrees and merge to `develop`; commits use the noreply identity only;
privacy scan before every push; push only when the owner says so.
