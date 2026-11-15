# Notes for Claude Code

Work is continuing from the 2026-11-15 handoff to a new machine. At the start of a session:

1. If `~/.claude/projects/<this clone's path with / replaced by ->/memory/MEMORY.md` does not exist yet, run
   `docs/handoff/restore.sh` to install the project memory notes (goals, rules, workstreams).
2. Read `docs/handoff/README.md`: long-term goal, state of `main`, the `wip/*` branches (fetch them from origin) and the
   work queue. Continue the autonomous loop from there.

Rules in short (details in docs/handoff/memory): code style per docs/code-style.md; format-agnostic core (format names
only in `Sharplight.Import.*`); importers and graph generation stay outside Core; everything storable in `.slm`; work in
git worktrees and merge to `main`; commits use the noreply email only; push only when the user says so.
