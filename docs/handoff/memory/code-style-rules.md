---
name: code-style-rules
description: "Sharplight C# style rules from the user - one type per file, braces always, extension blocks, documented public API, readable samples"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-11-13T03:52:37.179Z
---

User (2026-11-13) asked for a full cleanup and set these rules for all code (including agents' code):
- Every class/struct/record/interface/enum in its own file; only extension members may share a file with their target type.
- Extensions use C# 14 `extension` blocks, not old `static class X { static T M(this ...) }` style.
- Never `if`/`for`/`foreach`/`while`/`else` with a single-line body without braces — always curly braces (they break during merges).
- Proper visibility on every type/member (internal unless meant to be public API); clearly documented public API (XML docs) for the engine.
- Remove dead code and duplication; readable code. Samples must look professional (not "junior" style): structured, small methods, clear option parsing.

**Why:** code had grown partly badly readable through fast parallel agent work; merges broke brace-less statements.

**How to apply:** follow in every edit and put these rules into every agent brief; enforce via .editorconfig/analyzers where possible. Related: [[git-worktrees-parallel]], [[format-agnostic-core]].
