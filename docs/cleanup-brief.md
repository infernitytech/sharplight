# Cleanup brief (shared by all cleanup agents)

Read docs/code-style.md first; it is the rule set. Mechanical rules (braces, explicit accessibility modifiers,
file-scoped namespaces, unused usings) were already applied by `dotnet format` in commit 9f361d2. Keep them intact in
every line you touch, and use braces on every if/else/for/foreach/while/do/using/lock body.

## Your job (semantic cleanup, behavior-preserving)
1. One type per file. Split every file that holds more than one top-level type (class, struct, record, interface,
   enum, delegate) into one file per type, named after the type, in a folder mirroring the namespace. Large or
   externally used nested types become top-level types. Partial classes may stay split across several files
   (Foo.cs, Foo.Ops.cs) when the type is big. Extension members may share a file with the type they extend.
2. Extension methods: convert every old `static ... M(this T x)` extension method to a C# 14 `extension(T x) { ... }`
   block (LangVersion is 15). Keep the call sites working.
3. Visibility: every type and member must have the narrowest correct accessibility. A type or member may become `internal`
   only if NO other production assembly uses it. Grep the whole repo (src/, samples/) before narrowing. Tests get access
   through `InternalsVisibleTo` in the project file (`<InternalsVisibleTo Include="Sharplight.X.Tests" />`), not by
   widening visibility. Classes not designed for inheritance are `sealed`.
4. Public API docs: every public type and member gets XML docs that state contracts (shapes, units, ownership,
   thread-safety, exceptions), not the obvious. Public API is the product: describe it clearly.
5. Remove dead code: unused types, members, parameters (IDE0060, IDE0051, IDE0052, CS0169, CS9113 warnings), commented-out
   code and stale experiment switches. Keep diagnostic environment switches that are still useful, and list each
   project's switches in one place (a static class `DiagnosticSwitches` or a doc comment block).
6. Remove duplication: factor repeated logic into one helper within your projects.
7. Readability: split dense lines (multiple statements per line, comma-chained assignments like `int a = 1, b = 2;`
   with complex initializers, `x = ...; y = ...;` on one line). Use descriptive names instead of cryptic local names
   where it helps; established math/domain short names (T, D, HQ, KV, i/j loops) are fine. Long methods should be
   broken into well-named private methods. Comments explain why.

## Hard constraints
- Only edit files inside your assigned projects (listed in your task). Other agents clean up the rest in parallel, and
  overlapping edits will cause merge conflicts.
- Do NOT rename or change the signature of anything public or anything used by another assembly. Do NOT change
  namespaces. Splitting files and narrowing visibility per rule 3 are fine.
- No behavior changes and no performance changes. Hot loops (CPU kernels, lowering) must not get slower: no LINQ or
  allocations in hot paths, and keep `[MethodImpl]` attributes, `unsafe` and spans as they are.
- .slang kernel files: leave them alone.
- Build with `dotnet build Sharplight.slnx -c Release` and get 0 errors. In your projects the warnings
  IDE0005/IDE0011/IDE0040/IDE0051/IDE0052/IDE0060/IDE0161/CS0169/CS9113/CS1573 must be 0. Check with
  `dotnet build src/X -c Release 2>&1 | grep -E "warning (IDE|CS)"`.
- Run the test suites named in your task (`dotnet run --project tests/Sharplight.X.Tests -c Release`); all must pass as
  before. The GPU is shared with other agents and apps, so retry a timing-only failure once before investigating.
- Work in your worktree. Commit in small logical commits (e.g. "Cpu: one type per file", "Cpu: extension blocks",
  "Cpu: XML docs") ending with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push and do not
  touch master.
- Final report: what was split/removed/narrowed, any public API you thought should change but didn't (list it as a
  suggestion), test totals, and remaining warnings.
