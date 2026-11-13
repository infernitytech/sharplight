# Sharplight code style

These rules apply to all C# code in the repository (libraries, samples, tests). `.editorconfig` enforces the mechanical
ones as build warnings; the rest is checked in review.

## Files and types

- One type per file: every `class`, `struct`, `record`, `interface`, `enum` and `delegate` lives in its own file named after
  the type (`GgufImporter.cs`). Generic arity goes into the name only when two types differ by arity (`Foo.cs`, `Foo`1.cs`).
- The only exception: extension members may live in the same file as the type they extend.
- Nested types are allowed when they are an implementation detail of the outer type and are small; anything that is
  used outside the outer type, or is larger than a few lines, becomes a top-level type in its own file.
- Folder structure mirrors namespaces inside each project.
- File-scoped namespaces.

## Extensions

Use C# 14 extension blocks, not the old `static` method style:

```csharp
public static class TensorExtensions
{
    extension(ITensor tensor)
    {
        /// <summary>Number of elements.</summary>
        public long ElementCount => tensor.Shape.Elements;

        public ITensor Reshape(params long[] shape) => ...;
    }
}
```

## Control flow

- Always use braces for the bodies of `if`, `else`, `for`, `foreach`, `while`, `do`, `using` and `lock`, even for one
  statement. Brace-less bodies break silently during merges.
- Prefer early returns over deep nesting; keep methods short and focused.

## Visibility and API surface

- Every type and member has an explicit accessibility modifier.
- Make everything `internal` (or `private`) unless it is intentionally part of the public API. Tests use
  `InternalsVisibleTo` rather than widening visibility.
- `sealed` by default for classes that are not designed for inheritance.
- Every public type and member has XML documentation (`<summary>`, plus `<param>`, `<returns>`, `<exception>` where
  they add information). Documentation states contracts (units, shapes, ownership, thread safety), not the obvious.

## Readability

- No dead code: unused types, members, parameters, commented-out code and stale experiment switches are removed.
  Diagnostic environment switches that are still useful are documented in one place per project.
- No duplication: shared logic is factored into one helper.
- Names say what things are; avoid abbreviations except established domain ones (KV, GEMM, RoPE, ...).
- One statement per line; no dense multi-assignment lines. Long expressions are split with named locals.
- Comments explain *why*, not *what*.

## Samples

Samples are documentation. They must read like production code:

- `Program.cs` contains only top-level orchestration; option parsing lives in an options type, distinct steps in small
  well-named methods or classes.
- Command-line parsing uses `System.CommandLine` (or one small, typed options parser shared by the samples), with help
  text for every option.
- No magic numbers or ad-hoc string slicing; errors are reported with clear messages and exit codes.
