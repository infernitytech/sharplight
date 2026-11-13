#!/usr/bin/env python3
"""Checks the one-type-per-file rule of docs/code-style.md.

Every C# file may declare at most one top-level type, named like the file (partial files `Foo.Part.cs` and generic
arity files `Foo`1.cs` are accepted). A static `*Extensions` class holding extension members may share a file with the
type it extends. Exit code 1 lists the violations.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TYPE_DECL = re.compile(
    r'^(?:\[[^\]]*\]\s*)*(?:(?:public|internal|private|protected|file|sealed|static|abstract|readonly|partial|unsafe|ref|new)\s+)*'
    r'(class|struct|interface|enum|delegate|record(?:\s+(?:class|struct))?)\s+(?:[\w<>\[\],\s.?]*?\s)?(\w+)\s*(?:<|\(|:|\{|$|;|where)')
SKIP_DIRS = {'bin', 'obj', '.claude', '.work', 'worktrees'}


def top_level_types(path: pathlib.Path):
    types = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line[0].isspace():
            continue   # nested declarations are indented; file-scoped namespaces keep top-level types at column 0
        match = TYPE_DECL.match(line)
        if match:
            types.append(match.group(2))
    return types


def main() -> int:
    violations = []
    for path in sorted(ROOT.rglob('*.cs')):
        if SKIP_DIRS.intersection(path.relative_to(ROOT).parts):
            continue
        types = top_level_types(path)
        stem = path.name.split('.')[0].split('`')[0]
        primary = [t for t in types if not t.endswith('Extensions')]
        if len(primary) > 1 or (primary and primary[0] != stem and not path.name.startswith(primary[0] + '.')):
            violations.append(f'{path.relative_to(ROOT)}: {", ".join(types)}')
    for violation in violations:
        print(violation)
    print(f'{len(violations)} file(s) violate the one-type-per-file rule.' if violations else 'OK: one type per file.')
    return 1 if violations else 0


if __name__ == '__main__':
    sys.exit(main())
