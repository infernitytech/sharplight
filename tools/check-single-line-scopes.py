#!/usr/bin/env python3
"""Checks the "no single-line scopes" rule of docs/code-style.md.

A braced block (the body of a statement, method, constructor, accessor, local function or lambda) must not sit on one line:
`if (x) { return; }`, `public void Dispose() { }`, `x => { Run(); }` and `get { return _x; }` are reported. Object and
collection initializers (`new Foo { A = 1 }`, `new() { ... }`), auto-property accessor lists (`{ get; set; }`), empty
lambdas used as no-op callbacks (`_ => { }`) and code inside string literals (e.g. generated shader source) are not blocks
and are allowed.

usage: check-single-line-scopes.py [--fix-list] [paths...]   (default: src samples apps tests bench)
Exit code 1 when violations are found.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIRS = ["src", "samples", "apps", "tests", "bench"]

# A block opener: something that introduces a statement body, then '{' ... '}' closed on the same line.
OPENER = re.compile(
    r"(?:\)|=>|\belse|\btry|\bfinally|\bdo|\bget|\bset|\binit|\badd|\bremove|\bunsafe|\bchecked|\bunchecked)\s*\{(?P<body>[^{}]*)\}")
# Object creation with arguments followed by an initializer: new Foo(1) { A = 2 } / new(x) { ... }.
INITIALIZER = re.compile(r"\bnew\b[^;=]*\)\s*\{[^{}]*\}")
# 'with' expressions and anonymous types are not blocks either.
NON_BLOCK = re.compile(r"\bwith\s*\{|\bnew\s*\{")


def blank_literals(text: str) -> str:
    """Blanks the contents of comments, string literals (regular, verbatim, interpolated, raw, multi-line) and char literals,
    keeping line breaks so line numbers stay valid. Interpolation holes are blanked too: they hold expressions, not blocks."""
    out = []
    i = 0
    n = len(text)

    def blank(segment: str) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in segment)

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            out.append(blank(text[i:end]))
            i = end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(blank(text[i:end]))
            i = end
            continue
        if c in "$@\"":
            j = i
            while j < n and text[j] in "$@":
                j += 1
            if j < n and text[j] == '"' and (j > i or c == '"'):
                if text.startswith('"""', j):
                    quotes = j
                    while quotes < n and text[quotes] == '"':
                        quotes += 1
                    delimiter = '"' * (quotes - j)
                    end = text.find(delimiter, quotes)
                    end = n if end < 0 else end + len(delimiter)
                    out.append('""' + blank(text[i + 2:end]))
                    i = end
                    continue
                verbatim = "@" in text[i:j]
                k = j + 1
                while k < n:
                    if text[k] == "\\" and not verbatim:
                        k += 2
                        continue
                    if text[k] == '"':
                        if verbatim and k + 1 < n and text[k + 1] == '"':
                            k += 2
                            continue
                        break
                    if text[k] == "\n" and not verbatim:
                        break
                    k += 1
                out.append('""' + blank(text[i + 2:k + 1]))
                i = k + 1
                continue
        if c == "'":
            k = i + 1
            while k < n and text[k] != "'" and text[k] != "\n":
                k += 2 if text[k] == "\\" else 1
            out.append("' '" + blank(text[i + 3:k + 1]))
            i = k + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def is_accessor_list(body: str, prefix: str) -> bool:
    """`{ get; set; }`, `{ get; init; }`, `{ get; private set; }`: auto-property accessor lists."""
    return prefix.rstrip().endswith(("get", "set", "init")) is False and re.fullmatch(
        r"\s*(?:(?:public|private|protected|internal)\s+)*(?:get|set|init)\s*;(?:\s*(?:(?:public|private|protected|internal)\s+)*(?:get|set|init)\s*;)*\s*",
        body) is not None


def violations(path: str):
    with open(path, encoding="utf-8") as f:
        original = f.read().split("\n")
    with open(path, encoding="utf-8") as f:
        code_lines = blank_literals(f.read()).split("\n")
    for number, code in enumerate(code_lines, 1):
        for m in OPENER.finditer(code):
            prefix = code[: m.start()]
            text = m.group(0)
            body = m.group("body")
            if is_accessor_list(body, prefix):
                continue
            if text.startswith("=>") and body.strip() == "":
                # An empty lambda (`_ => { }`) is a no-op callback, not a scope.
                continue
            window = code[max(0, m.start() - 200): m.end()]
            if ";" not in body and (INITIALIZER.search(window) or (text.startswith(")") and body.strip())):
                # new Foo(...) { A = 1 } (possibly with the arguments on earlier lines): an initializer, not a block
                continue
            if NON_BLOCK.search(code[max(0, m.start() - 10): m.start() + len(text)]):
                continue
            yield number, original[number - 1].strip()
            break


def files(args):
    if args:
        for a in args:
            if os.path.isdir(a):
                for d, _, fs in os.walk(a):
                    if "/bin" in d or "/obj" in d:
                        continue
                    for f in fs:
                        if f.endswith(".cs"):
                            yield os.path.join(d, f)
            else:
                yield a
        return
    out = subprocess.run(["git", "ls-files", "--", *[f"{d}/*.cs" for d in DEFAULT_DIRS]], cwd=ROOT, capture_output=True, text=True, check=True)
    for f in out.stdout.split():
        yield os.path.join(ROOT, f)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    found = 0
    for path in files(args):
        if not os.path.exists(path):
            continue
        for number, text in violations(path):
            found += 1
            print(f"{os.path.relpath(path, ROOT)}:{number}: single-line scope: {text}")
    if found:
        print(f"{found} single-line scope(s); put the braces of a block on their own lines (docs/code-style.md).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
