# Generates chat_cases.json and chat_goldens.json by rendering every template in ../Templates with real Python jinja2
# using the Hugging Face transformers environment settings (ImmutableSandboxedEnvironment, trim/lstrip, tojson override).
# Usage: venv/bin/python gen_chat_goldens.py <Templates dir> <out dir>
import json, sys, os, datetime
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

FIXED_NOW = datetime.datetime(2026, 1, 15, 10, 30, 45)
BOS, EOS = "<BOS>", "<EOS>"

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

class Raised(Exception):
    pass

def raise_exception(message):
    raise Raised(message)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = lambda fmt: FIXED_NOW.strftime(fmt)

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                      "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calc", "description": "Evaluate an expression.",
     "parameters": {"type": "object", "properties": {"expr": {"type": "string"}, "precision": {"type": "integer"}}, "required": ["expr"]}}},
]
SYS = {"role": "system", "content": "You are helpful."}
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather",
        "arguments": {"city": "Zürich", "unit": "c", "n": 3, "x": 1.5, "flag": True, "items": [1, 2], "o": None}}}
CALL2 = {"id": "call_2", "type": "function", "function": {"name": "calc", "arguments": {"expr": "2+2", "precision": 2}}}

CASES = {
    "sys_user": {"messages": [SYS, {"role": "user", "content": "Hi"}]},
    "user_only": {"messages": [{"role": "user", "content": "Hello there"}]},
    "multi_turn": {"messages": [SYS, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
                                {"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}, {"role": "user", "content": "Thanks"}]},
    "unicode": {"messages": [{"role": "user", "content": "Grüße 😀 日本語 \"quoted\" \\ back"}]},
    "reasoning": {"messages": [SYS, {"role": "user", "content": "Solve 1+1"},
                               {"role": "assistant", "content": "2", "reasoning_content": "Let me think. 1+1=2."}, {"role": "user", "content": "And 2+2?"}]},
    "tools_user": {"messages": [SYS, {"role": "user", "content": "Weather in Paris?"}], "tools": TOOLS},
    "tool_roundtrip": {"messages": [{"role": "user", "content": "Weather in Zürich?"},
                                    {"role": "assistant", "content": "", "tool_calls": [CALL]},
                                    {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp\": 20}"}], "tools": TOOLS},
    "tool_parallel": {"messages": [SYS, {"role": "user", "content": "Weather in Zürich and 2+2?"},
                                   {"role": "assistant", "content": "Checking both.", "tool_calls": [CALL, CALL2]},
                                   {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp\": 20}"},
                                   {"role": "tool", "tool_call_id": "call_2", "name": "calc", "content": "4"},
                                   {"role": "assistant", "content": "It is 20 degrees and 4."}, {"role": "user", "content": "ok"}], "tools": TOOLS},
    "image": {"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]},
    "system_parts": {"messages": [{"role": "system", "content": [{"type": "text", "text": "Be brief."}]}, {"role": "user", "content": "Hi"}]},
}
COMMON = ["sys_user", "multi_turn", "tools_user", "tool_roundtrip", "image", "reasoning"]
FULL = list(CASES)  # for the primary templates

def key(case, gen, think):
    return f"{case}|gen={int(gen)}|think={think}"

def run(tpl, case, gen, think):
    c = CASES[case]
    kw = dict(messages=c["messages"], add_generation_prompt=gen, bos_token=BOS, eos_token=EOS, tools=c.get("tools"))
    if think != "n":
        kw["enable_thinking"] = think == "t"
    return tpl.render(**kw)

def main(tdir, out):
    json.dump(CASES, open(os.path.join(out, "chat_cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    goldens = {}
    stats = {}
    for fn in sorted(os.listdir(tdir)):
        if not fn.endswith(".jinja"): continue
        name = fn[:-6]
        src = open(os.path.join(tdir, fn), encoding="utf-8").read()
        primary = name.startswith("Qwen3.8") or name.startswith("gemma-4")
        # HF's `generation` tag is a tokenizer-side extension; for the Python reference it is a plain block.
        import re
        src = re.sub(r"\{%(-?)\s*generation\s*(-?)%\}", r"{%\1 if true \2%}", src)
        src = re.sub(r"\{%(-?)\s*endgeneration\s*(-?)%\}", r"{%\1 endif \2%}", src)
        try:
            tpl = env.from_string(src)
        except Exception as ex:
            print("COMPILE FAIL", name, type(ex).__name__, ex, file=sys.stderr)
            goldens[name] = {"__compile_error__": type(ex).__name__}
            continue
        g = {}
        cases = FULL if primary else COMMON
        for case in cases:
            for gen in (True, False):
                thinks = ["n", "t", "f"] if primary else ["n"]
                for think in thinks:
                    if not primary and not gen and case not in ("sys_user", "tool_roundtrip"):
                        continue
                    k = key(case, gen, think)
                    try:
                        g[k] = {"output": run(tpl, case, gen, think)}
                    except Raised as ex:
                        g[k] = {"error": str(ex)}
                    except Exception as ex:
                        stats.setdefault(name, []).append(f"{k}: {type(ex).__name__}: {str(ex)[:100]}")
        goldens[name] = g
    json.dump(goldens, open(os.path.join(out, "chat_goldens.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    for n, v in stats.items():
        print("PYERR", n, len(v), v[0], file=sys.stderr)
    print(sum(len(v) for v in goldens.values()), "goldens")

main(sys.argv[1], sys.argv[2])
