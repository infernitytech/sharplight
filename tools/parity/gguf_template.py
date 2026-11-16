"""Writes the chat template embedded in a GGUF file to a text file (nothing when it has none). Usage: python -I gguf_template.py <gguf> <out>"""
import sys

import gguf

reader = gguf.GGUFReader(sys.argv[1])
field = reader.fields.get("tokenizer.chat_template")
if field is not None:
    open(sys.argv[2], "w", encoding="utf-8").write(bytes(field.parts[field.data[0]]).decode("utf-8"))
