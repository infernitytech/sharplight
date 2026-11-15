"""Prints the transformers token ids of a text for a checkpoint directory (reference for the sweep's tokenizer check).

Usage: python -I hf_tokenize.py <checkpoint dir> <text>
Remote code is never run; a tokenizer that needs it prints an error line instead.
"""
import json
import sys

from transformers import AutoTokenizer

try:
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=False)
    print(json.dumps(tokenizer.encode(sys.argv[2], add_special_tokens=False)))
except Exception as error:  # noqa: BLE001 - any failure is reported to the sweep
    print("ERROR " + type(error).__name__ + ": " + str(error).replace("\n", " ")[:200])
