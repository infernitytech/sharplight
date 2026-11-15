"""Loads AutoAWQ's packing modules (awq/modules/linear/gemv_fast.py, marlin.py) without importing the awq package, whose __init__ pulls in
model classes that need a transformers version older than the one used here. The modules only use try_import from awq.utils.module."""
import importlib.util
import os
import sys
import types


def load(name):
    if "awq.utils.module" not in sys.modules:
        for package in ("awq", "awq.utils", "awq.utils.module"):
            sys.modules[package] = types.ModuleType(package)
        sys.modules["awq.utils.module"].try_import = lambda module: (None, f"{module} is not installed")
    spec = importlib.util.find_spec("autoawq") if False else None
    root = None
    for entry in sys.path:
        candidate = os.path.join(entry, "awq", "modules", "linear", name + ".py")
        if os.path.exists(candidate):
            root = candidate
            break
    spec = importlib.util.spec_from_file_location("awq_" + name, root)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
