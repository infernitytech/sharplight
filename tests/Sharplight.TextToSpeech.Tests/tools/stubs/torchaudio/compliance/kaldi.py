"""Stand-in for torchaudio.compliance.kaldi. fbank delegates to torchaudio's own pure-torch kaldi.py when the environment variable
SL_TORCHAUDIO_KALDI names that file (taken from a torchaudio wheel, whose compiled part may not match the installed torch)."""
import importlib.util
import os


def _reference():
    path = os.environ.get("SL_TORCHAUDIO_KALDI")
    if not path:
        raise NotImplementedError("torchaudio stub: set SL_TORCHAUDIO_KALDI to torchaudio's compliance/kaldi.py for kaldi.fbank")
    spec = importlib.util.spec_from_file_location("_torchaudio_kaldi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fbank(*args, **kwargs):
    return _reference().fbank(*args, **kwargs)
