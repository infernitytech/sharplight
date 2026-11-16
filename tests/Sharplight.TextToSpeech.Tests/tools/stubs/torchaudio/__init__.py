"""Minimal stand-in for torchaudio (no build for this torch): only what the TTS references import."""
from . import transforms, compliance, functional  # noqa: F401
__version__ = "0.0-stub"


def load(path, *args, **kwargs):
    import soundfile
    import torch
    data, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), rate
