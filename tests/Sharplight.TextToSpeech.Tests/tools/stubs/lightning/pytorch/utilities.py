"""Stand-in for lightning.pytorch.utilities: Matcha-TTS's logger helper imports rank_zero_only (training only)."""


def rank_zero_only(fn):
    return fn
