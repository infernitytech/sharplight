import math


def _hz_to_mel(freq, mel_scale="htk"):
    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))
    f_sp = 200.0 / 3
    mels = freq / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels, mel_scale="htk"):
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    raise NotImplementedError("torchaudio stub: slaney _mel_to_hz")
