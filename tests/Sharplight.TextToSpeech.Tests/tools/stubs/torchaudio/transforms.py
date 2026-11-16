import math

import torch


class Resample(torch.nn.Module):
    def __init__(self, orig_freq, new_freq, **kwargs):
        super().__init__()
        self.orig, self.new = int(orig_freq), int(new_freq)

    def forward(self, x):
        import librosa
        y = librosa.resample(x.detach().cpu().numpy(), orig_sr=self.orig, target_sr=self.new)
        return torch.from_numpy(y).to(x.device, x.dtype)


def _hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate):
    """torchaudio.functional.melscale_fbanks with mel_scale='htk', norm=None: [n_freqs, n_mels]."""
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_pts = torch.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.max(torch.zeros(1), torch.min(down, up))


class _Spectrogram(torch.nn.Module):
    def __init__(self, n_fft, win, hop, power, center, pad_mode):
        super().__init__()
        self.n_fft, self.win, self.hop, self.power, self.center, self.pad_mode = n_fft, win, hop, power, center, pad_mode
        self.register_buffer("window", torch.hann_window(win))

    def forward(self, x):
        spec = torch.stft(x, self.n_fft, self.hop, self.win, self.window, center=self.center, pad_mode=self.pad_mode, return_complex=True).abs()
        return spec if self.power == 1.0 else spec.pow(self.power)


class _MelScale(torch.nn.Module):
    def __init__(self, n_mels, sample_rate, f_min, f_max, n_stft):
        super().__init__()
        self.register_buffer("fb", melscale_fbanks(n_stft, f_min, f_max, n_mels, sample_rate))

    def forward(self, spec):
        return torch.matmul(spec.transpose(-1, -2), self.fb).transpose(-1, -2)


class MelSpectrogram(torch.nn.Module):
    """torchaudio.transforms.MelSpectrogram (Hann window, reflect-centered STFT, HTK mel scale without norm), same submodule and buffer names."""

    def __init__(self, sample_rate=16000, n_fft=400, win_length=None, hop_length=None, f_min=0.0, f_max=None, n_mels=128, power=2.0,
                 center=True, normalized=False, norm=None, mel_scale="htk", pad_mode="reflect", **kwargs):
        super().__init__()
        assert norm is None and mel_scale == "htk" and not normalized
        win = win_length or n_fft
        self.spectrogram = _Spectrogram(n_fft, win, hop_length or win // 2, power, center, pad_mode)
        self.mel_scale = _MelScale(n_mels, sample_rate, f_min, f_max or sample_rate / 2, n_fft // 2 + 1)

    def forward(self, x):
        return self.mel_scale(self.spectrogram(x))


class Spectrogram(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("torchaudio stub: Spectrogram")
