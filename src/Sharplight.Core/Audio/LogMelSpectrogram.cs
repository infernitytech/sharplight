using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using Sharplight.Models;

namespace Sharplight.Audio;

/// <summary>
/// Log-mel spectrogram front end (feature-extractor <c>type</c> <c>log-mel</c>): optional pre-emphasis, a short-time Fourier transform with
/// a Hann window, the power spectrum, a mel filter bank, the logarithm with a floor or an additive guard, an optional dynamic-range clamp
/// and affine rescale, and an optional per-channel normalization. With Whisper's settings it computes
/// <c>(max(log10(max(mel, 1e-10)), peak − 8) + 4) / 4</c> over the recording plus 30 s of silence; with NeMo's (Parakeet, Canary)
/// <c>ln(mel + 2⁻²⁴)</c> of the pre-emphasized signal, each channel normalized to zero mean and unit variance over the recording.
/// </summary>
/// <remarks>
/// Component configuration (all lengths in samples): <c>sampleRate</c>, <c>fftSize</c>, <c>hopLength</c>, <c>melBins</c>, optional
/// <c>windowSize</c> (default fftSize; centered in the FFT frame), <c>periodicWindow</c> (default true; false: the symmetric Hann window),
/// <c>preemphasis</c> (coefficient a of y[n] = x[n] − a·x[n − 1], default 0), <c>center</c> (default true: frames are centered on multiples of the hop,
/// the signal is padded by fftSize / 2 on both sides), <c>padMode</c> (<c>reflect</c> (default) or <c>constant</c>),
/// <c>trailingPadding</c> (zeros appended to the recording, default 0), <c>dropLastFrame</c> (default false), <c>logBase</c> (<c>10</c>
/// (default) or <c>e</c>), <c>logFloor</c> (default 1e-10), <c>logGuard</c> (when positive, log(mel + guard) instead of the floor; default 0),
/// <c>dynamicRange</c> (clamp to the recording's peak minus this many log units; 0 = none, the default), <c>logOffset</c> and <c>logScale</c>
/// (result = (log + offset) · scale; defaults 0 and 1), <c>normalize</c> (<c>none</c> (default) or <c>per-channel</c>: each channel minus its
/// mean over the frames, divided by its standard deviation (n − 1 denominator) plus <c>normalizeEpsilon</c>, default 1e-5). The filter bank is the
/// attachment <see cref="ModelComponentAttachments.MelFilters"/>: little-endian f32 [melBins, fftSize / 2 + 1].
/// </remarks>
internal sealed class LogMelSpectrogram : IAudioFeatureExtractor
{
    /// <summary>The feature-extractor type this class implements.</summary>
    public const string Type = "log-mel";

    private readonly int _fftSize;
    private readonly int _bins;
    private readonly bool _center;
    private readonly bool _reflect;
    private readonly int _trailingPadding;
    private readonly bool _dropLastFrame;
    private readonly bool _log10;
    private readonly double _logFloor;
    private readonly double _logGuard;
    private readonly double _preemphasis;
    private readonly bool _normalizeChannels;
    private readonly double _normalizeEpsilon;
    private readonly double _dynamicRange;
    private readonly double _logOffset;
    private readonly double _logScale;
    private readonly double[] _window;
    private readonly float[] _filters;
    private readonly FourierTransform _transform;

    /// <summary>Creates the extractor from a feature-extractor component's configuration and filter bank.</summary>
    /// <param name="component">The component (configuration keys in the remarks).</param>
    /// <param name="filters">The <c>mel-filters</c> attachment bytes.</param>
    /// <exception cref="InvalidDataException">The configuration or filter bank is inconsistent.</exception>
    public LogMelSpectrogram(ModelComponent component, ReadOnlySpan<byte> filters)
    {
        SampleRate = component.ConfigValue("sampleRate", 0);
        _fftSize = component.ConfigValue("fftSize", 0);
        HopLength = component.ConfigValue("hopLength", 0);
        Channels = component.ConfigValue("melBins", 0);
        int windowSize = component.ConfigValue("windowSize", _fftSize);
        if (SampleRate <= 0 || _fftSize <= 0 || HopLength <= 0 || Channels <= 0 || windowSize <= 0 || windowSize > _fftSize)
        {
            throw new InvalidDataException($"log-mel feature extractor: sampleRate {SampleRate}, fftSize {_fftSize}, hopLength {HopLength}, "
                + $"melBins {Channels} and windowSize {windowSize} must be positive (windowSize at most fftSize).");
        }

        _bins = (_fftSize / 2) + 1;
        if (filters.Length != Channels * _bins * sizeof(float))
        {
            throw new InvalidDataException($"log-mel feature extractor: the filter bank has {filters.Length} bytes, expected [{Channels}, {_bins}] f32.");
        }

        _filters = new float[Channels * _bins];
        for (int i = 0; i < _filters.Length; i++)
        {
            _filters[i] = BinaryPrimitives.ReadSingleLittleEndian(filters[(i * sizeof(float))..]);
        }

        _center = component.ConfigValue("center", true);
        _reflect = component.ConfigValue("padMode", "reflect") switch
        {
            "reflect" => true,
            "constant" => false,
            var other => throw new InvalidDataException($"log-mel feature extractor: unknown padMode '{other}'."),
        };
        _trailingPadding = component.ConfigValue("trailingPadding", 0);
        _dropLastFrame = component.ConfigValue("dropLastFrame", false);
        _log10 = component.ConfigValue("logBase", "10") switch
        {
            "10" => true,
            "e" => false,
            var other => throw new InvalidDataException($"log-mel feature extractor: unknown logBase '{other}'."),
        };
        _logFloor = component.ConfigValue("logFloor", 1e-10);
        _logGuard = component.ConfigValue("logGuard", 0.0);
        _preemphasis = component.ConfigValue("preemphasis", 0.0);
        _normalizeChannels = component.ConfigValue("normalize", "none") switch
        {
            "none" => false,
            "per-channel" => true,
            var other => throw new InvalidDataException($"log-mel feature extractor: unknown normalize '{other}'."),
        };
        _normalizeEpsilon = component.ConfigValue("normalizeEpsilon", 1e-5);
        _dynamicRange = component.ConfigValue("dynamicRange", 0.0);
        _logOffset = component.ConfigValue("logOffset", 0.0);
        _logScale = component.ConfigValue("logScale", 1.0);
        _window = new double[_fftSize];
        int start = (_fftSize - windowSize) / 2;
        int period = component.ConfigValue("periodicWindow", true) ? windowSize : windowSize - 1;
        for (int i = 0; i < windowSize; i++)
        {
            _window[start + i] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / period));
        }

        _transform = new FourierTransform(_fftSize);
    }

    /// <inheritdoc/>
    public int SampleRate { get; }

    /// <inheritdoc/>
    public int HopLength { get; }

    /// <inheritdoc/>
    public int Channels { get; }

    /// <inheritdoc/>
    public AudioFeatures Extract(ReadOnlySpan<float> samples)
    {
        double[] signal = PaddedSignal(samples);
        int frames = 1 + ((signal.Length - _fftSize) / HopLength);
        if (_dropLastFrame)
        {
            frames--;
        }

        frames = Math.Max(frames, 0);
        var mel = new double[Channels * frames];
        Parallel.For(0, frames, () => new FrameScratch(_fftSize), (frame, _, scratch) =>
        {
            ComputeFrame(signal, frame, frames, mel, scratch);
            return scratch;
        }, static _ => { });
        var features = Normalize(mel);
        if (_normalizeChannels)
        {
            NormalizeChannels(features, frames);
        }

        return new AudioFeatures(features, Channels, frames);
    }

    /// <summary>The recording with the trailing padding and, for centered frames, fftSize / 2 samples of padding on both sides.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private double[] PaddedSignal(ReadOnlySpan<float> samples)
    {
        int margin = _center ? _fftSize / 2 : 0;
        int content = samples.Length + _trailingPadding;
        var signal = new double[content + (2 * margin)];
        for (int i = 0; i < samples.Length; i++)
        {
            signal[margin + i] = samples[i];
        }

        if (_preemphasis != 0)
        {
            // Backwards, so every step still reads the original previous sample.
            for (int i = samples.Length - 1; i >= 1; i--)
            {
                signal[margin + i] -= _preemphasis * samples[i - 1];
            }
        }

        if (!_reflect || margin == 0)
        {
            return signal;
        }

        for (int i = 1; i <= margin; i++)
        {
            signal[margin - i] = i < content ? signal[margin + i] : 0;
            signal[margin + content - 1 + i] = content - 1 - i >= 0 ? signal[margin + content - 1 - i] : 0;
        }

        return signal;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ComputeFrame(double[] signal, int frame, int frames, double[] mel, FrameScratch scratch)
    {
        int offset = frame * HopLength;
        for (int i = 0; i < _fftSize; i++)
        {
            scratch.Frame[i] = signal[offset + i] * _window[i];
        }

        _transform.Forward(scratch.Frame, scratch.Real, scratch.Imaginary, scratch.Work);
        for (int k = 0; k < _bins; k++)
        {
            scratch.Power[k] = (scratch.Real[k] * scratch.Real[k]) + (scratch.Imaginary[k] * scratch.Imaginary[k]);
        }

        for (int m = 0; m < Channels; m++)
        {
            double sum = 0;
            var filter = _filters.AsSpan(m * _bins, _bins);
            for (int k = 0; k < _bins; k++)
            {
                sum += filter[k] * scratch.Power[k];
            }

            mel[(m * frames) + frame] = sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private float[] Normalize(double[] mel)
    {
        double peak = double.NegativeInfinity;
        for (int i = 0; i < mel.Length; i++)
        {
            double value = _logGuard > 0 ? mel[i] + _logGuard : Math.Max(mel[i], _logFloor);
            mel[i] = _log10 ? Math.Log10(value) : Math.Log(value);
            peak = Math.Max(peak, mel[i]);
        }

        double lowest = _dynamicRange > 0 ? peak - _dynamicRange : double.NegativeInfinity;
        var features = new float[mel.Length];
        for (int i = 0; i < mel.Length; i++)
        {
            features[i] = (float)((Math.Max(mel[i], lowest) + _logOffset) * _logScale);
        }

        return features;
    }

    // Each channel to zero mean and unit standard deviation over the frames (sample variance, the epsilon added to the deviation).
    private void NormalizeChannels(float[] features, int frames)
    {
        for (int channel = 0; channel < Channels; channel++)
        {
            var row = features.AsSpan(channel * frames, frames);
            double mean = 0;
            foreach (float v in row)
            {
                mean += v;
            }

            mean /= Math.Max(frames, 1);
            double sum = 0;
            foreach (float v in row)
            {
                sum += (v - mean) * (v - mean);
            }

            double deviation = Math.Sqrt(sum / Math.Max(frames - 1, 1));
            double scale = 1 / (deviation + _normalizeEpsilon);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = (float)((row[i] - mean) * scale);
            }
        }
    }

    /// <summary>Per-thread work buffers of one frame.</summary>
    private sealed class FrameScratch(int size)
    {
        public double[] Frame { get; } = new double[size];
        public double[] Real { get; } = new double[size];
        public double[] Imaginary { get; } = new double[size];
        public double[] Work { get; } = new double[2 * size];
        public double[] Power { get; } = new double[(size / 2) + 1];
    }
}
