using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using Sharplight.Models;

namespace Sharplight.Audio;

/// <summary>
/// Log-mel spectrogram front end (feature-extractor <c>type</c> <c>log-mel</c>): a short-time Fourier transform with a periodic Hann
/// window, the power spectrum, a mel filter bank, the logarithm with a floor and an optional dynamic-range clamp and affine rescale.
/// With Whisper's settings it computes <c>(max(log10(max(mel, 1e-10)), peak − 8) + 4) / 4</c> over the recording plus 30 s of silence,
/// as the reference implementations do.
/// </summary>
/// <remarks>
/// Component configuration (all lengths in samples): <c>sampleRate</c>, <c>fftSize</c>, <c>hopLength</c>, <c>melBins</c>, optional
/// <c>windowSize</c> (default fftSize; centered in the FFT frame), <c>center</c> (default true: frames are centered on multiples of the hop,
/// the signal is padded by fftSize / 2 on both sides), <c>padMode</c> (<c>reflect</c> (default) or <c>constant</c>),
/// <c>trailingPadding</c> (zeros appended to the recording, default 0), <c>dropLastFrame</c> (default false), <c>logBase</c> (<c>10</c>
/// (default) or <c>e</c>), <c>logFloor</c> (default 1e-10), <c>dynamicRange</c> (clamp to the recording's peak minus this many log units; 0 =
/// none, the default), <c>logOffset</c> and <c>logScale</c> (result = (log + offset) · scale; defaults 0 and 1). The filter bank is the
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
        _dynamicRange = component.ConfigValue("dynamicRange", 0.0);
        _logOffset = component.ConfigValue("logOffset", 0.0);
        _logScale = component.ConfigValue("logScale", 1.0);
        _window = new double[_fftSize];
        int start = (_fftSize - windowSize) / 2;
        for (int i = 0; i < windowSize; i++)
        {
            _window[start + i] = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / windowSize));
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
        return new AudioFeatures(Normalize(mel), Channels, frames);
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
            double value = Math.Max(mel[i], _logFloor);
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
