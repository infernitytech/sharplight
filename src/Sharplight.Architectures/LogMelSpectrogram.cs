using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Log-mel spectrogram of speech encoders as graph ops: a short-time Fourier transform as a strided convolution with a Hann-windowed DFT
/// basis, the power spectrum, a 1x1 convolution with the mel filter bank, log10 with a floor, clamping to the window's peak minus a dynamic
/// range, and an affine rescale (Whisper: <c>(max(log10(max(mel, 1e-10)), peak − 8) + 4) / 4</c>, as whisper.cpp and transformers compute it).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>{arch}.audio.*</c>): <c>sample_rate</c>, <c>fft_size</c> (analysis frame and DFT length), <c>hop_length</c>,
/// <c>frames</c> (frames per encoder pass), optional <c>log_floor</c> (default 1e-10), <c>dynamic_range</c> (log10 units, default 8),
/// <c>log_offset</c> (default 4) and <c>log_scale</c> (default 0.25). Weight <c>audio.mel_filters.weight</c> [mel bins, fft_size / 2 + 1] f32.
/// The input holds <c>frames · hop_length + fft_size − hop_length</c> samples; frame f covers samples <c>[f · hop, f · hop + fft_size)</c>, so
/// the caller supplies <c>fft_size / 2</c> samples of context before the window (centered framing).
/// </remarks>
internal sealed class LogMelSpectrogram
{
    private const string FilterWeight = "audio.mel_filters.weight";
    private const string BasisWeight = "audio.dft_basis.weight";
    private const string FilterKernelWeight = "audio.mel_filters.kernel";

    private readonly float _logFloor;
    private readonly float _dynamicRange;
    private readonly float _logOffset;
    private readonly float _logScale;

    private LogMelSpectrogram(ModelSource src)
    {
        SampleRate = src.Arch<int>("audio.sample_rate");
        FftSize = src.Arch<int>("audio.fft_size");
        HopLength = src.Arch<int>("audio.hop_length");
        Frames = src.Arch<int>("audio.frames");
        _logFloor = src.Arch("audio.log_floor", 1e-10f);
        _dynamicRange = src.Arch("audio.dynamic_range", 8f);
        _logOffset = src.Arch("audio.log_offset", 4f);
        _logScale = src.Arch("audio.log_scale", 0.25f);
        var filters = src.Weight(FilterWeight) ?? throw new NotSupportedException($"{src.Architecture}: audio encoders need '{FilterWeight}'.");
        MelBins = (int)filters.Shape[0].Fixed;
        if (filters.Shape.Rank != 2 || filters.Shape[1].Fixed != Bins)
        {
            throw new NotSupportedException($"{src.Architecture}: '{FilterWeight}' must be [mel bins, {Bins}], got {filters.Shape}.");
        }

        Weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal)
        {
            [BasisWeight] = WeightInfo.Derived(BasisWeight, DataType.F32, new Shape(2L * Bins, 1, 1, FftSize), DftBasis),
            [FilterKernelWeight] = filters.As(FilterKernelWeight, new Shape(MelBins, Bins, 1, 1)),
        };
    }

    /// <summary>Sample rate of the input in Hz.</summary>
    public int SampleRate { get; }
    /// <summary>Analysis frame and DFT length in samples.</summary>
    public int FftSize { get; }
    /// <summary>Samples between frames.</summary>
    public int HopLength { get; }
    /// <summary>Frames per pass.</summary>
    public int Frames { get; }
    /// <summary>Mel bins per frame.</summary>
    public int MelBins { get; }
    /// <summary>One-sided spectrum bins (fft_size / 2 + 1).</summary>
    public int Bins => (FftSize / 2) + 1;
    /// <summary>Samples of one input window (the frames plus the frame overhang).</summary>
    public int InputSamples => (Frames * HopLength) + FftSize - HopLength;
    /// <summary>Samples of context the input holds before the first frame's center.</summary>
    public int ContextSamples => FftSize / 2;
    /// <summary>The weights the graph adds to the model's own (DFT basis, filter kernel view).</summary>
    public IReadOnlyDictionary<string, WeightInfo> Weights { get; }

    /// <summary>The front end of an audio model, or null when the source has no audio hyper-parameters.</summary>
    public static LogMelSpectrogram? TryCreate(ModelSource src) => src.Has($"{src.Architecture}.audio.fft_size") ? new LogMelSpectrogram(src) : null;

    /// <summary>Emits the spectrogram of <paramref name="samples"/> [<see cref="InputSamples"/>]: f32 [1, mel bins, 1, frames].</summary>
    public Value Build(GraphBuilder b, Value samples)
    {
        using var _ = b.Group("audio");
        var signal = b.Reshape(samples, 1, 1, 1, InputSamples);
        var spectrum = b.Emit1("Conv", [signal, b.Weight(BasisWeight)], Attributes.From(("strides", new long[] { 1, HopLength })), "stft");
        var parts = b.Reshape(spectrum, 2, Bins, Frames);
        var power = b.Emit1("Reduce", [b.Unary("Square", parts)], Attributes.From(("mode", "sum"), ("axes", new long[] { 0 }), ("keepdims", false)), "power");
        var mel = b.Emit1("Conv", [b.Reshape(power, 1, Bins, 1, Frames), b.Weight(FilterKernelWeight)], null, "mel");
        var floored = b.Binary("Max", mel, Scalar(b, "log_floor", _logFloor));
        var log10 = b.Scale(b.Unary("Log", floored), 1f / MathF.Log(10f), 0f, "log_mel");
        var peak = b.Emit1("Reduce", [log10], Attributes.From(("mode", "max"), ("axes", new long[] { 0, 1, 2, 3 }), ("keepdims", true)), "log_mel_peak");
        var clamped = b.Binary("Max", log10, b.Scale(peak, 1f, -_dynamicRange));
        return b.Scale(clamped, _logScale, _logOffset * _logScale, "features");
    }

    private static Value Scalar(GraphBuilder b, string name, float value) => b.Constant(name, DataType.F32, new Shape(1), BitConverter.GetBytes(value));

    /// <summary>Rows [0, Bins): w·cos, rows [Bins, 2·Bins): −w·sin of the DFT, w the periodic Hann window; [2·Bins, 1, 1, fft_size] f32.</summary>
    private byte[] DftBasis()
    {
        int n = FftSize;
        var basis = new float[2 * Bins * n];
        for (int i = 0; i < n; i++)
        {
            double window = 0.5 - (0.5 * Math.Cos(2 * Math.PI * i / n));
            for (int k = 0; k < Bins; k++)
            {
                // Reduce k·i modulo n first so the angle stays exact for large products.
                double angle = 2 * Math.PI * ((long)k * i % n) / n;
                basis[(k * n) + i] = (float)(window * Math.Cos(angle));
                basis[((Bins + k) * n) + i] = (float)(-window * Math.Sin(angle));
            }
        }

        var bytes = new byte[basis.Length * sizeof(float)];
        Buffer.BlockCopy(basis, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
