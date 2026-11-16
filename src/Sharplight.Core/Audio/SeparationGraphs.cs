namespace Sharplight.Audio;

/// <summary>
/// The graph contract of the <see cref="Models.ModelKinds.AudioSeparation"/> kind (Demucs): one component (<see cref="Models.ModelComponentRoles.Encoder"/>) with the graph
/// <see cref="SeparateGraph"/> over one fixed-length segment. The host computes the short-time Fourier transform and the normalization (<see cref="SpectrumInput"/>: the
/// complex spectrogram as real and imaginary channels per audio channel, <see cref="WaveformInput"/>: the segment), the graph returns the estimates of both domains
/// (<see cref="SpectrumOutput"/>, <see cref="WaveformOutput"/>), and the host turns the spectrum estimate back into audio and adds the two.
/// </summary>
public static class SeparationGraphs
{
    /// <summary>The graph: <see cref="SpectrumInput"/> f32 <c>[1, 2 · channels, bins, frames]</c> and <see cref="WaveformInput"/> f32 <c>[1, channels, samples]</c> in, <see cref="SpectrumOutput"/> f32 <c>[1, stems · 2 · channels, bins, frames]</c> and <see cref="WaveformOutput"/> f32 <c>[stems · channels, samples]</c> out.</summary>
    public const string SeparateGraph = "separate";

    /// <summary>The normalized spectrogram input.</summary>
    public const string SpectrumInput = "spectrum";

    /// <summary>The normalized waveform input.</summary>
    public const string WaveformInput = "waveform";

    /// <summary>The spectrogram estimate of every stem (normalization not yet undone).</summary>
    public const string SpectrumOutput = "spectrum_out";

    /// <summary>The waveform estimate of every stem (normalization not yet undone).</summary>
    public const string WaveformOutput = "waveform_out";

    /// <summary>The configuration key of the sample rate.</summary>
    public const string SampleRateKey = "sampleRate";

    /// <summary>The configuration key of the audio channels.</summary>
    public const string ChannelsKey = "channels";

    /// <summary>The configuration key of the stem names (comma separated).</summary>
    public const string StemsKey = "stems";

    /// <summary>The configuration key of the samples of one segment.</summary>
    public const string SegmentSamplesKey = "segmentSamples";

    /// <summary>The configuration key of the FFT size of the spectrogram input.</summary>
    public const string FftSizeKey = "fftSize";

    /// <summary>The configuration key of the hop of the spectrogram input (the number of frames is <c>ceil(samples / hop)</c>).</summary>
    public const string HopKey = "hop";
}
