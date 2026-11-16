using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Kokoro model (canonical architecture <see cref="Architecture"/>, <c>kokoro.*</c> keys).</summary>
/// <param name="VocabSize">Number of phoneme symbols.</param>
/// <param name="Hidden">Width of the text and duration encoders.</param>
/// <param name="StyleDim">Width of one style vector (a voice row holds two: the decoder's and the predictor's).</param>
/// <param name="Layers">Depth of the text encoder's convolution stack and of the duration encoder.</param>
/// <param name="MaxDuration">Number of duration logits per symbol.</param>
/// <param name="TextKernel">Kernel size of the text encoder convolutions.</param>
/// <param name="BertHidden">Width of the ALBERT text encoder.</param>
/// <param name="BertHeads">Attention heads of the ALBERT encoder.</param>
/// <param name="BertFeedForward">Feed-forward width of the ALBERT encoder.</param>
/// <param name="BertLayers">Times the shared ALBERT layer is applied.</param>
/// <param name="BertContext">Longest symbol sequence.</param>
/// <param name="InitialChannels">Channel count at the start of the generator.</param>
/// <param name="UpsampleRates">Transposed-convolution strides of the generator.</param>
/// <param name="UpsampleKernels">Transposed-convolution kernel sizes of the generator.</param>
/// <param name="ResBlockKernels">Kernel sizes of the generator's residual blocks (one per parallel branch).</param>
/// <param name="ResBlockDilations">Dilations of the three convolutions of each residual block, per branch.</param>
/// <param name="NFft">FFT length of the final inverse STFT.</param>
/// <param name="Hop">Hop of the final inverse STFT.</param>
/// <param name="SampleRate">Sample rate of the output.</param>
/// <param name="Harmonics">Number of sine harmonics of the excitation source.</param>
/// <param name="SineAmplitude">Amplitude of the voiced sines.</param>
/// <param name="NoiseStd">Standard deviation of the noise added to voiced sources.</param>
/// <param name="VoicedThreshold">F0 (Hz) above which a frame counts as voiced.</param>
internal sealed record KokoroParameters(int VocabSize, int Hidden, int StyleDim, int Layers, int MaxDuration, int TextKernel, int BertHidden, int BertHeads,
    int BertFeedForward, int BertLayers, int BertContext, int InitialChannels, long[] UpsampleRates, long[] UpsampleKernels, long[] ResBlockKernels,
    long[][] ResBlockDilations, int NFft, int Hop, int SampleRate, int Harmonics, float SineAmplitude, float NoiseStd, float VoicedThreshold)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "kokoro";

    /// <summary>The number of audio samples one decoder frame covers (<c>Π upsample rates · hop · 2</c>: the F0 curve has twice the frame rate).</summary>
    public long SamplesPerFrame => UpsampleRates.Aggregate(1L, (a, b) => a * b) * Hop * 2;

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static KokoroParameters From(ModelSource source) => new(
        source.Arch<int>("vocab_size"), source.Arch<int>("hidden_size"), source.Arch<int>("style_dim"), source.Arch<int>("layer_count"),
        source.Arch<int>("max_duration"), source.Arch<int>("text_encoder.kernel_size"), source.Arch<int>("bert.hidden_size"), source.Arch<int>("bert.head_count"),
        source.Arch<int>("bert.intermediate_size"), source.Arch<int>("bert.layer_count"), source.Arch<int>("bert.context_length"),
        source.Arch<int>("generator.initial_channels"), source.Arch<long[]>("generator.upsample_rates"), source.Arch<long[]>("generator.upsample_kernel_sizes"),
        source.Arch<long[]>("generator.resblock_kernel_sizes"),
        [.. source.Arch<string>("generator.resblock_dilations").Split(';').Select(r => r.Split(',').Select(long.Parse).ToArray())],
        source.Arch<int>("generator.n_fft"), source.Arch<int>("generator.hop"), source.Arch<int>("audio.sample_rate"), source.Arch<int>("source.harmonics"),
        source.Arch<float>("source.sine_amplitude"), source.Arch<float>("source.noise_std"), source.Arch<float>("source.voiced_threshold"));
}
