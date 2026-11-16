using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a DAC model (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.HuggingFace</c>).</summary>
/// <param name="SampleRate">Sample rate of the audio.</param>
/// <param name="Hidden">Width of the latent the quantizer sees.</param>
/// <param name="EncoderHidden">Channels after the encoder's first convolution (doubled per stage).</param>
/// <param name="DecoderHidden">Channels after the decoder's first convolution (halved per stage).</param>
/// <param name="DownsamplingRatios">Stride of each encoder stage.</param>
/// <param name="UpsamplingRatios">Stride of each decoder stage.</param>
/// <param name="CodebookSize">Entries per codebook.</param>
/// <param name="CodebookDim">Width of a codebook entry (the factorized code space).</param>
/// <param name="Codebooks">The number of codebooks.</param>
internal sealed record DacParameters(int SampleRate, int Hidden, int EncoderHidden, int DecoderHidden, long[] DownsamplingRatios, long[] UpsamplingRatios, int CodebookSize,
    int CodebookDim, int Codebooks)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "dac";

    /// <summary>The number of audio samples one frame covers.</summary>
    public long SamplesPerFrame => DownsamplingRatios.Aggregate(1L, (a, b) => a * b);

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static DacParameters From(ModelSource source) => new(
        source.Arch<int>("sample_rate"), source.Arch<int>("hidden_size"), source.Arch<int>("encoder_hidden_size"), source.Arch<int>("decoder_hidden_size"),
        source.Arch<long[]>("downsampling_ratios"), source.Arch<long[]>("upsampling_ratios"), source.Arch<int>("codebook_size"), source.Arch<int>("codebook_dim"),
        source.Arch<int>("codebook_count"));
}
