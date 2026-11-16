using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a SNAC decoder (canonical architecture <see cref="Architecture"/>, see <c>Sharplight.Import.Snac</c>).</summary>
/// <param name="SampleRate">Sample rate of the decoded audio.</param>
/// <param name="DecoderDim">Channel count after the decoder's input projection.</param>
/// <param name="DecoderRates">Upsampling factor of each decoder block.</param>
/// <param name="VqStrides">Time stride of each codebook (codebook <c>i</c> has one code per <c>stride</c> latent frames).</param>
/// <param name="CodebookSize">Entries per codebook.</param>
/// <param name="CodebookDim">Width of a codebook entry.</param>
/// <param name="LatentDim">Width of the latent the codebooks add up to.</param>
/// <param name="Noise">Whether the decoder blocks add input-dependent noise.</param>
/// <param name="Depthwise">Whether the decoder uses depthwise convolutions.</param>
internal sealed record SnacParameters(int SampleRate, int DecoderDim, long[] DecoderRates, long[] VqStrides, int CodebookSize, int CodebookDim, int LatentDim,
    bool Noise, bool Depthwise)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "snac";

    /// <summary>The number of audio samples one code of the coarsest codebook covers (<c>Π decoder rates · stride₀</c>).</summary>
    public long SamplesPerCoarseCode => DecoderRates.Aggregate(1L, (a, b) => a * b) * VqStrides[0];

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static SnacParameters From(ModelSource source) => new(
        source.Arch<int>("sample_rate"), source.Arch<int>("decoder_dim"), source.Arch<long[]>("decoder_rates"), source.Arch<long[]>("vq_strides"),
        source.Arch<int>("codebook_size"), source.Arch<int>("codebook_dim"), source.Arch<int>("latent_dim"), source.Arch<bool>("noise"), source.Arch<bool>("depthwise"));
}
