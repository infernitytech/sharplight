using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of the decoder of a Qwen3-TTS 12 Hz codec (canonical architecture <see cref="Architecture"/>; the encoder is a Mimi, <see cref="MimiParameters"/>).</summary>
/// <param name="Hidden">Width of the pre-transformer.</param>
/// <param name="Latent">Width of the latent (the transformer's input and output, the upsampling stages).</param>
/// <param name="CodebookDim">Width of a projected codebook vector (twice the width of a codebook entry).</param>
/// <param name="CodebookSize">Entries per codebook.</param>
/// <param name="Dim">Channels after the first vocoder convolution (halved by every vocoder stage).</param>
/// <param name="Layers">Layers of the pre-transformer.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="HeadDim">Width of a head.</param>
/// <param name="Eps">The RMS norm epsilon.</param>
/// <param name="Window">The sliding attention window.</param>
/// <param name="RopeBase">The rotary base.</param>
/// <param name="Quantizers">The number of codebooks.</param>
/// <param name="SemanticQuantizers">The number of codebooks of the first (semantic) quantizer.</param>
/// <param name="UpsampleRates">The upsampling rate of each vocoder stage.</param>
/// <param name="UpsamplingRatios">The upsampling factor of each ConvNeXt stage before the vocoder.</param>
internal sealed record Qwen3TtsCodecParameters(int Hidden, int Latent, int CodebookDim, int CodebookSize, int Dim, int Layers, int Heads, int HeadDim, float Eps, int Window, float RopeBase,
    int Quantizers, int SemanticQuantizers, long[] UpsampleRates, long[] UpsamplingRatios)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "qwen3_tts_codec";

    /// <summary>The audio samples one frame of codes covers.</summary>
    public long SamplesPerFrame => UpsampleRates.Aggregate(1L, (a, b) => a * b) * UpsamplingRatios.Aggregate(1L, (a, b) => a * b);

    /// <summary>Reads the decoder parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static Qwen3TtsCodecParameters From(ModelSource source) => new(
        source.Arch<int>("decoder.hidden_size"), source.Arch<int>("decoder.latent_dim"), source.Arch<int>("decoder.codebook_dim"), source.Arch<int>("decoder.codebook_size"),
        source.Arch<int>("decoder.dim"), source.Arch<int>("decoder.layers"), source.Arch<int>("decoder.heads"), source.Arch<int>("decoder.head_dim"), source.Arch<float>("decoder.norm_eps"),
        source.Arch<int>("decoder.sliding_window"), source.Arch<float>("decoder.rope_base"), source.Arch<int>("decoder.quantizers"), source.Arch<int>("decoder.semantic_quantizers"),
        source.Arch<long[]>("decoder.upsample_rates"), source.Arch<long[]>("decoder.upsampling_ratios"));
}
