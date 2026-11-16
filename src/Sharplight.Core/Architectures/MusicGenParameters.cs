using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a MusicGen model (canonical architecture <see cref="Architecture"/>, see <see cref="MusicGenAdapter"/>).</summary>
/// <param name="TextLayers">Layers of the T5 text encoder.</param>
/// <param name="TextHidden">Width of the text encoder.</param>
/// <param name="TextHeads">Attention heads of the text encoder.</param>
/// <param name="TextHeadDim">Width per head of the text encoder.</param>
/// <param name="TextEps">RMS norm epsilon of the text encoder.</param>
/// <param name="TextBuckets">Relative position buckets of the text encoder.</param>
/// <param name="TextMaxDistance">Distance from which the last relative position bucket applies.</param>
/// <param name="TextMaxTokens">The most tokens of a prompt.</param>
/// <param name="Layers">Decoder layers.</param>
/// <param name="Hidden">Decoder width.</param>
/// <param name="Heads">Decoder attention heads.</param>
/// <param name="Codebooks">Codebooks (streams of codes the decoder predicts per step).</param>
/// <param name="Vocabulary">Codes per codebook (the start and padding code is this value).</param>
/// <param name="MaxSteps">The most decoder steps (positions of the sinusoidal table).</param>
/// <param name="Eps">LayerNorm epsilon of the decoder.</param>
/// <param name="Guidance">The default classifier-free guidance scale.</param>
/// <param name="FrameRate">Code frames per second.</param>
internal sealed record MusicGenParameters(int TextLayers, int TextHidden, int TextHeads, int TextHeadDim, float TextEps, int TextBuckets, int TextMaxDistance, int TextMaxTokens,
    int Layers, int Hidden, int Heads, int Codebooks, int Vocabulary, int MaxSteps, float Eps, float Guidance, double FrameRate)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "musicgen";

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static MusicGenParameters From(ModelSource source) => new(
        source.Arch<int>("text.layer_count"), source.Arch<int>("text.hidden_size"), source.Arch<int>("text.attention.head_count"), source.Arch<int>("text.head_dim"),
        source.Arch("text.norm_eps", 1e-6f), source.Arch<int>("text.relative_buckets"), source.Arch<int>("text.relative_max_distance"), source.Arch<int>("text.context_length"),
        source.Arch<int>("layer_count"), source.Arch<int>("hidden_size"), source.Arch<int>("attention.head_count"), source.Arch<int>("codebook_count"),
        source.Arch<int>("vocabulary"), source.Arch<int>("max_steps"), source.Arch("norm_eps", 1e-5f), source.Arch("guidance_scale", 3f), source.Arch<double>("frame_rate"));

    /// <summary>The codec's parameters (canonical keys under <c>codec.</c>).</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static EncodecParameters Codec(ModelSource source) => new(
        source.Arch<int>("codec.sample_rate"), source.Arch<int>("codec.hidden_size"), source.Arch<int>("codec.num_filters"), source.Arch<int>("codec.kernel_size"),
        source.Arch<int>("codec.last_kernel_size"), source.Arch<int>("codec.residual_kernel_size"), source.Arch<int>("codec.dilation_growth_rate"), source.Arch<int>("codec.residual_layers"),
        source.Arch<int>("codec.compress"), source.Arch<int>("codec.lstm_layers"), source.Arch<long[]>("codec.upsampling_ratios"), source.Arch<int>("codec.codebook_size"),
        source.Arch<int>("codec.codebook_dim"), source.Arch<int>("codec.codebook_count"), source.Arch<int>("codec.codebook_count"), source.Arch("codec.conv_shortcut", true),
        source.Arch("codec.causal", true));
}
