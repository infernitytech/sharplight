using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a Dia model (canonical architecture <see cref="Architecture"/>, see <see cref="DiaAdapter"/>).</summary>
/// <param name="EncoderLayers">Layers of the text encoder.</param>
/// <param name="EncoderHidden">Width of the text encoder.</param>
/// <param name="EncoderHeads">Attention heads of the text encoder.</param>
/// <param name="EncoderHeadDim">Width per head of the text encoder.</param>
/// <param name="EncoderIntermediate">Hidden width of the text encoder's gated MLP.</param>
/// <param name="TextVocabulary">Text tokens (bytes).</param>
/// <param name="MaxText">The most text tokens.</param>
/// <param name="Layers">Decoder layers.</param>
/// <param name="Hidden">Decoder width.</param>
/// <param name="Heads">Query heads of the decoder's self-attention.</param>
/// <param name="KvHeads">Key and value heads of the decoder's self-attention.</param>
/// <param name="HeadDim">Width per head of the decoder's self-attention.</param>
/// <param name="CrossHeads">Heads of the decoder's cross-attention.</param>
/// <param name="CrossHeadDim">Width per head of the decoder's cross-attention.</param>
/// <param name="Intermediate">Hidden width of the decoder's gated MLP.</param>
/// <param name="Channels">Codebooks per frame.</param>
/// <param name="Vocabulary">Entries per codebook head (codes plus the end, padding and start tokens).</param>
/// <param name="MaxAudio">The most decoder positions.</param>
/// <param name="Eps">RMS norm epsilon.</param>
/// <param name="RopeBase">Rotary frequency base.</param>
/// <param name="DelayPattern">The delay in frames of every codebook.</param>
/// <param name="BosToken">The start token of the codebooks.</param>
/// <param name="EosToken">The end token of the first codebook.</param>
/// <param name="PadToken">The padding token of the codebooks.</param>
internal sealed record DiaParameters(int EncoderLayers, int EncoderHidden, int EncoderHeads, int EncoderHeadDim, int EncoderIntermediate, int TextVocabulary, int MaxText,
    int Layers, int Hidden, int Heads, int KvHeads, int HeadDim, int CrossHeads, int CrossHeadDim, int Intermediate, int Channels, int Vocabulary, int MaxAudio, float Eps,
    float RopeBase, int[] DelayPattern, int BosToken, int EosToken, int PadToken)
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "dia";

    /// <summary>The largest delay of <see cref="DelayPattern"/>.</summary>
    public int MaxDelay => DelayPattern.Max();

    /// <summary>Reads the parameters of a canonical source.</summary>
    /// <param name="source">A source of architecture <see cref="Architecture"/>.</param>
    public static DiaParameters From(ModelSource source) => new(
        source.Arch<int>("encoder.layer_count"), source.Arch<int>("encoder.hidden_size"), source.Arch<int>("encoder.attention.head_count"), source.Arch<int>("encoder.head_dim"),
        source.Arch<int>("encoder.intermediate_size"), source.Arch<int>("encoder.vocabulary"), source.Arch<int>("encoder.context_length"),
        source.Arch<int>("layer_count"), source.Arch<int>("hidden_size"), source.Arch<int>("attention.head_count"), source.Arch<int>("attention.head_count_kv"),
        source.Arch<int>("head_dim"), source.Arch<int>("cross_attention.head_count"), source.Arch<int>("cross_attention.head_dim"), source.Arch<int>("intermediate_size"),
        source.Arch<int>("channels"), source.Arch<int>("vocabulary"), source.Arch<int>("context_length"), source.Arch("norm_eps", 1e-5f), source.Arch("rope.base", 10000f),
        [.. source.Arch<long[]>("delay_pattern").Select(d => (int)d)], source.Arch<int>("bos_token"), source.Arch<int>("eos_token"), source.Arch<int>("pad_token"));
}
