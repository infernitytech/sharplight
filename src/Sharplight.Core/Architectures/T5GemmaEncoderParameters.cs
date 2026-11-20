using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a T5Gemma encoder (the Gemma 2 layers with bidirectional attention) kept under a key prefix of another model.</summary>
/// <param name="Hidden">Width.</param>
/// <param name="Layers">Layers.</param>
/// <param name="Intermediate">Width of the gated feed-forward layer.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="KvHeads">Key-value heads.</param>
/// <param name="HeadDim">Width of a head.</param>
/// <param name="NormEps">RMS norm epsilon.</param>
/// <param name="RopeBase">Rotary base.</param>
/// <param name="Softcap">Attention logit soft cap (0 for none).</param>
/// <param name="QueryScale">The attention scale is <c>QueryScale^-1/2</c>.</param>
/// <param name="SlidingWindow">The window of the sliding-window layers.</param>
/// <param name="SlidingLayers">1 for the layers with a sliding window, 0 for the others.</param>
internal sealed record T5GemmaEncoderParameters(int Hidden, int Layers, int Intermediate, int Heads, int KvHeads, int HeadDim, float NormEps, float RopeBase, float Softcap,
    float QueryScale, int SlidingWindow, long[] SlidingLayers)
{
    /// <summary>Reads the parameters stored under <c>{architecture}.{prefix}</c>.</summary>
    /// <param name="source">The model's source.</param>
    /// <param name="prefix">The key prefix after the architecture (e.g. <c>text.</c>).</param>
    public static T5GemmaEncoderParameters From(ModelSource source, string prefix) => new(
        source.Arch<int>(prefix + "hidden_size"), source.Arch<int>(prefix + "layer_count"), source.Arch<int>(prefix + "intermediate_size"), source.Arch<int>(prefix + "head_count"),
        source.Arch<int>(prefix + "kv_head_count"), source.Arch<int>(prefix + "head_dim"), source.Arch(prefix + "norm_eps", 1e-6f), source.Arch(prefix + "rope_base", 10000f),
        source.Arch(prefix + "softcap", 0f), source.Arch<float>(prefix + "query_scale"), source.Arch(prefix + "sliding_window", 0),
        source.Arch(prefix + "sliding_layers", Array.Empty<long>()));
}
