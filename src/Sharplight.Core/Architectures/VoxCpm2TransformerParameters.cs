using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The sizes of one of VoxCPM2's local (bidirectional, MiniCPM-4 layer) transformers.</summary>
/// <param name="Hidden">The width.</param>
/// <param name="Ffn">The gated feed-forward width.</param>
/// <param name="Heads">The query heads.</param>
/// <param name="KvHeads">The key and value heads.</param>
/// <param name="HeadDim">The head width.</param>
/// <param name="Layers">The layers.</param>
internal sealed record VoxCpm2TransformerParameters(int Hidden, int Ffn, int Heads, int KvHeads, int HeadDim, int Layers)
{
    /// <summary>Reads <c>{prefix}{hidden_size,ffn_hidden_size,head_count,kv_head_count,head_dim,layer_count}</c>.</summary>
    /// <param name="src">The source.</param>
    /// <param name="prefix">The key prefix, for example <c>voxcpm2.encoder.</c>.</param>
    public static VoxCpm2TransformerParameters From(ModelSource src, string prefix)
        => new(src.Get<int>(prefix + "hidden_size"), src.Get<int>(prefix + "ffn_hidden_size"), src.Get<int>(prefix + "head_count"), src.Get<int>(prefix + "kv_head_count"),
            src.Get<int>(prefix + "head_dim"), src.Get<int>(prefix + "layer_count"));
}
