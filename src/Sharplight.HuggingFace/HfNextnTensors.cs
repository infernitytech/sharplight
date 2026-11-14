namespace Sharplight.HuggingFace;

/// <summary>
/// The tensors only a DeepSeek-style multi-token-prediction (NextN) layer has, which checkpoints append after the decoder layers
/// (<c>model.layers.{num_hidden_layers + j}.</c>). The rest of such a layer is an ordinary decoder layer; the canonical form keeps it
/// below <c>mtp.layers.{j}.</c> (see <c>MtpDraftHead.FirstNextnModule</c>).
/// </summary>
internal static class HfNextnTensors
{
    /// <summary>
    /// HF suffix to canonical suffix. The per-module embedding and LM head copies map to null: the modules share the trunk's, which
    /// the draft head uses instead (no second copy is loaded).
    /// </summary>
    public static IReadOnlyDictionary<string, string?> Suffixes { get; } = new Dictionary<string, string?>(StringComparer.Ordinal)
    {
        ["eh_proj.weight"] = "fc.weight",
        ["enorm.weight"] = "embedding_norm.weight",
        ["hnorm.weight"] = "hidden_norm.weight",
        ["shared_head.norm.weight"] = "output_norm.weight",
        ["shared_head.head.weight"] = null,
        ["embed_tokens.weight"] = null,
    };
}
