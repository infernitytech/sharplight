using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>
/// The batched form of <see cref="StandardOps.Attention"/>: attributes <see cref="RowsAttribute"/> and <see cref="KeysAttribute"/>
/// split the query and key rows into equal-length sequences that attend only within themselves.
/// </summary>
public static class BatchedAttention
{
    /// <summary>Attribute: query rows per sequence (long; 0 or absent = not batched).</summary>
    public const string RowsAttribute = "sequence_rows";

    /// <summary>Attribute: key rows per sequence (long).</summary>
    public const string KeysAttribute = "sequence_keys";

    /// <summary>Query rows per sequence of an Attention node, or 0 when it is not in the batched form.</summary>
    /// <param name="node">An Attention node.</param>
    public static long SequenceRows(Node node) => node.Attributes.Get(RowsAttribute, 0L);

    /// <summary>Key rows per sequence of an Attention node in the batched form.</summary>
    /// <param name="node">An Attention node.</param>
    public static long SequenceKeys(Node node) => node.Attributes.Get(KeysAttribute, 0L);

    /// <summary>
    /// Whether a batched Attention node is well formed: both lengths positive, no spans, causal mask, window, ALiBi or position
    /// bias, and an optional f32 mask (per key row [S] or per query and key row [T, S]).
    /// </summary>
    /// <param name="node">An Attention node with <see cref="RowsAttribute"/> set.</param>
    public static bool IsValid(Node node)
    {
        var a = node.Attributes;
        bool plain = !a.Get("causal", false) && a.Get("window", 0) == 0 && a.Get<float[]?>("alibi_slopes", null) is null
                     && a.Get<float[]?>("position_bias", null) is null;
        bool maskOk = node.Inputs.Length == 3 || (node.Inputs.Length == 4 && node.Inputs[3].Type == DataType.F32 && node.Inputs[3].Rank is 1 or 2);
        return SequenceRows(node) > 0 && SequenceKeys(node) > 0 && plain && maskOk;
    }
}
