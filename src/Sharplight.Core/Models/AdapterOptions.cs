namespace Sharplight.Models;

/// <summary>Options passed to adapters.</summary>
public sealed class AdapterOptions
{
    /// <summary>
    /// Data type of attention KV caches: f16 (default), f32, or the block-quantized q8_0 / q4_0 (see <see cref="KvCacheTypeFor"/>
    /// for rows that are not whole blocks).
    /// </summary>
    public Tensors.DataType KvCacheType { get; init; } = Tensors.DataType.F16;
    /// <summary>Emit auxiliary graphs (multi-token prediction heads, embedding-input decode, ...).</summary>
    public bool IncludeAuxiliaryGraphs { get; init; } = true;
    /// <summary>
    /// Context length the model will run with (0 = unknown, use the model's maximum). Lets an adapter make context-dependent
    /// choices, e.g. longrope's short vs. long frequency factors.
    /// </summary>
    public int ContextLength { get; init; }
    /// <summary>Pooling of the <see cref="WellKnown.EmbedGraph"/> ("mean", "cls", "last", "none"); null = what the model declares.</summary>
    public string? EmbeddingPooling { get; init; }
    /// <summary>Whether <see cref="WellKnown.EmbedGraph"/> L2-normalizes; null = what the model declares.</summary>
    public bool? EmbeddingNormalize { get; init; }

    /// <summary>
    /// The type of a KV cache whose rows (per position and head) hold <paramref name="rowLength"/> values: <see cref="KvCacheType"/>,
    /// or f16 when that type is block-quantized and the row is not a whole number of blocks.
    /// </summary>
    public Tensors.DataType KvCacheTypeFor(long rowLength) => KvRowType(KvCacheType, rowLength);

    /// <summary>
    /// <paramref name="requested"/>, or f16 when it is block-quantized and rows of <paramref name="rowLength"/> values are not a
    /// whole number of blocks.
    /// </summary>
    public static Tensors.DataType KvRowType(Tensors.DataType requested, long rowLength)
        => requested.BlockSize > 1 && rowLength % requested.BlockSize != 0 ? Tensors.DataType.F16 : requested;

    /// <summary>Returns a copy of these options with another <see cref="ContextLength"/>.</summary>
    public AdapterOptions WithContext(int contextLength) => With(KvCacheType, contextLength);

    /// <summary>Returns a copy of these options with another <see cref="KvCacheType"/> and <see cref="ContextLength"/>.</summary>
    public AdapterOptions With(Tensors.DataType kvCacheType, int contextLength) => new()
    {
        KvCacheType = kvCacheType, IncludeAuxiliaryGraphs = IncludeAuxiliaryGraphs, ContextLength = contextLength,
        EmbeddingPooling = EmbeddingPooling, EmbeddingNormalize = EmbeddingNormalize,
    };
}
