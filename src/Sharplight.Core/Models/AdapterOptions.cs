namespace Sharplight.Models;

/// <summary>Options passed to adapters.</summary>
public sealed class AdapterOptions
{
    /// <summary>Data type of attention KV caches.</summary>
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

    public AdapterOptions WithContext(int contextLength) => new()
    {
        KvCacheType = KvCacheType, IncludeAuxiliaryGraphs = IncludeAuxiliaryGraphs, ContextLength = contextLength,
        EmbeddingPooling = EmbeddingPooling, EmbeddingNormalize = EmbeddingNormalize,
    };
}
