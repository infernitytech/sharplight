using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>Operands, dimensions and kernel macros of one Attention / CachedAttention node being lowered.</summary>
internal sealed class AttentionPlan
{
    /// <summary>The attention node.</summary>
    public required Node Node { get; init; }

    /// <summary>True for bidirectional "Attention" over plain K/V tensors; false for attention over a KV cache.</summary>
    public required bool Full { get; init; }

    /// <summary>Queries [rows, HQ, D] with a contiguous innermost axis.</summary>
    public required TensorRef Q { get; init; }

    /// <summary>Keys [positions, HKV, D].</summary>
    public required TensorRef K { get; init; }

    /// <summary>Values [positions, HKV, DV].</summary>
    public required TensorRef V { get; init; }

    /// <summary>The attention output [rows, HQ, DV].</summary>
    public required TensorRef Output { get; init; }

    /// <summary>Optional per-head attention sinks [HQ] (f32).</summary>
    public TensorRef? Sinks { get; init; }

    /// <summary>Multi-sequence form: cache row of each sequence slot.</summary>
    public TensorRef? Seq { get; init; }

    /// <summary>Multi-sequence form: runs table of (first row, count) per sequence.</summary>
    public TensorRef? Runs { get; init; }
    /// <summary>Paged cache: the block table [slots, blocks per slot] (null for a contiguous cache).</summary>
    public TensorRef? Pages { get; init; }

    /// <summary>Query-conditioned relative position bias [T, HQ, E] (attribute <c>relative_bias</c> = E), or null.</summary>
    public TensorRef? RelativeBias { get; init; }

    /// <summary>Learned relative position bias by distance, f32 [HQ · (2R + 1)] (attribute <c>position_bias</c>, a constant buffer), or null.</summary>
    public TensorRef? PositionBias { get; init; }

    /// <summary>Sparse attention: the key positions each query row attends, i32 [rows, K] (entries below 0 are skipped).</summary>
    public TensorRef? Selection { get; init; }

    /// <summary>Compressed cache of a compressed attention (entries of pooled windows joining the softmax); null otherwise.</summary>
    public TensorRef? Compressed { get; init; }

    /// <summary>
    /// Cached attention with attribute <c>lookahead</c>: i32 [A], rows of the pass each query row also sees after its own (A = 1: none); only
    /// the per-(head, query) kernel with <c>LOOKAHEAD</c> reads it.
    /// </summary>
    public TensorRef? Lookahead { get; init; }

    /// <summary>"Attention" with i32 spans: the key range each query may see.</summary>
    public TensorRef? Spans { get; init; }

    /// <summary>"Attention" with an additive f32 score mask ([rows, key rows], or [key rows] for every query).</summary>
    public TensorRef? Mask { get; init; }

    /// <summary>Batched "Attention": query rows per sequence (0 when not batched).</summary>
    public long SequenceRows { get; init; }

    /// <summary>"Attention" over the keys of another sequence (cross-attention): the key row count is a kernel parameter.</summary>
    public bool Cross { get; init; }

    /// <summary>Query heads.</summary>
    public required long HQ { get; init; }

    /// <summary>Key/value heads.</summary>
    public required long HKV { get; init; }

    /// <summary>Query/key head size.</summary>
    public required long D { get; init; }

    /// <summary>Value head size.</summary>
    public required long DV { get; init; }

    /// <summary>Kernel macros shared by every attention kernel.</summary>
    public required Dictionary<string, string> Macros { get; init; }
}
