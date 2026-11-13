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

    /// <summary>"Attention" with i32 spans: the key range each query may see.</summary>
    public TensorRef? Spans { get; init; }

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
