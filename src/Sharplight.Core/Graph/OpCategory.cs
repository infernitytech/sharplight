namespace Sharplight.Graph;

/// <summary>Broad classification used by optimization passes (e.g. elementwise fusion) and planners.</summary>
public enum OpCategory
{
    /// <summary>Pointwise op with numpy-style broadcasting; freely fusable.</summary>
    Elementwise,
    /// <summary>Re-interprets the input buffer (reshape, slice, permute): no computation, aliases its input.</summary>
    View,
    /// <summary>Reduction / normalization over the innermost axis.</summary>
    Reduction,
    /// <summary>Matrix multiplication family.</summary>
    MatMul,
    /// <summary>Data movement (gather, concat, copy, cache writes).</summary>
    Data,
    /// <summary>Higher-level fused operator (attention, rope, recurrent rules, ...).</summary>
    Composite,
    /// <summary>Runs subgraphs (If): executed on the host between device segments, never by a device.</summary>
    ControlFlow,
}
