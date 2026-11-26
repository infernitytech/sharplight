namespace Sharplight.Distributed;

/// <summary>
/// How f32 boundary activations travel between pipeline stages on different machines (<see cref="RemoteSegmentOptions.BoundaryPrecision"/>).
/// The narrower forms halve the bytes on the wire; the next stage then computes from rounded activations, so tokens and logits may differ
/// slightly from a run in one process.
/// </summary>
public enum BoundaryPrecision
{
    /// <summary>The f32 values as computed (default): the results equal a run in one process.</summary>
    Float32,

    /// <summary>IEEE half precision, rounded to nearest even (11 significant bits; values beyond ±65504 become infinite).</summary>
    Float16,

    /// <summary>bfloat16, rounded to nearest even (8 significant bits, the f32 range).</summary>
    BFloat16,
}
