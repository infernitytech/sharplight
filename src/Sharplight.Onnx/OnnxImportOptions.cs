namespace Sharplight.Onnx;

/// <summary>
/// Options for importing and translating an ONNX model. Instances are read during translation and are not modified by it;
/// they may be shared between imports.
/// </summary>
public sealed class OnnxImportOptions
{
    /// <summary>
    /// Full-shape overrides per graph input name, e.g. <c>["pixel_values"] = [-1, 3, 384, 384]</c>.
    /// Use a value of 0 or less for the (symbolic) batch axis. Needed when the model has dynamic spatial dimensions.
    /// </summary>
    public Dictionary<string, long[]> InputShapes { get; init; } = new(StringComparer.Ordinal);

    /// <summary>
    /// Graph inputs pinned to compile-time constants (flat values, shape from <see cref="InputShapes"/> or the declared dims), e.g.
    /// <c>["sr"] = [16000]</c> or an <c>orig_im_size</c> that determines output shapes. They are not runtime inputs of the compiled graph.
    /// </summary>
    public Dictionary<string, double[]> ConstantInputs { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Makes the outermost dimension of every input symbolic ("N") even if it is fixed in the file.</summary>
    public bool ForceSymbolicBatch { get; init; }

    /// <summary>
    /// Largest trip count of a <c>Loop</c> or <c>Scan</c> whose body depends on runtime tensors that is unrolled into the graph
    /// (bodies that only compute constants are evaluated at import time regardless, up to <see cref="MaxLoopIterations"/>).
    /// </summary>
    public int MaxUnrolledLoop { get; init; } = 64;

    /// <summary>Upper limit on the iterations of a <c>Loop</c> evaluated or unrolled at import time.</summary>
    public int MaxLoopIterations { get; init; } = 100_000;

    /// <summary>
    /// Upper bound on the iterations of a runtime <c>Loop</c> whose iteration count is not known at import time (a runtime trip count or
    /// condition) and that has scan outputs: their buffers are sized for it, and a run that needs more iterations fails.
    /// </summary>
    public int RuntimeLoopScanBound { get; init; } = 4096;

    /// <summary>Placement group size: number of compute-heavy ops (Conv/MatMul/Gemm/BatchMatMul) per group; values below 1 act as 1.</summary>
    public int GroupSize { get; init; } = 8;
}
