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

    /// <summary>
    /// Number of elements of sequence-typed graph inputs. Every element becomes a graph input of its own named
    /// <see cref="OnnxGraphAdapter.SequenceElementName"/>, whose shape <see cref="InputShapes"/> gives under that name (and which
    /// <see cref="ConstantInputs"/> can pin).
    /// </summary>
    public Dictionary<string, int> SequenceInputLengths { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Optional-typed graph inputs that are empty; every other optional input holds a value (a tensor or a sequence).</summary>
    public HashSet<string> AbsentInputs { get; init; } = new(StringComparer.Ordinal);

    /// <summary>
    /// Declared upper bounds of the values of integer graph inputs (all values in <c>[0, bound]</c>), e.g. <c>["text_lengths"] = 512</c>.
    /// They let a <c>Range</c> whose limit is computed from such an input get a bounded dynamic length; a value outside the
    /// declared range at run time yields a truncated result.
    /// </summary>
    public Dictionary<string, long> InputValueBounds { get; init; } = new(StringComparer.Ordinal);

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

    /// <summary>
    /// Lets block-quantized weights (<c>MatMulNBits</c>) whose f32 block scales are not exactly representable in f16 be stored
    /// in the 4.5 / 8.5-bit block formats with the scales rounded to f16 (relative error at most 2⁻¹¹ per block, about 5e-4
    /// relative RMS on BERT outputs). Off by default: such weights are then dequantized to f32 (exact, 7x the memory traffic).
    /// Weights with f16 scales are always kept in block form, since that is exact.
    /// </summary>
    public bool AllowHalfBlockScales { get; init; }

    /// <summary>Placement group size: number of compute-heavy ops (Conv/MatMul/Gemm/BatchMatMul) per group; values below 1 act as 1.</summary>
    public int GroupSize { get; init; } = 8;
}
