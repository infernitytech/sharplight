using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Shapes, macros and kernel-choice rules of one weight matmul (<c>y = x · Wᵀ</c>), shared by the launches
/// <see cref="GraphLowerer"/> emits for it (matvec, GEMM, f16 activation conversion, split-K).
/// </summary>
internal sealed class MatMulPlan
{
    /// <summary>The MatMul node being lowered.</summary>
    public required Node Node { get; init; }

    /// <summary>Activations, already in a layout the kernels can read (see <c>TryRows</c>).</summary>
    public required TensorRef X { get; init; }

    /// <summary>The weight matrix [N, K].</summary>
    public required TensorRef W { get; init; }

    /// <summary>The result tensor.</summary>
    public required TensorRef Y { get; init; }

    /// <summary>Output features (rows of W).</summary>
    public required long N { get; init; }

    /// <summary>Reduction length (columns of W).</summary>
    public required long K { get; init; }

    /// <summary>Row stride of <see cref="X"/> in elements.</summary>
    public required long XStride { get; init; }

    /// <summary>Rows of the activation matrix per leading index of <see cref="X"/> (middle dimensions collapsed).</summary>
    public required long RowsPerLead { get; init; }

    /// <summary>Kernel macros shared by every variant.</summary>
    public required Dictionary<string, string> Macros { get; init; }

    /// <summary>True when tensor-core GEMMs are usable for this matmul.</summary>
    public required bool Coop { get; init; }

    /// <summary>True when the workgroup-matrix GEMM replaces gemm_cm2 for prefill.</summary>
    public required bool WorkgroupMatrix { get; init; }

    /// <summary>Split-K factor for a given number of activation rows.</summary>
    public required Func<long, int> SplitRows { get; init; }

    /// <summary>Activation rows for these bindings.</summary>
    public long RowCount(Bindings bindings) => GraphLowerer.Rows(bindings, X) * RowsPerLead;

    /// <summary>Split-K factor for these bindings.</summary>
    public int Split(Bindings bindings) => SplitRows(RowCount(bindings));
}
