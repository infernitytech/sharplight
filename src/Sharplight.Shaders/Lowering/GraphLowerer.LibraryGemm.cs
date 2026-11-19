using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Fewest rows a matmul runs on the vendor GEMM library with (<c>SHARPLIGHT_LIBRARY_GEMM_MIN_ROWS</c>, default 256): below that the
    /// generated GEMMs' small-batch tiles and K splits are as fast and skip the bf16 conversion of x.
    /// </summary>
    private static long LibraryGemmMinRows => DiagnosticSwitches.LibraryGemmMinRows ?? 256;

    /// <summary>
    /// True when matmul <paramref name="p"/> runs on the vendor GEMM library (<see cref="DeviceFeatures.LibraryGemm"/>) for batches of
    /// <see cref="LibraryGemmMinRows"/> rows and more: bf16 weights in their plain [N, K] layout, one activation row per output row,
    /// no epilogue to fuse, and batches that large are possible.
    /// </summary>
    private bool UseLibraryGemm(MatMulPlan p, bool epilogue)
        => !epilogue && !p.SplitHalf && LibraryGemmUsable(p.W.Type, p.N, p.K, p.RowsPerLead, p.X.Shape[0]);

    /// <summary>
    /// True when a matmul of <paramref name="type"/> weights [<paramref name="n"/>, <paramref name="k"/>] with up to
    /// <paramref name="maxRows"/> activation rows can run on the vendor GEMM library (see <see cref="UseLibraryGemm"/>).
    /// </summary>
    private bool LibraryGemmUsable(DataType type, long n, long k, long rowsPerLead, long maxRows)
        => _opt.Has(DeviceFeatures.LibraryGemm) && type == DataType.BF16 && rowsPerLead == 1 && k % 8 == 0 && n % 8 == 0
           && maxRows >= LibraryGemmMinRows;

    /// <summary>
    /// The library GEMM of <paramref name="p"/> for <see cref="LibraryGemmMinRows"/> rows and more: a bf16 copy of x (shared by the
    /// matmuls reading the same rows) and the library launch.
    /// </summary>
    private void EmitLibraryGemm(MatMulPlan p)
    {
        string label = MatMulLabel(p);
        var x = QuantizedActivations(label, [p.Node], p.Node.Inputs[0], p.X, p.XStride, 1, p.K, LibraryGemmMinRows, p.X.Shape[0], ActivationCode.BF16);
        var kernel = LibraryKernel.Gemm(_opt.Target, "bf16", p.N, p.K);
        Emit(label + " library", [p.Node],
            [new KernelVariant(b => p.RowCount(b) >= LibraryGemmMinRows, kernel, _ => new Grid(1))],
            [Pointer(x), Pointer(p.W), Pointer(p.Y), TokenArg(p.X)],
            [x, p.W], [p.Y]);
    }
}
