using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// True when the fp8 tensor-core GEMM (<c>gemm_fp8.slang</c>, <see cref="DeviceFeatures.MatrixFp8"/>) serves
    /// prefill-sized batches of a matmul with block-scaled E4M3 weights: activations quantized to E4M3 with one scale per
    /// 128 values, f8 x f8 MMAs with f32 accumulation, one scaling per weight block. Without the feature (or with
    /// <c>SHARPLIGHT_FP8_GEMM=0</c>) the f16 tensor-core GEMM decodes the weights instead.
    /// </summary>
    private bool UseFp8Gemm(DataType type, long k)
        => DiagnosticSwitches.Fp8Gemm && _opt.Has(DeviceFeatures.MatrixFp8) && _opt.Has(DeviceFeatures.SubgroupOps) && _opt.SubgroupSize == 32
           && (type == DataType.F8E4M3_B128 || type == DataType.F8E4M3_B32) && k % 128 == 0;

    /// <summary>Emits the E4M3 activation quantization and the fp8 GEMM launches of a planned matmul (see <see cref="UseFp8Gemm"/>).</summary>
    private void EmitFp8Gemm(MatMulPlan p) => EmitQuantizedGemm(p, "fp8", ActivationCode.Fp8, 128,
        (tileRows, split) => Lib("gemm_fp8.slang", "gemm_fp8", Fp8GemmMacros(p.W.Type, p.N, p.K, tileRows, split)), tokenTilesFirst: true);

    /// <summary>
    /// Largest batch for which the dual fp8 GEMM serves every row count (one 64-row tile). Larger batches use it only where the
    /// separate GEMMs would not split K (see <see cref="DualFp8Serves"/>).
    /// </summary>
    private const long DualFp8AlwaysMaxRows = 64;

    /// <summary>
    /// True when the dual fp8 GEMM serves <paramref name="rows"/> activation rows of an [<paramref name="k"/>→<paramref name="n"/>]
    /// pair: from <see cref="Fp8GemmMinRows"/> to 64 rows always; above, only where the separate fp8 GEMMs run without split-K.
    /// There each output element sums its K blocks in the same order as the separate launch (the accumulation is an explicit
    /// fused multiply-add in both), and the epilogue evaluates the same expression as the elementwise kernel, so prompt
    /// prefill stays bit-identical to the separate launches. Where they split K, their partial sums round differently.
    /// <c>SHARPLIGHT_FP8_DUAL_MAX_ROWS=64</c> keeps larger batches on the separate launches.
    /// </summary>
    private static bool DualFp8Serves(long rows, long n, long k)
        => rows >= Fp8GemmMinRows && (rows <= DualFp8AlwaysMaxRows
                                      || (rows <= (DiagnosticSwitches.Fp8DualGemmMaxRows ?? long.MaxValue) && QuantizedGemmSplit(n, k, rows, 128) == 1));

    /// <summary>
    /// Emits the dual fp8 GEMM of a matmul pair sharing x (gate / up) for the batches <see cref="DualFp8Serves"/> accepts: one
    /// launch reads the E4M3 activations once, multiplies both weight matrices (their 16-row fragments alternate in the B tile)
    /// and stores the fused elementwise group <paramref name="epilogueSource"/> defines as <c>epi2</c>, instead of two GEMMs
    /// (with split-K partials and reductions at small batches) and an elementwise launch. No split-K: a workgroup covers half as
    /// many outputs, so the grid has twice the column tiles of a single matmul. Returns false when it does not apply.
    /// </summary>
    private bool EmitDualFp8Gemm(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef weightA,
        TensorRef weightB, TensorRef y, long n, long k, string epilogueSource)
    {
        long minRows = Fp8GemmMinRows;
        long rowsMax = x.Shape[0] * rowsPerLead;
        if (!DiagnosticSwitches.Fp8DualGemm || !UseFp8Gemm(weightA.Type, k) || UseMmq(weightA.Type, n, k) || rowsMax < minRows
            || MmqLayout(64)[3] % 2 != 0)
        {
            return false;
        }

        var quantized = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, minRows, rowsMax, ActivationCode.Fp8);
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        const int tileRows = (int)DualFp8AlwaysMaxRows;
        int outputs = MmqColumns(tileRows) / 2;
        var macros = Fp8GemmMacros(weightA.Type, n, k, tileRows, 1);
        macros["DUAL"] = "1";
        macros["KERNEL_GEMM_FP8"] = "1";
        var kernel = _compiler.Compile("gemm_fp8_dual.slang", epilogueSource, "gemm_fp8", _opt.Target, macros, _opt.Cache);
        var variant = new KernelVariant(b => DualFp8Serves(rows(b), n, k), kernel,
            b => new Grid(Grid.Div(rows(b), tileRows), Grid.Div(n, outputs)));
        Emit($"{label} fp8", nodes, [variant], [Pointer(quantized), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)],
            [quantized, weightA, weightB], [y]);
        return true;
    }

    private Dictionary<string, string> Fp8GemmMacros(DataType type, long n, long k, int tileRows, int split)
    {
        var layout = MmqLayout(tileRows);
        if (layout[0] * layout[2] * 16 != tileRows)
        {
            throw new NotSupportedException($"fp8 GEMM layout {string.Join(',', layout)} does not give {tileRows}-token tiles.");
        }

        return Macros(("KERNEL_GEMM_FP8", 1), ("WBLOCK", type.BlockSize), ("K", k), ("N", n), ("Y_STRIDE", n), ("ROW_BYTES", type.ByteSize(k) + "ull"),
            ("F8_WM", layout[0]), ("F8_WN", layout[1]), ("F8_FM", layout[2]), ("F8_FN", layout[3]), ("SPLITK", split),
            ("F8_GENERIC_LAYOUT", DiagnosticSwitches.MmqGenericLayout));
    }
}
