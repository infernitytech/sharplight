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

    /// <summary>Largest batch the dual fp8 GEMM serves (one 64-row tile); larger batches run the two GEMMs, which are as fast there.</summary>
    private const long DualFp8MaxRows = 64;

    /// <summary>
    /// Emits the dual fp8 GEMM of a matmul pair sharing x (gate / up) for <see cref="Fp8GemmMinRows"/> to
    /// <see cref="DualFp8MaxRows"/> rows: one launch reads the E4M3 activations once, multiplies both weight matrices (their
    /// 16-row fragments alternate in the B tile) and stores the fused elementwise group <paramref name="epilogueSource"/>
    /// defines as <c>epi2</c>, instead of two split-K GEMMs, their reductions and an elementwise launch. No split-K: a
    /// workgroup covers half as many outputs, so the grid has twice the column tiles of a single matmul. From 65 rows the
    /// separate GEMMs (no split-K there) are as fast, and keep prompt prefill bit-identical to them. Returns the largest
    /// batch it serves, or 0 when it does not apply.
    /// </summary>
    private long EmitDualFp8Gemm(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef weightA,
        TensorRef weightB, TensorRef y, long n, long k, string epilogueSource)
    {
        long minRows = Fp8GemmMinRows;
        long rowsMax = x.Shape[0] * rowsPerLead;
        if (!DiagnosticSwitches.Fp8DualGemm || !UseFp8Gemm(weightA.Type, k) || UseMmq(weightA.Type, n, k) || rowsMax < minRows
            || MmqLayout(64)[3] % 2 != 0)
        {
            return 0;
        }

        var quantized = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, minRows, rowsMax, ActivationCode.Fp8);
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        const int tileRows = (int)DualFp8MaxRows;
        int outputs = MmqColumns(tileRows) / 2;
        var macros = Fp8GemmMacros(weightA.Type, n, k, tileRows, 1);
        macros["DUAL"] = "1";
        macros["KERNEL_GEMM_FP8"] = "1";
        var kernel = _compiler.Compile("gemm_fp8_dual.slang", epilogueSource, "gemm_fp8", _opt.Target, macros, _opt.Cache);
        var variant = new KernelVariant(b => rows(b) >= minRows && rows(b) <= DualFp8MaxRows, kernel,
            b => new Grid(Grid.Div(rows(b), tileRows), Grid.Div(n, outputs)));
        Emit($"{label} fp8", nodes, [variant], [Pointer(quantized), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)],
            [quantized, weightA, weightB], [y]);
        return DualFp8MaxRows;
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
