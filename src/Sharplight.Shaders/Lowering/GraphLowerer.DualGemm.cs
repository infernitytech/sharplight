using System.Globalization;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Emits the dual gemm_cm2 of a matmul pair sharing x (gate / up of a gated FFN) with f16 or bf16 weights for the prefill batches
    /// <see cref="DualGemm2Serves"/> accepts: one launch reads the f16 activations once, multiplies both weight matrices (their
    /// 16-row fragments alternate in the B tile) and stores the fused elementwise group <paramref name="epilogueSource"/> defines as
    /// <c>epi2</c>, instead of two GEMMs writing two f32 products and an elementwise launch reading them back. Every product is
    /// accumulated exactly as in the separate GEMM (same K steps, same tensor-core operations), and the epilogue is the elementwise
    /// kernel's expression, so the output equals that of the separate launches. Returns the rows it serves, or null when it does
    /// not apply.
    /// </summary>
    private Func<long, bool>? EmitDualGemm2(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef weightA,
        TensorRef weightB, TensorRef y, long n, long k, string epilogueSource, Value? output = null)
    {
        long rowsMax = x.Shape[0] * rowsPerLead;
        var layout = Gemm2Layout(Gemm2TileRows, splitHalf: false);
        if (!DiagnosticSwitches.DualGemm2 || !DiagnosticSwitches.Gemm2 || !DiagnosticSwitches.GemmHalfActivations || PreferWorkgroupMatrixGemm
            || (weightA.Type != DataType.F16 && weightA.Type != DataType.BF16) || weightB.Type != weightA.Type
            || !CoopUsable(nodes[0], weightA.Type, k) || k % 64 != 0 || DiagnosticSwitches.GemmBlockK != 64 || layout[3] % 2 != 0
            || rowsMax <= DualGemm2MinRows)
        {
            return null;
        }

        long wave = GemmWaveTiles;
        bool Serves(long rows) => DualGemm2Serves(rows, n, k, wave);
        var quantized = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, DualGemm2MinRows + 1, rowsMax, ActivationCode.F16);
        var macros = Gemm2Macros(Macros(("WFMT", WeightFormat(weightA.Type)!.Value), ("K", k), ("N", n), ("X_STRIDE", k), ("Y_STRIDE", n),
            ("ROW_BYTES", weightA.Type.ByteSize(k) + "ull"), ("X_F16", 1), ("DUAL", 1), ("KERNEL_GEMM_CM2", 1)), Gemm2TileRows);
        if (DiagnosticSwitches.GemmTileGroup > 0 && 2 * weightA.Type.ByteSize(k) * n >= TileGroupMinWeightBytes)
        {
            // Both matrices stream through the tiles: their sum decides the grouped order.
            macros["GEMM_GROUP"] = DiagnosticSwitches.GemmTileGroup.ToString(CultureInfo.InvariantCulture);
        }

        // The f16 copy of y that a following tensor-core matmul reads (e.g. the down projection after gate / up), stored next
        // to y instead of a conversion launch reading y back (Qwen3-8B bf16 at 2048 rows: -0.15 ms per layer).
        var half = output is null ? null : ProducerHalfCopy(output, y, Serves);
        if (half is not null)
        {
            macros["DUAL_Y16"] = "1";
        }

        var kernel = _compiler.Compile("matmul_dual.slang", epilogueSource, "gemm_cm2", _opt.Target, macros, _opt.Cache);
        int outputs = Gemm2TileColumns / 2;
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        Emit($"{label} gemm2", nodes,
            [new KernelVariant(b => Serves(rows(b)), kernel, b => new Grid(Grid.Div(n, outputs), Grid.Div(rows(b), Gemm2TileRows)))],
            half is null
                ? [Pointer(quantized), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)]
                : [Pointer(quantized), Pointer(weightA), Pointer(weightB), Pointer(y), Pointer(half), TokenArg(x, rowsPerLead)],
            [quantized, weightA, weightB], half is null ? [y] : [y, half]);
        return Serves;
    }

    /// <summary>Rows up to which the dual gemm_cm2 never serves (smaller batches take 32- or 64-token tiles).</summary>
    private const long DualGemm2MinRows = 64;

    /// <summary>
    /// True when the dual gemm_cm2 serves <paramref name="rows"/> activation rows of an [<paramref name="k"/>→<paramref name="n"/>]
    /// pair: 128-token tiles, and only where the separate GEMMs run without K splits, so that each product sums its K steps in the
    /// same order.
    /// </summary>
    private static bool DualGemm2Serves(long rows, long n, long k, long wave)
        => rows > DualGemm2MinRows && Gemm2TileRowsFor(rows) == Gemm2TileRows
           && ChooseSplit(n, k, rows, Gemm2TileRows, Gemm2TileColumns, 64, wave) == 1;
}
