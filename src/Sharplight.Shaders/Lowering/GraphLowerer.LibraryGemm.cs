using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Fewest rows a matmul runs on the vendor GEMM library with (<c>SHARPLIGHT_LIBRARY_GEMM_MIN_ROWS</c>, default 256): below that the
    /// generated GEMMs' small-batch tiles and K splits are as fast.
    /// </summary>
    private static long LibraryGemmMinRows => DiagnosticSwitches.LibraryGemmMinRows ?? 256;

    /// <summary>
    /// Fewest rows a matmul with a bias row runs on the hand-written f16 GEMM with the bias in its store (<see cref="UseNativeBiasGemm"/>):
    /// below, gemm_cm2 splits K and adds the bias in its reduction, which was faster for gpt-oss-20b's projections at 256 and 512 rows
    /// (mixed serving steps of 64 + 192 / 64 + 448 rows: 28.49 / 43.96 against 29.95 / 44.35 ms) and slower from 1024 rows (64 + 960
    /// rows 74.57 against 73.73 ms; a 2048-row prefill 15,543 → 15,834 tok/s).
    /// </summary>
    private static long NativeBiasGemmMinRows => Math.Max(LibraryGemmMinRows, 1024);

    /// <summary>
    /// Fewest outputs a matmul runs on the library path with: narrower products (an MoE router's 32 or 128 expert logits) give the
    /// hand-written GEMM's 128-row token tiles one column tile, a grid of rows / 128 blocks (16 at 2,048 rows on 82 SMs), where the
    /// generated GEMM splits K. RTX PRO 4500, gpt-oss-20b's router [2880 → 32] at 2,048 rows: 39.6 µs native, 16.9 + 3.9 µs on
    /// gemm_cm2 with split-K and its reduction (which adds the bias).
    /// </summary>
    private const long LibraryGemmMinColumns = 256;

    /// <summary>
    /// True when the f16 products of <see cref="UseLibraryGemm"/> have a GEMM to run on: the hand-written one of the provider
    /// (<see cref="NativeGemmF16"/>) or the vendor library (<see cref="DeviceFeatures.LibraryGemm"/>).
    /// </summary>
    private bool LibraryGemmAvailable => NativeGemmF16 || _opt.Has(DeviceFeatures.LibraryGemm);

    /// <summary>
    /// True when matmul <paramref name="p"/> runs on the vendor GEMM library (<see cref="DeviceFeatures.LibraryGemm"/>) for batches of
    /// <see cref="LibraryGemmMinRows"/> rows and more: f16 weights (bf16 weights are stored as f16 for it, see
    /// <see cref="LibraryHalfWeight"/>) in their plain [N, K] layout, one activation row per output row, no epilogue to fuse, and
    /// batches that large are possible.
    /// </summary>
    private bool UseLibraryGemm(MatMulPlan p, bool epilogue)
        => !epilogue && !p.SplitHalf && LibraryGemmUsable(p.W.Type, p.N, p.K, p.RowsPerLead, p.X.Shape[0]);

    /// <summary>
    /// True when matmul <paramref name="p"/> with the epilogue <paramref name="epilogue"/> runs on the hand-written f16 GEMM with the
    /// epilogue in its store from <see cref="NativeBiasGemmMinRows"/> rows: the epilogue adds a constant f32 bias row of the output's width
    /// (the q / k / v / o biases of gpt-oss or Qwen2), and the matmul suits the library path. <c>SHARPLIGHT_NATIVE_GEMM_BIAS=0</c> keeps
    /// such matmuls on the generated GEMMs.
    /// </summary>
    private bool UseNativeBiasGemm(MatMulPlan p, MatMulEpilogue? epilogue)
        => epilogue is { Group: [{ Op.Name: "Add" } add] } && add.Output == epilogue.Output && DiagnosticSwitches.NativeGemmBias && NativeGemmF16
           && !p.SplitHalf && LibraryGemmUsable(p.W.Type, p.N, p.K, p.RowsPerLead, p.X.Shape[0]) && p.X.Shape[0] * p.RowsPerLead >= NativeBiasGemmMinRows
           && add.Inputs[add.Inputs[0] == p.Node.Output ? 1 : 0] is { Kind: ValueKind.Weight, Shape.Rank: 1 } row && row.Type == DataType.F32
           && row.Shape[0].Equals(p.Node.Inputs[1].Shape[0]) && (add.Inputs[0] == p.Node.Output || add.Inputs[1] == p.Node.Output);

    /// <summary>
    /// True when a matmul of <paramref name="type"/> weights [<paramref name="n"/>, <paramref name="k"/>] with up to
    /// <paramref name="maxRows"/> activation rows can run on the vendor GEMM library (see <see cref="UseLibraryGemm"/>).
    /// </summary>
    private bool LibraryGemmUsable(DataType type, long n, long k, long rowsPerLead, long maxRows)
        => LibraryGemmAvailable && _opt.Has(DeviceFeatures.MatrixF16) && type == DataType.F16 && rowsPerLead == 1 && n >= LibraryGemmMinColumns
           && k % 8 == 0 && n % 8 == 0 && (!NativeGemmF16 || k % 64 == 0) && maxRows >= LibraryGemmMinRows && DiagnosticSwitches.Gemm2 && DiagnosticSwitches.GemmHalfActivations;

    /// <summary>
    /// True when bf16 weight <paramref name="weight"/> is stored as f16 for the library GEMM: the device has it, every consumer in the
    /// segment reads the weight as the [N, K] matrix of a MatMul, and the shape suits the library. The library takes both operands in
    /// one type, and the activations are the generated GEMMs' f16 copies (row-ranged where values may exceed f16), so its products
    /// have the precision of the generated tensor-core GEMMs, which convert bf16 weights to f16 on load the same way.
    /// </summary>
    private bool LibraryHalfWeight(Value weight)
    {
        if (!LibraryGemmAvailable || weight.Type != DataType.BF16 || weight.Shape.Rank != 2 || weight.Shape.Dims.Any(d => d.IsSymbolic)
            || weight.Shape[0].Fixed % 8 != 0 || weight.Shape[1].Fixed % 8 != 0)
        {
            return false;
        }

        bool any = false;
        foreach (var node in _seg.Nodes)
        {
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                if (node.Inputs[i] != weight)
                {
                    continue;
                }

                if (node.Op.Name != "MatMul" || i != 1)
                {
                    return false;
                }

                any = true;
            }
        }

        return any;
    }

    /// <summary>
    /// The library GEMM of <paramref name="p"/> for <see cref="LibraryGemmMinRows"/> rows and more, on the f16 copy of x the
    /// generated GEMMs read (<paramref name="half"/>, else converted here), with the row-range fix-up of that copy.
    /// </summary>
    private void EmitLibraryGemm(MatMulPlan p, TensorRef? half, MatMulEpilogue? bias = null)
    {
        string label = MatMulLabel(p);
        var x = half ?? QuantizedActivations(label, [p.Node], p.Node.Inputs[0], p.X, p.XStride, 1, p.K, LibraryGemmMinRows, p.X.Shape[0], ActivationCode.F16);
        if (bias is not null)
        {
            // The hand-written GEMM adds the bias row in its store (and undoes the rows' range scales first).
            var add = bias.Group[0];
            var row = Ref(add.Inputs[add.Inputs[0] == p.Node.Output ? 1 : 0]);
            var y = Produce(bias.Output);
            long? rangeScales = _rangeScales.TryGetValue(x, out long xrs) ? xrs : null;
            var notMulti = NoteMultiGemmTallPart(p.Node, new MultiGemmTallPart(false, x, p.W, y, row, rangeScales, p.RowCount, NativeBiasGemmMinRows, p.X.Shape[0],
                TallCondition(NativeBiasGemmMinRows, p.RowCount), TokenArg(p.X)));
            Emit($"{label} native fused[Add]", [p.Node, add],
                Unless(NativeF16Variants(p.N, p.K, p.X.Shape[0], null, b => p.RowCount(b), NativeBiasGemmMinRows, bias: true, rangeScales: rangeScales), notMulti),
                [Pointer(x), Pointer(p.W), Pointer(y), Pointer(row), TokenArg(p.X)],
                [x, p.W, row], [y]);
            return;
        }

        // The provider's hand-written f16 GEMM where it has one, else the vendor library (same operands and output).
        // A member of a q / k / v group leaves the rows its multi-output form serves to it (the hand-written GEMM only).
        Func<Bindings, bool> notMultiTall = _ => true;
        if (NativeGemmF16 && p.Y.IsDense && p.Macros.GetValueOrDefault("Y_STRIDE") == p.N.ToString(CultureInfo.InvariantCulture))
        {
            long? xrs = _rangeScales.TryGetValue(x, out long rs) ? rs : null;
            notMultiTall = NoteMultiGemmTallPart(p.Node, new MultiGemmTallPart(false, x, p.W, p.Y, null, xrs, p.RowCount, LibraryGemmMinRows, p.X.Shape[0],
                TallCondition(LibraryGemmMinRows, p.RowCount), TokenArg(p.X)));
        }

        // The hand-written GEMM undoes the rows' range scales in its store (H_XRS); the library's products are unscaled below.
        long? ranges = _rangeScales.TryGetValue(x, out long xScales) ? xScales : null;
        KernelVariant[] variants = NativeGemmF16
            ? Unless(NativeF16Variants(p.N, p.K, p.X.Shape[0], null, b => p.RowCount(b), LibraryGemmMinRows, rangeScales: ranges), notMultiTall)
            : [new KernelVariant(b => p.RowCount(b) >= LibraryGemmMinRows, LibraryKernel.Gemm(_opt.Target, "f16", p.N, p.K), _ => new Grid(1))];
        Emit(label + (NativeGemmF16 ? " native" : " library"), [p.Node], variants,
            [Pointer(x), Pointer(p.W), Pointer(p.Y), TokenArg(p.X)],
            [x, p.W], [p.Y]);
        if (!NativeGemmF16 && ranges is { } scales)
        {
            // Rows of a row-ranged copy that were scaled leave scaled outputs.
            Emit($"{label} f16 range", [p.Node],
                [new KernelVariant(b => p.RowCount(b) >= LibraryGemmMinRows,
                    Lib("data.slang", "f16_range_unscale", Macros(("N", p.N), ("Y_STRIDE", p.Macros["Y_STRIDE"]), ("X_RS", scales + "ull"))),
                    b => new Grid(Grid.Div(p.RowCount(b), 64)))],
                [Pointer(x), Pointer(p.Y), TokenArg(p.X)],
                [x, p.Y], [p.Y]);
        }
    }
}
