using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Activation rows from which the FP4 GEMM (<c>gemm_fp4_native.cu</c>) serves matmuls of NVFP4 weights: right above the integer-dot
    /// matvec's 8 rows. <c>SHARPLIGHT_NATIVE_FP4_MIN_ROWS</c> overrides it.
    /// </summary>
    private static long NativeFp4MinRows => DiagnosticSwitches.NativeFp4MinRows ?? 9;

    /// <summary>Dynamic shared memory the FP4 GEMM's largest tile needs (64 x 128 rows of 160 bytes, three stages).</summary>
    private const int NativeFp4SharedBytes = 3 * (64 + 128) * 160;

    /// <summary>Bytes of one activation row of the FP4 GEMM (<see cref="ActivationCode.Fp4"/>): 160 per 256 values and 32 for the row's scale.</summary>
    private static long Fp4ActivationRowBytes(long k) => k / 256 * 160 + 32;

    /// <summary>
    /// True when matmuls of <paramref name="type"/> weights [<paramref name="n"/>, <paramref name="k"/>] reading activation rows
    /// <paramref name="xStride"/> floats apart and writing rows <paramref name="yStride"/> apart run on the FP4 GEMM from
    /// <see cref="NativeFp4MinRows"/> rows: the device has the block-scaled E2M1 MMA (<see cref="DeviceFeatures.BlockScaledFp4"/>) and
    /// takes CUDA C++, the weights are NVFP4 blocks (with or without the second-level scale), K is a multiple of 256, and the node may
    /// quantize its activations. <c>SHARPLIGHT_NATIVE_FP4=0</c> keeps such matmuls on the other paths.
    /// </summary>
    private bool UseNativeFp4Gemm(Node node, DataType type, long n, long k, long xStride, long yStride)
        => DiagnosticSwitches.NativeFp4 && _opt.Has(DeviceFeatures.BlockScaledFp4) && _compiler.CompilesCuda(_opt.Target)
           && _opt.MaxSharedMemory >= NativeFp4SharedBytes && (type == DataType.NVFP4 || type == DataType.NVFP4_GS)
           && k % 256 == 0 && k <= 65536 && xStride % 4 == 0 && yStride % 2 == 0 && n >= 8 && QuantizedActivationsAllowed(node);

    /// <summary>True when <see cref="UseNativeFp4Gemm"/> holds for the planned matmul <paramref name="p"/>.</summary>
    private bool UseNativeFp4Gemm(MatMulPlan p)
        => p.Macros.TryGetValue("Y_STRIDE", out var ys) && UseNativeFp4Gemm(p.Node, p.W.Type, p.N, p.K, p.XStride, long.Parse(ys, CultureInfo.InvariantCulture));

    /// <summary>True when matmuls of <paramref name="type"/> weights leave the rows from <see cref="NativeFp4MinRows"/> to the FP4 GEMM where it applies.</summary>
    private bool UsesNativeFp4(DataType type)
        => DiagnosticSwitches.NativeFp4 && _opt.Has(DeviceFeatures.BlockScaledFp4) && _compiler.CompilesCuda(_opt.Target)
           && (type == DataType.NVFP4 || type == DataType.NVFP4_GS);

    /// <summary>The <c>quantize_fp4</c> kernel (<c>quantize_fp4.cu</c>) for rows of <paramref name="k"/> values <paramref name="xStride"/> floats apart.</summary>
    /// <remarks>
    /// A file of its own: the CUDA translator gives every kernel of a source that issues <c>griddepcontrol.wait</c> itself (the GEMM,
    /// after its weight copies) no wait at its entry, and this kernel reads the previous launch's rows from its first instruction.
    /// </remarks>
    private CompiledKernel QuantizeFp4Kernel(long k, long xStride)
        => _compiler.CompileCuda("quantize_fp4.cu", "", "quantize_fp4", 128, 0, _opt.Target, Macros(("K", k), ("X_STRIDE", xStride)), _opt.Cache);

    /// <summary>Emits the FP4 activation quantization and the FP4 GEMM launches of planned matmul <paramref name="p"/> for batches from <see cref="NativeFp4MinRows"/> rows.</summary>
    private void EmitNativeFp4Gemm(MatMulPlan p)
    {
        long rowsMax = p.X.Shape[0] * p.RowsPerLead;
        long minRows = NativeFp4MinRows;
        if (rowsMax < minRows)
        {
            return;
        }

        string label = MatMulLabel(p);
        var quantized = QuantizedActivations(label, [p.Node], p.Node.Inputs[0], p.X, p.XStride, p.RowsPerLead, p.K, minRows, rowsMax, ActivationCode.Fp4);
        long yStride = long.Parse(p.Macros["Y_STRIDE"], CultureInfo.InvariantCulture);
        Emit($"{label} fp4", [p.Node], NativeFp4Variants(p.W.Type, p.N, p.K, yStride, null, minRows, rowsMax, p.RowCount),
            [Pointer(quantized), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
            [quantized, p.W], [p.Y]);
    }

    /// <summary>
    /// Emits the dual FP4 GEMM of a matmul pair sharing x (gate / up) with the elementwise group as its store (<paramref name="cudaEpilogue"/>,
    /// <see cref="DualEpilogueCudaSource"/>) for batches from <see cref="NativeFp4MinRows"/> rows, and returns the row counts it serves
    /// (none when it does not apply).
    /// </summary>
    private Func<long, bool> EmitDualFp4Gemm(string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef weightA,
        TensorRef weightB, TensorRef y, long n, long k, string? cudaEpilogue)
    {
        long rowsMax = x.Shape[0] * rowsPerLead;
        long minRows = NativeFp4MinRows;
        if (cudaEpilogue is null || weightB.Type != weightA.Type || !UseNativeFp4Gemm(nodes[0], weightA.Type, n, k, xStride, n) || rowsMax < minRows)
        {
            return _ => false;
        }

        var quantized = QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, minRows, rowsMax, ActivationCode.Fp4);
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        Emit($"{label} fp4", nodes, NativeFp4Variants(weightA.Type, n, k, n, cudaEpilogue, minRows, rowsMax, rows),
            [Pointer(quantized), Pointer(weightA), Pointer(weightB), Pointer(y), TokenArg(x, rowsPerLead)],
            [quantized, weightA, weightB], [y]);
        return r => r >= minRows;
    }

    /// <summary>
    /// The tile shapes of the FP4 GEMM by batch: (token rows, weight rows, warps along tokens, warps along weight rows, stages, blocks
    /// per SM). Up to 64 rows (decode batches) one token tile and narrow weight tiles, so the weights stream from many blocks; above, 64-
    /// and 128-row token tiles over 128 weight rows. RTX PRO 4500, [4096 → 12288] NVFP4_GS after an L2 flush: 16 rows 43 µs (733 GB/s),
    /// 64 rows 49 µs, 256 rows 100 µs, 2,048 rows 497 µs (415 TFLOPS).
    /// </summary>
    private static (int Bm, int Bn, int Wm, int Wn, int Stages, int MinBlocks) NativeFp4Tile(long rows, bool dual) => rows switch
    {
        <= 16 => (16, dual ? 64 : 32, 1, 4, 4, dual ? 2 : 3),
        <= 32 => (32, 64, 1, 4, 4, 2),
        <= 64 => (64, 64, 2, 4, 4, 1),
        <= 256 => (64, 128, 2, 4, 3, 1),
        _ => (128, 128, 2, 4, 2, 1),
    };

    /// <summary>Launch variants of the FP4 GEMM for batches of <paramref name="minRows"/> to <paramref name="maxRows"/> rows, one per tile shape.</summary>
    private KernelVariant[] NativeFp4Variants(DataType type, long n, long k, long yStride, string? dualEpilogue, long minRows, long maxRows,
        Func<Bindings, long> rows)
    {
        bool dual = dualEpilogue is not null;
        long[] bounds = [16, 32, 64, 256, long.MaxValue];
        var variants = new List<KernelVariant>();
        long low = minRows;
        foreach (long high in bounds)
        {
            if (low > maxRows)
            {
                break;
            }

            if (high < low)
            {
                continue;
            }

            var tile = NativeFp4Tile(low, dual);
            var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", yStride), ("F4_BM", tile.Bm), ("F4_BN", tile.Bn), ("F4_WM", tile.Wm), ("F4_WN", tile.Wn),
                ("F4_STAGES", tile.Stages), ("F4_MINB", tile.MinBlocks), ("F4_GS", type == DataType.NVFP4_GS ? 1 : 0), ("DUAL", dual ? 1 : 0));
            var kernel = _compiler.CompileCuda("gemm_fp4_native.cu", dualEpilogue?.Replace("F8N_EPI2", "F4N_EPI2", StringComparison.Ordinal) ?? "",
                "gemm_fp4_native", tile.Wm * tile.Wn * 32, tile.Stages * (tile.Bm + tile.Bn) * 160, _opt.Target, macros, _opt.Cache);
            long from = low, to = high;
            int outputs = dual ? tile.Bn / 2 : tile.Bn;
            variants.Add(new KernelVariant(b => rows(b) >= from && rows(b) <= to, kernel,
                b => new Grid(Grid.Div(rows(b), tile.Bm), Grid.Div(n, outputs))));
            low = high + 1;
        }

        return [.. variants];
    }

    /// <summary>
    /// True when the expert product <paramref name="node"/> (a MatMulId over [E, <paramref name="n"/>, <paramref name="k"/>] expert stacks
    /// of <paramref name="type"/>, activation rows <paramref name="xStride"/> floats apart) runs on the grouped form of the FP4 GEMM for
    /// every batch above the per-item matvecs: the conditions of <see cref="UseNativeFp4Gemm"/>.
    /// </summary>
    private bool UseNativeFp4Moe(Node node, DataType type, long n, long k, long xStride)
        => UseNativeFp4Gemm(node, type, n, k, xStride, n) && DiagnosticSwitches.NativeFp4Moe;

    /// <summary>The routing tile heights of the grouped FP4 GEMM.</summary>
    private static readonly int[] NativeFp4MoeTiles = [16, 32, 64];

    /// <summary>
    /// Routing tile rows of the grouped FP4 GEMM for <paramref name="items"/> routed items over <paramref name="experts"/> experts: 16-row
    /// tiles while experts see a few items on average (batched decode: each tile streams its expert's weights once), 32 and 64 above.
    /// </summary>
    private static int NativeFp4MoeTileRows(long items, long experts)
    {
        long perExpert = items / Math.Max(1, experts);
        return perExpert <= 12 ? 16 : perExpert <= 48 ? 32 : 64;
    }

    /// <summary>
    /// Emits the FP4 quantization of an expert product's activations (per token, or per (token, slot) item when x is per slot) and the
    /// grouped FP4 GEMM over the routing tiles of <paramref name="scratch"/> (moe_route) for batches above <paramref name="matvecMaxItems"/>
    /// items, storing the product per item into <paramref name="y"/>.
    /// </summary>
    /// <remarks>
    /// With <paramref name="dual"/> (the second expert stack of a gate / up pair and the CUDA form of the group, <see cref="DualEpilogueCudaSource"/>)
    /// one launch multiplies both stacks and stores the group into <paramref name="y"/>.
    /// </remarks>
    private void EmitNativeFp4Moe(string label, Node[] nodes, Node node, TensorRef x, long xStride, long rowsPerLead, bool perSlot, TensorRef w, TensorRef ids, TensorRef y,
        long maxTiles, long matvecMaxItems, Func<Bindings, long> items, Func<Bindings, int> tile, Func<Bindings, uint> tiles, TensorRef scratch, KernelArg scratchArg,
        (TensorRef W2, string Epilogue)? dual = null)
    {
        long n = w.Shape[1];
        long k = w.Shape[2];
        long slots = ids.Shape[1];
        long rowsMax = x.Shape[0] * rowsPerLead;
        long minRows = Math.Min(rowsMax, perSlot ? matvecMaxItems + 1 : matvecMaxItems / slots + 1);
        var quantized = QuantizedActivations(label, nodes, node.Inputs[0], x, xStride, rowsPerLead, k, minRows, rowsMax, ActivationCode.Fp4);
        var variants = new List<KernelVariant>();
        bool isDual = dual is not null;
        foreach (int bm in NativeFp4MoeTiles)
        {
            var shape = NativeFp4Tile(bm, isDual);
            var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("F4_BM", shape.Bm), ("F4_BN", shape.Bn), ("F4_WM", shape.Wm), ("F4_WN", shape.Wn),
                ("F4_STAGES", shape.Stages), ("F4_MINB", shape.MinBlocks), ("F4_GS", w.Type == DataType.NVFP4_GS ? 1 : 0), ("DUAL", isDual ? 1 : 0), ("F4_MOE", 1),
                ("EXPERT_BYTES", w.Type.ByteSize(k) * n + "ull"), ("S", slots), ("MAXTILES", maxTiles), ("X_TOKEN", perSlot ? 0 : 1));
            var kernel = _compiler.CompileCuda("gemm_fp4_native.cu", dual?.Epilogue.Replace("F8N_EPI2", "F4N_EPI2", StringComparison.Ordinal) ?? "", "gemm_fp4_native",
                shape.Wm * shape.Wn * 32, shape.Stages * (shape.Bm + shape.Bn) * 160, _opt.Target, macros, _opt.Cache);
            int columns = isDual ? shape.Bn / 2 : shape.Bn;
            variants.Add(new KernelVariant(b => items(b) > matvecMaxItems && tile(b) == bm, kernel, b => new Grid(Grid.Div(n, columns), tiles(b))));
        }

        KernelArg[] args = dual is { } d
            ? [Pointer(quantized), Pointer(w), Pointer(d.W2), scratchArg, Pointer(y), TokenArg(ids)]
            : [Pointer(quantized), Pointer(w), scratchArg, Pointer(y), TokenArg(ids)];
        TensorRef[] reads = dual is { } r ? [quantized, w, r.W2, scratch] : [quantized, w, scratch];
        Emit($"{label} fp4", nodes, [.. variants], args, reads, [y]);
    }
}
