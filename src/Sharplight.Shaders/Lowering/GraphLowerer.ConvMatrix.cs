using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Convolutions as implicit GEMMs on cooperative matrices (conv_cm.slang). Strict precision: every operand split into three
/// bf16 parts, six products (about 24 mantissa bits, f32 equivalent; <see cref="DeviceFeatures.MatrixBF16"/>). Reduced
/// precision: one f16 product (<see cref="DeviceFeatures.MatrixF16"/>), else two bf16 parts and three products (about 16
/// bits). Without those features the SIMT conv_gemm runs.
/// </summary>
public sealed partial class GraphLowerer
{
    /// <summary>Output channels per subgroup row of conv_cm (the tile has 1 or 2 such rows: WARPS_M).</summary>
    private const long ConvMatrixWarpRows = 64;
    private const long ConvMatrixTileK = 32;

    /// <summary>Operand parts (1: f16, 2: bf16 hi + lo, 3: bf16 hi + mid + lo) and pixel columns of the tile.</summary>
    private readonly record struct ConvMatrixForm(int Parts, long TileColumns)
    {
        /// <summary>Pixel fragments of 16 per subgroup (FRAG_N): three-part operands use a 64-column tile to fit shared memory.</summary>
        public int FragN => (int)(TileColumns / 64);
    }

    /// <summary>Smallest reduction length / output channels per group worth a tensor-core tile.</summary>
    private const long ConvMatrixMinK = 64;
    private const long ConvMatrixMinChannels = 32;

    /// <summary>
    /// Smallest output pixel count (all images) worth a tensor-core tile: below one 64-pixel tile the padding is cheaper than
    /// the SIMT conv (7x7 maps of deep layers, 49 pixels, still run 3-4x faster on the matrix units).
    /// </summary>
    private const long ConvMatrixMinColumns = 32;

    /// <summary>
    /// Shared memory of conv_cm in bytes: one packed operand tile pair per part and the epilogue staging of
    /// 4 * <paramref name="warpsM"/> subgroups.
    /// </summary>
    private static long ConvMatrixSharedBytes(int warpsM, ConvMatrixForm form)
        => form.Parts * (warpsM * ConvMatrixWarpRows + form.TileColumns) * (ConvMatrixTileK + 8) * 2 + warpsM * 4 * 256 * 4;

    /// <summary>Subgroup rows of the conv_cm tile: 2 (128 channels) when OG fills it and shared memory allows, else 1.</summary>
    private int ConvMatrixWarpsM(long OG, ConvMatrixForm form)
        => OG >= 2 * ConvMatrixWarpRows && ConvMatrixSharedBytes(2, form) <= _opt.MaxSharedMemory ? 2 : 1;

    /// <summary>
    /// Which conv_cm form applies (null: none), see the class remarks. <paramref name="columns"/> is the output map size,
    /// <paramref name="batchBound"/> the largest batch and <paramref name="groups"/> the conv groups.
    /// </summary>
    private ConvMatrixForm? ChooseConvMatrixForm(long K, long OG, long columns, int weightFormat, long batchBound = 1, long groups = 1)
    {
        if (_opt.SubgroupSize != 32 || weightFormat is not (0 or 1 or 2) || K < ConvMatrixMinK || OG < ConvMatrixMinChannels
            || DiagnosticSwitches.ConvNoMatrix)
        {
            return null;
        }

        bool bf16 = _opt.Has(DeviceFeatures.MatrixBF16) && _opt.Has(DeviceFeatures.BFloat16);
        bool f16 = _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16);
        // Reduced precision: 128-pixel tiles, or 64 when the 128-pixel grid would not fill the device (small maps at batch 1).
        // Those convs are bound by staging the im2col tile, not by the products, so fewer, wider tiles are slower even with a
        // sixth of the products of the strict form.
        long reducedColumns = (columns * Math.Max(1, batchBound) + 127) / 128 * ((OG + 127) / 128) * groups < TargetWorkgroups ? 64 : 128;
        ConvMatrixForm? form = _opt.Compile.AllowReducedPrecision
            ? f16 ? new ConvMatrixForm(1, reducedColumns) : bf16 ? new ConvMatrixForm(2, reducedColumns) : null
            : bf16 ? new ConvMatrixForm(3, 64) : null;
        return form is { } f && columns * Math.Max(1, batchBound) >= ConvMatrixMinColumns && ConvMatrixSharedBytes(1, f) <= _opt.MaxSharedMemory ? f : null;
    }

    /// <summary>K split for conv_cm: aims for <see cref="TargetWorkgroups"/>, at least 4 K steps per split.</summary>
    private static (int Split, long Kps) ChooseConvMatrixSplit(long N, long OG, long G, long HW, long K, long tileRows, long tileColumns)
    {
        long tiles = (N * HW + tileColumns - 1) / tileColumns * ((OG + tileRows - 1) / tileRows) * G;
        if (tiles == 0)
        {
            // Empty batch (a data-dependent count of 0, e.g. no detections): nothing to split.
            return (1, K);
        }

        long kTiles = (K + ConvMatrixTileK - 1) / ConvMatrixTileK;
        long split = Math.Min(Math.Min((TargetWorkgroups + tiles - 1) / tiles, kTiles / 4), 16);
        if (split < 2)
        {
            return (1, K);
        }

        long kps = (kTiles + split - 1) / split * ConvMatrixTileK;
        return ((int)((K + kps - 1) / kps), kps);
    }

    /// <summary>Lowers a (non-depthwise) conv through conv_cm when <see cref="ChooseConvMatrixForm"/> allows; false otherwise.</summary>
    private bool TryLowerConvMatrix(Node n, Node[] nodes, ConvEpilogueCode? epilogue, TensorRef x, TensorRef w, TensorRef? bias, TensorRef y,
        (string, object)[] common, string name)
    {
        long G = n.Attributes.Get("groups", 1L);
        long O = w.Shape[0];
        long OG = O / G;
        long HW = y.Shape[2] * y.Shape[3];
        long K = w.Shape[1] * w.Shape[2] * w.Shape[3];
        int format = WeightFormat(w.Type) ?? -1;
        long bound = x.Symbol is null ? x.Shape[0] : Bound(Dim.Sym(x.Symbol));
        if (ChooseConvMatrixForm(K, OG, HW, format, bound, G) is not { } form)
        {
            return false;
        }

        int warpsM = ConvMatrixWarpsM(OG, form);
        long tileRows = warpsM * ConvMatrixWarpRows;
        Func<long, (int Split, long Kps)> choose = batch => ChooseConvMatrixSplit(batch, OG, G, HW, K, tileRows, form.TileColumns);
        long workspace = 0;
        for (long v = x.Symbol is null ? bound : 1; v <= bound; v++)
        {
            var c = choose(v);
            if (c.Split > 1)
            {
                workspace = Math.Max(workspace, c.Split * v * O * HW);
            }
        }

        bool useWorkspace = workspace > 0;
        TensorRef? ws = useWorkspace ? Temp(DataType.F32, [workspace], null) : null;
        var biasPointer = bias ?? x;
        var macros = common.Where(c => c.Item1 != "HAS_BIAS").ToList();
        macros.Add(("HAS_BIAS", bias is not null && !useWorkspace));
        macros.Add(("PARTS", form.Parts));
        macros.Add(("FRAG_N", form.FragN));
        macros.Add(("WARPS_M", warpsM));
        macros.Add(("A_VEC", K % 8 == 0));
        var variants = ChooseVariants(x.Symbol, bound, choose, cfg =>
        {
            var kernel = ConvKernel("conv_cm.slang", "conv_cm", Macros([.. macros, ("SPLITK", cfg.Split), ("KPS", cfg.Kps)]), useWorkspace ? null : epilogue);
            return (kernel, batch => new Grid(Grid.Div(batch * HW, form.TileColumns), Grid.Div(OG, tileRows), (uint)(G * cfg.Split)));
        });
        string label = $"{name} {form.Parts switch { 3 => "bf16x6", 2 => "bf16x3", _ => "f16" }} cm";
        var dst = ws ?? y;
        var resArgs = epilogue?.Residual is { } resRef ? new KernelArg[] { Pointer(resRef) } : [];
        var resReads = epilogue?.Residual is { } resRead ? new[] { resRead } : [];
        Emit(useWorkspace ? label + " gemm" : label, nodes, variants,
            [Pointer(x), Pointer(w), Pointer(biasPointer), Pointer(dst), .. useWorkspace ? [] : resArgs, TokenArg(x)],
            [.. bias is null || useWorkspace ? [x, w] : new[] { x, w, bias }, .. useWorkspace ? [] : resReads], [dst]);
        if (!useWorkspace)
        {
            return true;
        }

        var reduceVariants = ChooseVariants(x.Symbol, bound, choose, cfg =>
        {
            var kernel = ConvKernel("conv.slang", "conv_splitk_reduce", Macros(("SPLITK", cfg.Split), ("O", O), ("HW", HW), ("HAS_BIAS", bias is not null)), epilogue);
            return (kernel, batch => new Grid(Grid.Div(batch * O * HW, 256)));
        });
        Emit(label + " reduce", nodes, reduceVariants,
            [Pointer(dst), Pointer(biasPointer), Pointer(y), .. resArgs, CountArg(x, O * HW)],
            [.. bias is null ? [dst] : new[] { dst, bias }, .. resReads], [y]);
        return true;
    }
}
