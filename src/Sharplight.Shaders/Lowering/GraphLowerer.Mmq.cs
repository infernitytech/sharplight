using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// True when the int8 tensor-core GEMM (<c>mmq.slang</c>, <see cref="DeviceFeatures.MatrixInt8"/>) serves prefill-sized
    /// batches of a matmul with weights of <paramref name="type"/> and reduction length <paramref name="k"/>: q8_1-quantized
    /// activations, int8 MMAs per 32-value block. Without the feature the f16 tensor-core GEMM runs instead. Formats in
    /// <see cref="MmqLargeOnlyFormats"/> use it by default only for matmuls with <paramref name="n"/> and
    /// <paramref name="k"/> of at least <see cref="MmqLargeDimension"/> (smaller grids lose to the f16 GEMM's split-K).
    /// </summary>
    private bool UseMmq(DataType type, long n, long k)
        => _opt.Has(DeviceFeatures.MatrixInt8) && _opt.Has(DeviceFeatures.SubgroupOps) && _opt.SubgroupSize == 32
           && k % 32 == 0 && MmqFormats.Contains(type) && (type == ShaderTypes.Q8_0R || type == DataType.IQ4_NL || k % 256 == 0)
           && (!MmqLargeOnlyFormats.Contains(type) || (n >= MmqLargeDimension && k >= MmqLargeDimension));

    /// <summary>
    /// Formats of <see cref="MmqFormats"/> that take the int8 GEMM only for large matmuls (see <see cref="UseMmq"/>): Q8_0 by
    /// default, none when <c>SHARPLIGHT_MMQ</c> names the formats. Process-wide; set before lowering.
    /// </summary>
    public static IReadOnlySet<DataType> MmqLargeOnlyFormats { get; set; } =
        DiagnosticSwitches.MmqFormats is null ? new HashSet<DataType> { ShaderTypes.Q8_0R } : new HashSet<DataType>();

    /// <summary>Smallest N and K at which the large-only formats take the int8 GEMM.</summary>
    private const long MmqLargeDimension = 4096;

    /// <summary>Weight formats the int8 GEMM can unpack (Q8_0 as its repacked form).</summary>
    public static IReadOnlySet<DataType> MmqSupportedFormats { get; } =
        new HashSet<DataType> { ShaderTypes.Q8_0R, DataType.Q4_K, DataType.Q5_K, DataType.Q6_K, DataType.IQ4_NL, DataType.IQ4_XS };

    /// <summary>
    /// Weight formats whose prefill matmuls use the int8 GEMM (a subset of <see cref="MmqSupportedFormats"/>). The default
    /// holds the formats where it beats the f16 tensor-core GEMM on the devices measured (see docs/benchmarks.md); the
    /// <c>SHARPLIGHT_MMQ</c> switch overrides it. Process-wide; set before lowering.
    /// </summary>
    public static IReadOnlySet<DataType> MmqFormats { get; set; } = DefaultMmqFormats();

    private static HashSet<DataType> DefaultMmqFormats() => DiagnosticSwitches.MmqFormats switch
    {
        null => [ShaderTypes.Q8_0R, DataType.IQ4_NL, DataType.IQ4_XS],
        "0" => [],
        "all" => [.. MmqSupportedFormats],
        var list => [.. list.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
            .Select(name => name == "q8_0" ? ShaderTypes.Q8_0R : DataType.Get(name)).Where(MmqSupportedFormats.Contains)],
    };

    /// <summary>Activation rows from which the int8 GEMM serves a matmul (smaller batches stay on the matvecs and skinny f16 tiles).</summary>
    private static long MmqMinRows => DiagnosticSwitches.MmqMinRows;

    /// <summary>
    /// K values staged per step of the int8 GEMM: the largest of 128, 64 and 32 dividing K (longer steps amortize the
    /// staging barriers). Q6_K stages a second copy of its codes and stays at 64 to fit the 48 KiB of shared memory.
    /// </summary>
    private static int MmqBlockK(DataType type, long k)
    {
        int preferred = DiagnosticSwitches.MmqBlockK ?? (type == DataType.Q6_K ? 64 : 128);
        for (int bk = preferred; bk > 32; bk /= 2)
        {
            if (k % bk == 0)
            {
                return bk;
            }
        }

        return 32;
    }

    /// <summary>
    /// Warp layout (warps M, warps N, fragments M, fragments N) of the int8 GEMM for a tile of <paramref name="tileRows"/>
    /// tokens: 128-token tiles use 2x4 subgroups with 4x2 fragments, 64-token tiles 2x4 with 2x2.
    /// </summary>
    private static int[] MmqLayout(int tileRows) => tileRows switch
    {
        64 => DiagnosticSwitches.MmqLayout64 ?? [2, 4, 2, 2],
        _ => DiagnosticSwitches.MmqLayout ?? [2, 4, 4, 2],
    };

    /// <summary>Token rows of the int8 GEMM tile for a batch of <paramref name="rows"/>.</summary>
    private static int MmqTileRowsFor(long rows) => rows <= 64 ? 64 : 128;

    private static int MmqColumns(int tileRows)
    {
        var layout = MmqLayout(tileRows);
        return layout[1] * layout[3] * 16;
    }

    private Dictionary<string, string> MmqMacros(DataType type, long n, long k, int tileRows, int split)
    {
        var layout = MmqLayout(tileRows);
        if (layout[0] * layout[2] * 16 != tileRows)
        {
            throw new NotSupportedException($"int8 GEMM layout {string.Join(',', layout)} does not give {tileRows}-token tiles.");
        }

        return Macros(("WFMT", WeightFormat(type)!.Value), ("K", k), ("N", n), ("Y_STRIDE", n), ("ROW_BYTES", type.ByteSize(k) + "ull"),
            ("MQ_WM", layout[0]), ("MQ_WN", layout[1]), ("MQ_FM", layout[2]), ("MQ_FN", layout[3]), ("MQ_BK", MmqBlockK(type, k)),
            ("SPLITK", split), ("MQ_GENERIC_LAYOUT", DiagnosticSwitches.MmqGenericLayout));
    }

    /// <summary>
    /// K split of an 8-bit GEMM (int8 or fp8) staging <paramref name="blockK"/> values per step for <paramref name="rows"/>
    /// activation rows (grids too small to fill the GPU).
    /// </summary>
    private static int QuantizedGemmSplit(long n, long k, long rows, int blockK)
    {
        int tileRows = MmqTileRowsFor(rows);
        return ChooseSplit(n, k, rows, tileRows, MmqColumns(tileRows), blockK);
    }

    /// <summary>
    /// Emits the activation quantization and the int8 GEMM launches (plus split-K reduction) of a planned matmul for
    /// batches of at least <see cref="MmqMinRows"/> rows. The caller keeps its f16 GEMM launches off that range.
    /// </summary>
    private void EmitMmq(MatMulPlan p) => EmitQuantizedGemm(p, "mmq", ActivationCode.Q8_1, MmqBlockK(p.W.Type, p.K),
        (tileRows, split) => Lib("mmq.slang", "gemm_mmq", MmqMacros(p.W.Type, p.N, p.K, tileRows, split)));

    /// <summary>
    /// Emits an 8-bit tensor-core GEMM (activations quantized to <paramref name="code"/>, 64- or 128-token tiles of the
    /// <see cref="MmqLayout"/> layouts, <paramref name="blockK"/> K values per staged step) with its split-K variants and
    /// reduction for batches of at least <see cref="MmqMinRows"/> rows. <paramref name="kernel"/> compiles the GEMM for a
    /// tile height and a K split.
    /// </summary>
    private void EmitQuantizedGemm(MatMulPlan p, string name, ActivationCode code, int blockK, Func<int, int, CompiledKernel> kernel)
    {
        long n = p.N;
        long k = p.K;
        long rowsMax = p.X.Shape[0] * p.RowsPerLead;
        if (rowsMax < MmqMinRows)
        {
            return;
        }

        string label = MatMulLabel(p);
        var quantized = QuantizedActivations(label, [p.Node], p.Node.Inputs[0], p.X, p.XStride, p.RowsPerLead, k, MmqMinRows, rowsMax, code);
        bool InRange(Bindings b) => p.RowCount(b) >= MmqMinRows;
        int Split(Bindings b) => QuantizedGemmSplit(n, k, p.RowCount(b), blockK);

        var direct = new List<KernelVariant>();
        foreach (int tileRows in new[] { 64, 128 })
        {
            int columns = MmqColumns(tileRows);
            direct.Add(new KernelVariant(b => InRange(b) && MmqTileRowsFor(p.RowCount(b)) == tileRows && Split(b) == 1,
                kernel(tileRows, 1),
                b => new Grid(Grid.Div(n, columns), Grid.Div(p.RowCount(b), tileRows))));
        }

        Emit($"{label} {name}", [p.Node], direct,
            [Pointer(quantized), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
            [quantized, p.W], [p.Y]);

        // Split-K: the same tiles summing K ranges into partials, then the reduction of matmul.slang.
        var tiledSplits = new SortedSet<(int TileRows, int Split)>();
        long scratchBytes = 0;
        for (long rows = MmqMinRows; rows <= rowsMax; rows = NextSplitProbe(rows, rowsMax))
        {
            int split = QuantizedGemmSplit(n, k, rows, blockK);
            if (split > 1)
            {
                tiledSplits.Add((MmqTileRowsFor(rows), split));
                scratchBytes = Math.Max(scratchBytes, split * rows * n * 4);
            }
        }

        if (tiledSplits.Count == 0)
        {
            return;
        }

        var scratch = SplitScratch(scratchBytes);
        var partial = new TensorRef(scratch, 0, DataType.F32, [scratchBytes / 4], [1], null);
        var partialVariants = tiledSplits.Select(ts => new KernelVariant(
            b => InRange(b) && MmqTileRowsFor(p.RowCount(b)) == ts.TileRows && Split(b) == ts.Split,
            kernel(ts.TileRows, ts.Split),
            b => new Grid(Grid.Div(n, MmqColumns(ts.TileRows)), Grid.Div(p.RowCount(b), ts.TileRows), (uint)ts.Split)));
        Emit($"{label} {name} split-k", [p.Node], [.. partialVariants],
            [Pointer(quantized), Pointer(p.W), new PointerArg(scratch, 0), TokenArg(p.X, p.RowsPerLead)],
            [quantized, p.W], [partial]);

        var reduceVariants = tiledSplits.Select(ts => ts.Split).Distinct().Select(split => new KernelVariant(
            b => InRange(b) && Split(b) == split,
            Lib("matmul.slang", "splitk_reduce", new Dictionary<string, string>(p.Macros) { ["SPLITK"] = split.ToString(CultureInfo.InvariantCulture) }),
            b => new Grid(Grid.Div(p.RowCount(b) * n, 256))));
        Emit($"matmul {p.Node.Output.Name} {name} split-k reduce", [p.Node], [.. reduceVariants],
            [new PointerArg(scratch, 0), Pointer(p.W), Pointer(p.Y), TokenArg(p.X, p.RowsPerLead)],
            [partial], [p.Y]);
    }
}
