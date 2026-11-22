using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Threads per block of <c>gemm_fp8_native.cu</c> (2 x 4 warps).</summary>
    private const int NativeFp8Threads = 256;

    /// <summary>Token rows and weight rows of a <c>gemm_fp8_native.cu</c> tile.</summary>
    private const int NativeFp8Tile = 128;

    /// <summary>Dynamic shared memory of <c>gemm_fp8_native.cu</c>: three stages of 128 + 128 rows of 128 codes and one scale each.</summary>
    private const int NativeFp8SharedBytes = 3 * (256 * 128 + 256 * 4);

    /// <summary>
    /// Fewest activation rows a matmul runs on a hand-written provider GEMM with (<c>SHARPLIGHT_NATIVE_GEMM_MIN_ROWS</c>, default
    /// 256): below that the generated GEMMs' 64-row tiles and K splits fill the GPU better.
    /// </summary>
    private static long NativeGemmMinRows => DiagnosticSwitches.NativeGemmMinRows ?? 65;

    /// <summary>
    /// True when matmuls of block-scaled E4M3 weights [<paramref name="n"/>, <paramref name="k"/>] run on the hand-written FP8 GEMM
    /// (<see cref="DeviceFeatures.NativeGemmFp8"/>, <c>gemm_fp8_native.cu</c>) from <see cref="NativeGemmMinRows"/> rows: weights in
    /// the 128-row groups of <see cref="WeightRepack.Fp8B128Grouped"/> (see <see cref="NativeFp8GroupedWeight"/>), on a target that
    /// takes CUDA C++.
    /// </summary>
    private bool UseNativeFp8Gemm(DataType type, long n, long k)
        => NativeGemmFp8 && _opt.MaxSharedMemory >= NativeFp8SharedBytes && type == ShaderTypes.F8E4M3_B128G && k % 512 == 0 && n % WeightRepacker.Fp8GroupRows == 0;

    /// <summary>True when the device runs the hand-written FP8 GEMM (feature and a CUDA C++ target).</summary>
    private bool NativeGemmFp8 => _opt.Has(DeviceFeatures.NativeGemmFp8) && _compiler.CompilesCuda(_opt.Target);

    /// <summary>
    /// True when a block-scaled E4M3 weight is stored in the 128-row groups of <see cref="WeightRepack.Fp8B128Grouped"/> for the
    /// hand-written FP8 GEMM: a [N, K] matrix (N % 128 == 0, K % 512 == 0) that only matmuls read as their weight. Every generated
    /// kernel reading FP8 weights decodes that layout too (<c>FP8_GROUPED</c>, <c>F8_GROUPED</c>), so the smaller batches and
    /// decode keep their kernels. <c>SHARPLIGHT_REPACK_FP8=0</c> keeps the stored block layout.
    /// </summary>
    private bool NativeFp8GroupedWeight(Value weight)
    {
        var w = weight.Weight!;
        if (!(DiagnosticSwitches.Fp8Grouped ?? NativeGemmFp8) || DiagnosticSwitches.NoWeightRepack || !DiagnosticSwitches.Fp8Repack || w.Type != DataType.F8E4M3_B128
            || w.Shape.Rank != 2 || w.Shape.Dims.Any(d => d.IsSymbolic) || w.Shape[0].Fixed % WeightRepacker.Fp8GroupRows != 0
            || w.Shape[1].Fixed % 512 != 0)
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
    /// Row capacity of the dense copy of the E4M3 activation scales (<c>[K / 128][xcap]</c> f32 after <c>xcap</c> rows, see
    /// <c>gemm_fp8_native.cu</c>) that the writers of activation rows for <paramref name="rows"/> rows add when the device runs the
    /// hand-written FP8 GEMM, or 0 (no copy).
    /// </summary>
    private long DenseScaleCapacity(long rows) => NativeGemmFp8 ? (rows + 127) / 128 * 128 : 0;

    /// <summary>
    /// Rows of activation-row tensors of <paramref name="rows"/> rows of [K codes][K / 128 scales] (<paramref name="rowBytes"/>
    /// bytes) with room for the dense scale copy after <see cref="DenseScaleCapacity"/> rows.
    /// </summary>
    private long Fp8ActivationRows(long rows, long rowBytes)
    {
        long cap = DenseScaleCapacity(rows);
        // The copy: cap * K / 32 bytes, less than cap / 32 rows of K + K / 32 bytes.
        return cap == 0 ? rows : cap + (cap * (rowBytes / 33) + rowBytes - 1) / rowBytes;
    }

    /// <summary>Row capacities of the dense scale copies of E4M3 activation rows, by buffer and offset (see <see cref="Fp8ActivationRows"/>).</summary>
    private readonly Dictionary<(BufferDecl Buffer, long Offset), long> _denseScaleCaps = [];

    /// <summary>Records that E4M3 activation rows <paramref name="quantized"/> for <paramref name="rows"/> rows carry the dense scale copy.</summary>
    private void NoteDenseScales(TensorRef quantized, long rows)
    {
        if (DenseScaleCapacity(rows) is > 0 and var cap)
        {
            _denseScaleCaps[(quantized.Buffer, quantized.ByteOffset)] = cap;
        }
    }

    /// <summary>The dense scale copy's row capacity of E4M3 activation rows <paramref name="quantized"/>, or 0 when they have none.</summary>
    private long DenseScaleCapacityOf(TensorRef quantized) => _denseScaleCaps.GetValueOrDefault((quantized.Buffer, quantized.ByteOffset));

    /// <summary>
    /// Relative time of a block of the hand-written FP8 GEMM with 128 token rows against one with 64, fitted to the tile choice that
    /// was faster for the Qwen3-8B shapes at 80-384 rows on the RTX PRO 4500 Blackwell (64-row tiles faster at 160, 192 and 320
    /// rows, 128-row tiles at 256 and 384: a ratio between 1.6 and 1.875).
    /// </summary>
    private const double NativeFp8TallBlockCost = 1.75;

    /// <summary>
    /// <see cref="NativeFp8TallBlockCost"/> when the 128-row tiles run on <c>gemm_fp8_tma.cu</c> (64-row tiles keep
    /// <c>gemm_fp8_native.cu</c>): fitted to Qwen3-8B-FP8 prefill (8192 rows in 1024-row passes: 9,971 tok/s at 1.75, 10,515 at 1.3,
    /// 10,486 at 1.0; 512 and 2048 rows within 1 %).
    /// </summary>
    private const double NativeFp8TmaTallBlockCost = 1.3;

    /// <summary>
    /// Token rows of the hand-written FP8 GEMM's tile for a batch (64 or 128): the one with the fewest estimated block waves times
    /// block time on the GPU's SMs (one block per SM: both use the 99 KB of shared memory).
    /// </summary>
    private int NativeFp8TileRows(long rows, long n, bool dual)
    {
        long sms = Math.Max(1, _opt.ComputeUnits);
        long columns = Grid.Div(n, dual ? NativeFp8Tile / 2 : NativeFp8Tile);
        double Cost(int tileRows, double blockCost) => (Grid.Div(rows, tileRows) * columns + sms - 1) / sms * blockCost;
        return Cost(64, 1.0) <= Cost(128, DiagnosticSwitches.NativeFp8TallCost ?? (NativeFp8TensorCopies ? NativeFp8TmaTallBlockCost : NativeFp8TallBlockCost)) ? 64 : 128;
    }

    /// <summary>
    /// Launch variants of the hand-written FP8 GEMM for [<paramref name="n"/>, <paramref name="k"/>] weights (single, or dual with
    /// <paramref name="dualEpilogue"/>) reading activation rows whose dense scale copy has <paramref name="xcap"/> rows, for batches of
    /// <paramref name="minRows"/> to <paramref name="maxRows"/> rows: one per token tile height (<see cref="NativeFp8TileRows"/>).
    /// </summary>
    private KernelVariant[] NativeFp8Variants(long n, long k, string? dualEpilogue, long xcap, long minRows, long maxRows, Func<Bindings, long> rows)
    {
        bool dual = dualEpilogue is not null;
        int decodeColumns = NativeFp8DecodeColumns;
        bool decodeForm = decodeColumns != NativeFp8Tile && minRows <= 64;
        bool Decode(long r) => decodeForm && r <= 64;
        var used = new SortedSet<int>();
        for (long r = minRows; r <= maxRows; r += 64)
        {
            used.Add(NativeFp8TileRows(r, n, dual));
        }

        used.Add(NativeFp8TileRows(maxRows, n, dual));
        var variants = used.Select(tileRows =>
        {
            bool Serves(Bindings b) => rows(b) >= minRows && !Decode(rows(b)) && NativeFp8TileRows(rows(b), n, dual) == tileRows;
            if (tileRows == NativeFp8Tile && NativeFp8TensorCopies)
            {
                return NativeFp8TmaVariant(n, k, dualEpilogue, xcap, Serves, rows);
            }

            return NativeFp8Variant(n, k, dualEpilogue, xcap, tileRows, NativeFp8Tile, Serves, rows);
        }).ToList();
        if (decodeForm)
        {
            variants.Add(NativeFp8Variant(n, k, dualEpilogue, xcap, 64, decodeColumns, b => rows(b) >= minRows && Decode(rows(b)) && rows(b) > NativeFp8ShortTileMaxRows, rows));
            bool streaming = UseFp8DecodeKernel(n, k, dual);
            bool Streams(long r) => streaming && r <= Fp8DecodeKernelMaxRows;
            variants.Add(NativeFp8Variant(n, k, dualEpilogue, xcap, 32, decodeColumns,
                b => rows(b) >= minRows && Decode(rows(b)) && rows(b) <= NativeFp8ShortTileMaxRows && !Streams(rows(b)), rows));
            if (streaming)
            {
                variants.Add(Fp8DecodeVariant(n, k, dualEpilogue, b => rows(b) >= minRows && Decode(rows(b)) && Streams(rows(b))));
            }
        }

        return [.. variants];
    }

    /// <summary>
    /// Weight rows per tile of the hand-written FP8 GEMM's decode form: batches of at most 64 rows (one token tile) on 64-row weight
    /// tiles, twice the blocks of the 128-row tiles for the same weights, so the last of the 2-5 block waves is shorter. RTX PRO 4500,
    /// Qwen3-8B-FP8 at 64 sequences (GPU µs per layer against the generated fp8 GEMM with split-K): gate / up 147.5 → 134.6, down
    /// 75.3 + 2.8 → 69.4, o 30.1 + 2.8 → 24.4; step 17.06 → 16.47 ms. 128-row weight tiles from 17 rows: 17.10 ms.
    /// <c>SHARPLIGHT_NATIVE_FP8_DECODE_BN=128</c> turns the decode form off.
    /// </summary>
    private static int NativeFp8DecodeColumns => DiagnosticSwitches.NativeFp8DecodeColumns ?? 64;

    /// <summary>
    /// Rows from which the hand-written FP8 GEMM's decode form (<see cref="NativeFp8DecodeColumns"/>) serves: below, the skinny GEMM.
    /// RTX PRO 4500, Qwen3-8B-FP8 decode steps (GPU ms): with 64-row token tiles from 32 rows (§63-67) 24 sequences 14.22 → 14.61
    /// (from 17 rows), 32: 14.90 → 14.71; with the 32-row token tiles (<see cref="NativeFp8ShortTileMaxRows"/>) from 9 rows, against
    /// the skinny GEMM and the generated fp8 GEMM's K split: 12 sequences 13.06-13.18 → 12.44-12.52, 16: 13.53-13.69 → 12.91-12.95,
    /// 20: 13.98 → 12.97, 24: 14.15-14.18 → 13.17-13.18 (with the q / k / v multi-output form), 32: 14.74-14.78 → 13.77-13.78.
    /// </summary>
    private const long NativeFp8DecodeMinRows = 9;

    /// <summary>
    /// Most rows the hand-written FP8 GEMM's decode form runs on 32-row token tiles (three stages, 38 KB: two blocks per SM) instead of
    /// 64-row tiles. bench/s57_gemm at 16 / 32 rows (µs, after an L2 flush, 64-row → 32-row token tiles): [4096 → 4096] 41 / 35 → 29 / 29,
    /// gate / up [4096 → 2 x 12288] 170 / 162 → 141 / 143, down [12288 → 4096] 92 / 84 → 84 / 82.
    /// </summary>
    private const long NativeFp8ShortTileMaxRows = 32;

    /// <summary>
    /// Rows from which the hand-written FP8 GEMM serves a matmul: from <see cref="NativeFp8DecodeMinRows"/> on its decode form when
    /// <paramref name="decode"/> (not a member of a multi-output q / k / v GEMM, which is faster at decode sizes: 43.4 against 51.8 µs per
    /// Qwen3-8B layer), else from <see cref="NativeGemmMinRows"/>.
    /// </summary>
    private static long NativeFp8MinRows(bool decode)
        => decode && NativeFp8DecodeColumns != NativeFp8Tile && DiagnosticSwitches.NativeGemmMinRows is null ? NativeFp8DecodeMinRows : NativeGemmMinRows;

    /// <summary>
    /// A launch variant of <c>gemm_fp8_native.cu</c> with tiles of <paramref name="tileRows"/> tokens and <paramref name="columns"/> weight
    /// rows, with the <paramref name="extra"/> macros (the multi-output form's).
    /// </summary>
    private KernelVariant NativeFp8Variant(long n, long k, string? dualEpilogue, long xcap, int tileRows, int columns, Func<Bindings, bool> serves,
        Func<Bindings, long> rows, IReadOnlyDictionary<string, string>? extra = null)
    {
        bool dual = dualEpilogue is not null;
        var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("XCAP", xcap), ("F8N_BM", tileRows), ("DUAL", dual ? 1 : 0));
        foreach (var (name, value) in extra ?? new Dictionary<string, string>())
        {
            macros[name] = value;
        }

        if (columns != NativeFp8Tile)
        {
            macros["F8N_BN"] = columns.ToString(CultureInfo.InvariantCulture);
            if ((DiagnosticSwitches.Fp8L2Prefetch ?? NativeFp8DecodeL2Blocks) is > 0 and var l2)
            {
                macros["F8N_L2"] = l2.ToString(CultureInfo.InvariantCulture);
            }
        }

        int stages = tileRows == 64 ? 4 : 3;
        var kernel = _compiler.CompileCuda("gemm_fp8_native.cu", dualEpilogue ?? "", "gemm_fp8_native", NativeFp8Threads, stages * (tileRows + columns) * 132,
            _opt.Target, macros, _opt.Cache);
        return new KernelVariant(serves, kernel, b => new Grid(Grid.Div(rows(b), tileRows), Grid.Div(n, dual ? columns / 2 : columns)));
    }

    /// <summary>
    /// K blocks of its weight rows each block of the decode form asks L2 for before it waits for the launch that writes x (when it runs
    /// as that launch's programmatic dependent): 0 for none.
    /// </summary>
    private const int NativeFp8DecodeL2Blocks = 0;

    /// <summary>Most rows the register-streaming FP8 decode GEMM (<c>gemm_fp8_decode.cu</c>) serves: two 8-token fragments.</summary>
    private const long Fp8DecodeKernelMaxRows = 16;

    /// <summary>Warps the register-streaming FP8 decode GEMM aims for: enough weight loads in flight on every SM.</summary>
    private const long Fp8DecodeKernelWarps = 512;

    /// <summary>
    /// True when batches of at most <see cref="Fp8DecodeKernelMaxRows"/> rows of the hand-written FP8 GEMM's decode form run on
    /// <c>gemm_fp8_decode.cu</c>: weight rows streamed into registers by every warp, no shared-memory stages, K split across the warps
    /// of a block. Single matmuls by default (Qwen3-8B-FP8 at 16 rows, µs per layer: down [12288 → 4096] 69.8 → 67.4, o 24.3 → 24.3);
    /// the dual gate / up pair keeps <c>gemm_fp8_native.cu</c> (131.5 → 141.8 on this kernel). <c>SHARPLIGHT_FP8_DECODE_KERNEL=1</c> /
    /// <c>=0</c> forces it on (dual too) / off (benchmarks.md §84).
    /// </summary>
    private static bool UseFp8DecodeKernel(long n, long k, bool dual) => (DiagnosticSwitches.Fp8DecodeKernel ?? !dual) && n % 128 == 0 && k % 512 == 0;

    /// <summary>
    /// The launch variant of <c>gemm_fp8_decode.cu</c>: warps of 32 weight rows (dual: 16 outputs of both matrices), K split over the
    /// fewest warps per row tile (a divisor of the K blocks, at most 8) that gives <see cref="Fp8DecodeKernelWarps"/> warps in all, or the most.
    /// </summary>
    private KernelVariant Fp8DecodeVariant(long n, long k, string? dualEpilogue, Func<Bindings, bool> serves)
    {
        bool dual = dualEpilogue is not null;
        long rowTiles = n / (dual ? 16 : 32);
        long blocksK = k / 128;
        int[] splits = [.. new[] { 1, 2, 3, 4, 6, 8 }.Where(s => blocksK % s == 0)];
        int split = splits.FirstOrDefault(s => rowTiles * s >= Fp8DecodeKernelWarps, splits[^1]);
        int rowTilesPerBlock = split == 1 ? 4 : 1;
        while (rowTiles % rowTilesPerBlock != 0)
        {
            rowTilesPerBlock /= 2;
        }

        var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("DUAL", dual ? 1 : 0), ("F8D_NT", 2), ("F8D_RF", 2), ("F8D_KS", split),
            ("F8D_RT", rowTilesPerBlock));
        if (DiagnosticSwitches.Fp8L2Prefetch is > 0 and var l2)
        {
            macros["F8D_L2"] = l2.ToString(CultureInfo.InvariantCulture);
        }

        var kernel = _compiler.CompileCuda("gemm_fp8_decode.cu", dualEpilogue?.Replace("F8N_EPI2", "F8D_EPI2", StringComparison.Ordinal) ?? "", "gemm_fp8_decode",
            32 * split * rowTilesPerBlock, 0, _opt.Target, macros, _opt.Cache);
        uint tiles = (uint)(rowTiles / rowTilesPerBlock);
        return new KernelVariant(serves, kernel, _ => new Grid(tiles));
    }

    /// <summary>Threads per block of <c>gemm_fp8_tma.cu</c>: 2 x 4 computing warps and the copy warp.</summary>
    private const int NativeFp8TmaThreads = 288;

    /// <summary>Dynamic shared memory of <c>gemm_fp8_tma.cu</c>: three stages of 256 rows of 128 codes, two slots of 256 scales, eight barriers.</summary>
    private const int NativeFp8TmaSharedBytes = 3 * 256 * 128 + 2 * 256 * 4 + 64;

    /// <summary>
    /// True when the hand-written FP8 GEMM's 128-row tiles run on its warp-specialized form with tensor copies (<c>gemm_fp8_tma.cu</c>,
    /// <see cref="DeviceFeatures.TensorCopies"/>): persistent blocks, a copy warp and full / empty barriers instead of per-thread
    /// asynchronous copies and a block barrier per K block (Qwen3-8B-FP8 prefill +8-9 % at 512-8192 rows on the RTX PRO 4500
    /// Blackwell, benchmarks.md section 57). Disabling <c>tensor_copies</c> or <c>SHARPLIGHT_FP8_TMA=0</c> keeps
    /// <c>gemm_fp8_native.cu</c>.
    /// </summary>
    private bool NativeFp8TensorCopies => DiagnosticSwitches.NativeFp8Tma && _opt.Has(DeviceFeatures.TensorCopies) && _opt.MaxSharedMemory >= NativeFp8TmaSharedBytes;

    /// <summary>Rows from which the dual FP8 GEMM stores E4M3 rows (<see cref="Fp8StoreMinRows"/>) unless <c>SHARPLIGHT_FP8_QSTORE_MIN_ROWS</c> says otherwise.</summary>
    private const long DefaultFp8StoreMinRows = 256;

    /// <summary>
    /// The smallest step from which the dual FP8 GEMM of a gate / up pair writing <paramref name="output"/> (f32 rows <paramref name="y"/>
    /// of <paramref name="n"/> values) stores E4M3 activation rows for the FP8 GEMMs reading it instead of f32 rows that a separate
    /// <c>quantize_fp8</c> launch turns into the same codes and scales (<c>QOUT</c> of <c>gemm_fp8_tma.cu</c>: bit for bit the same, so
    /// the next GEMM's inputs do not change; the f32 round trip and the launch go). long.MaxValue when that does not apply: without a
    /// CUDA epilogue, when a reader is not an FP8 GEMM or the value is a segment output, or with <c>SHARPLIGHT_FP8_QSTORE_MIN_ROWS=0</c>.
    /// The caller checks that the 128-row tiles run on tensor copies.
    /// </summary>
    private long Fp8StoreMinRows(Value? output, TensorRef y, long n, string? cudaEpilogue)
    {
        long min = DiagnosticSwitches.Fp8StoreMinRows ?? DefaultFp8StoreMinRows;
        if (min <= 0 || output is null || cudaEpilogue is null || n % 128 != 0 || y.Type != DataType.F32 || !y.IsDense
            || y.Buffer != _arena || RootOf(output) != output || _fp8Copies.ContainsKey(output) || _seg.Outputs.Any(o => RootOf(o) == output)
            || PrefillCopyCode(output, n) != ActivationCode.Fp8)
        {
            return long.MaxValue;
        }

        return Math.Max(min, NativeGemmMinRows);
    }

    /// <summary>Whether <see cref="Fp8BlockScaled"/> is on where the device allows it and <c>SHARPLIGHT_FP8_BLOCK_SCALED</c> is not set.</summary>
    private const bool DefaultFp8BlockScaled = false;

    /// <summary>
    /// True when the 128-row tiles of the FP8 GEMM run on the block-scaled MMA (<c>gemm_fp8_tma.cu</c> with <c>F8N_BLOCK_SCALED</c>,
    /// <see cref="DeviceFeatures.BlockScaledFp8"/>):
    /// the products accumulate in the tensor cores over the whole of K, with the scales as UE8M0 exponents. That needs power-of-two scales,
    /// so in this mode every E4M3 activation row gets them (<c>E4M3_UE8M0</c>, e4m3.slang) and the grouped E4M3 weights are requantized to
    /// them at load (<see cref="WeightRepack.Fp8B128GroupedUe8m0"/>); every other kernel reads the same values, so decode and prompt steps
    /// agree. A numerics change (benchmarks.md section 86), chosen by <c>SHARPLIGHT_FP8_BLOCK_SCALED</c>.
    /// </summary>
    private bool Fp8BlockScaled => (DiagnosticSwitches.Fp8BlockScaled ?? DefaultFp8BlockScaled) && NativeGemmFp8 && NativeFp8TensorCopies
        && _opt.Has(DeviceFeatures.BlockScaledFp8);

    /// <summary>
    /// The launch variant of <c>gemm_fp8_tma.cu</c> (block-scaled with <see cref="Fp8BlockScaled"/>) for the batches <paramref name="serves"/> selects: one block per SM at most, each
    /// walking its share of the 128-row tiles, with the tensor maps over x (its <paramref name="xcap"/> rows) and the 128-row weight groups.
    /// A non-zero <paramref name="qcap"/>: the dual form stores the outputs as E4M3 activation rows with power-of-two scales and a dense
    /// scale copy of that many rows (<c>QOUT</c> of <c>gemm_fp8_tma.cu</c>, see <see cref="Fp8StoreMinRows"/>) instead of f32 rows.
    /// </summary>
    private KernelVariant NativeFp8TmaVariant(long n, long k, string? dualEpilogue, long xcap, Func<Bindings, bool> serves, Func<Bindings, long> rows, long qcap = 0)
    {
        bool dual = dualEpilogue is not null;
        long rowBytes = k + k / 32;
        var inv = CultureInfo.InvariantCulture;
        string WeightMap(int pointer) => string.Create(inv, $"{pointer} 3 {k},128,{n / 128} {k},{128 * rowBytes} 128,{(dual ? 64 : 128)},1");
        var maps = new List<string> { string.Create(inv, $"0 2 {k},{xcap} {rowBytes} 128,128"), WeightMap(1) };
        if (dual)
        {
            maps.Add(WeightMap(2));
        }

        var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("XCAP", xcap), ("DUAL", dual ? 1 : 0));
        if (qcap > 0)
        {
            macros["QOUT"] = "1";
            macros["QCAP"] = qcap.ToString(inv);
        }

        if (Fp8BlockScaled)
        {
            macros["F8N_BLOCK_SCALED"] = "1";
            macros["SL_FAMILY_ARCH"] = "1";
        }

        var kernel = _compiler.CompileCuda("gemm_fp8_tma.cu", dualEpilogue ?? "", "gemm_fp8_tma", NativeFp8TmaThreads, NativeFp8TmaSharedBytes + (qcap > 0 ? 512 : 0),
            _opt.Target, macros, _opt.Cache, string.Join(';', maps));
        long sms = Math.Max(1, _opt.ComputeUnits);
        // QOUT walks pairs of 64-output tiles.
        long columns = Grid.Div(n, dual && qcap == 0 ? NativeFp8Tile / 2 : NativeFp8Tile);
        return new KernelVariant(serves, kernel, b => new Grid((uint)Math.Min(sms, (long)Grid.Div(rows(b), NativeFp8Tile) * columns)));
    }

    /// <summary>Threads per block of <c>gemm_f16_native.cu</c> per tile configuration (0: 128 x 256, 1: 128 x 128).</summary>
    private static readonly int[] s_nativeF16Threads = [256, 128];

    /// <summary>Weight rows per tile of <c>gemm_f16_native.cu</c> per tile configuration.</summary>
    private static readonly int[] s_nativeF16TileColumns = [256, 128];

    /// <summary>Dynamic shared memory of <c>gemm_f16_native.cu</c> per tile configuration: two stages of 384 rows, three of 256 rows.</summary>
    private static readonly int[] s_nativeF16Shared = [2 * 384 * 128, 3 * 256 * 128];

    /// <summary>
    /// Relative speed of the f16 tile configurations at full waves (RTX PRO 4500 Blackwell, Qwen3-8B shapes at 2048 rows: 128 x 256
    /// 155-161 TFLOPS, 128 x 128 140-150).
    /// </summary>
    private static readonly double[] s_nativeF16Speed = [1.0, 0.94];

    /// <summary>
    /// True when the f16 products of the library GEMM path (bf16 weights stored as f16, the f16 copies of x) run on the hand-written
    /// f16 GEMM of the provider (<see cref="DeviceFeatures.NativeGemmF16"/>, <c>gemm_f16_native.cu</c>), preferred over the vendor
    /// library when both are there; disabling <c>native_gemm_f16</c> leaves them to the library (or the generated GEMMs).
    /// </summary>
    private bool NativeGemmF16 => _opt.Has(DeviceFeatures.NativeGemmF16) && _compiler.CompilesCuda(_opt.Target);

    /// <summary>The tile configuration of the hand-written f16 GEMM with the fewest estimated waves of work for a batch.</summary>
    private int NativeF16Config(long rows, long n, bool dual, Func<int, bool>? allowed = null)
    {
        int sms = Math.Max(1, _opt.ComputeUnits);
        double best = double.MaxValue;
        int config = -1;
        for (int c = 0; c < s_nativeF16TileColumns.Length; c++)
        {
            if (allowed is not null && !allowed(c))
            {
                continue;
            }

            long columns = dual ? s_nativeF16TileColumns[c] / 2 : s_nativeF16TileColumns[c];
            long tiles = Grid.Div(rows, 128) * ((n + columns - 1) / columns);
            double cost = (double)((tiles + sms - 1) / sms) * s_nativeF16TileColumns[c] / s_nativeF16Speed[c];
            if (cost < best)
            {
                best = cost;
                config = c;
            }
        }

        return config;
    }

    /// <summary>
    /// Launch variants of the hand-written f16 GEMM (single, or dual with <paramref name="dualEpilogue"/>) for batches of
    /// <paramref name="minRows"/> rows and more: one per tile configuration, each serving the row counts it is estimated fastest for.
    /// <paramref name="allowed"/> limits the configurations (the multi-output form's member boundaries on whole column tiles), and
    /// <paramref name="extra"/> adds macros (the multi-output form's).
    /// </summary>
    private KernelVariant[] NativeF16Variants(long n, long k, long maxRows, string? dualEpilogue, Func<Bindings, long> rows, long minRows,
        bool bias = false, long? rangeScales = null, Func<int, bool>? allowed = null, IReadOnlyDictionary<string, string>? extra = null)
    {
        bool dual = dualEpilogue is not null;
        var used = new SortedSet<int>();
        for (long r = minRows; r <= Math.Max(minRows, maxRows); r += 128)
        {
            used.Add(NativeF16Config(r, n, dual, allowed));
        }

        used.Add(NativeF16Config(Math.Max(minRows, maxRows), n, dual, allowed));
        used.Remove(-1);
        return [.. used.Select(c =>
        {
            var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("DUAL", dual ? 1 : 0), ("H_CFG", c));
            foreach (var (key, value) in extra ?? new Dictionary<string, string>())
            {
                macros[key] = value;
            }

            if (bias)
            {
                macros["H_BIAS"] = "1";
            }

            if (rangeScales is { } scales)
            {
                macros["H_XRS"] = scales.ToString(CultureInfo.InvariantCulture) + "ull";
            }

            var kernel = _compiler.CompileCuda("gemm_f16_native.cu", dualEpilogue?.Replace("F8N_EPI2", "H_EPI2", StringComparison.Ordinal) ?? "",
                "gemm_f16_native", s_nativeF16Threads[c], s_nativeF16Shared[c], _opt.Target, macros, _opt.Cache);
            long columns = dual ? s_nativeF16TileColumns[c] / 2 : s_nativeF16TileColumns[c];
            return new KernelVariant(b => rows(b) >= minRows && NativeF16Config(rows(b), n, dual, allowed) == c, kernel,
                b => new Grid(Grid.Div(rows(b), 128), Grid.Div(n, columns)));
        })];
    }

    /// <summary>Token rows of the decode form of <c>gemm_f16_native.cu</c> (H_CFG 2): the largest batch it serves.</summary>
    private const int NativeF16DecodeTileRows = 64;

    /// <summary>
    /// Weight rows per tile of the decode form of the hand-written f16 GEMM for <paramref name="n"/> outputs
    /// (<c>SHARPLIGHT_NATIVE_F16_DECODE_BN</c>; 0: no decode form): 64, or 32 where 64-row tiles would leave a quarter of the SMs
    /// without a block. RTX PRO 4500 Blackwell (82 SMs), gpt-oss-20b shapes at 64 rows, sustained over 24 weight copies
    /// (bench/s68_f16dec, µs): q / k / v [2880 → 4096 + 512 + 512] 39.4 on 64-row tiles (80 blocks), 40.2 on 32; o [4096 → 2880]
    /// 33.3 on 64-row tiles (45 blocks), 32.0 on 32 (90 blocks, two per SM).
    /// </summary>
    private int NativeF16DecodeColumns(long n)
        => DiagnosticSwitches.NativeF16DecodeColumns ?? (Grid.Div(n, 64) * 4 >= Math.Max(1, _opt.ComputeUnits) * 3 ? 64 : 32);

    /// <summary>
    /// Fewest rows the hand-written f16 GEMM's decode form serves (<c>SHARPLIGHT_NATIVE_F16_DECODE_MIN_ROWS</c>): below, gemm_cm2's
    /// K split fills the GPU better.
    /// </summary>
    private static long NativeF16DecodeMinRows => DiagnosticSwitches.NativeF16DecodeMinRows ?? 33;

    /// <summary>
    /// True when f16 weights [<paramref name="n"/>, <paramref name="k"/>] (bf16 stored as f16) read by f16 activation rows of stride
    /// <paramref name="k"/> (<paramref name="macros"/>: the f16 copy of x) run their decode batches on the hand-written f16 GEMM's decode
    /// form: the provider has the hand-written GEMM, K is a multiple of its 64-value steps, and the weight tiles fill at least half the SMs
    /// (narrower outputs, such as a router's, keep the split-K GEMM).
    /// </summary>
    private bool UseNativeF16Decode(DataType type, long n, long k, IReadOnlyDictionary<string, string> macros)
        => NativeGemmF16 && NativeF16DecodeColumns(n) is 32 or 64 && type == DataType.F16 && k % 64 == 0
           && macros.GetValueOrDefault("X_F16") == "1" && macros.GetValueOrDefault("X_STRIDE") == k.ToString(CultureInfo.InvariantCulture)
           && Grid.Div(n, NativeF16DecodeColumns(n)) * 2 >= Math.Max(1, _opt.ComputeUnits);

    /// <summary>
    /// The launch variant of the decode form of <c>gemm_f16_native.cu</c> (64 token rows, <see cref="NativeF16DecodeColumns"/> weight
    /// rows per tile, no K split) for [<paramref name="n"/>, <paramref name="k"/>] weights, with the bias store (<paramref name="bias"/>),
    /// the row scales of a row-ranged copy of x at byte <paramref name="rangeScales"/>, and the <paramref name="extra"/> macros (the
    /// multi-output form's).
    /// </summary>
    private KernelVariant NativeF16DecodeVariant(long n, long k, Func<Bindings, bool> serves, Func<Bindings, long> rows, bool bias, long? rangeScales,
        IReadOnlyDictionary<string, string>? extra = null)
    {
        int columns = NativeF16DecodeColumns(n);
        var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("H_CFG", 2), ("H_DBN", columns));
        if (bias)
        {
            macros["H_BIAS"] = "1";
        }

        if (rangeScales is { } scales)
        {
            macros["H_XRS"] = scales.ToString(CultureInfo.InvariantCulture) + "ull";
        }

        foreach (var (name, value) in extra ?? new Dictionary<string, string>())
        {
            macros[name] = value;
        }

        const int stages = 4;
        int threads = 2 * (columns / 16) * 32;
        var kernel = _compiler.CompileCuda("gemm_f16_native.cu", "", "gemm_f16_native", threads, stages * (NativeF16DecodeTileRows + columns) * 128,
            _opt.Target, macros, _opt.Cache);
        return new KernelVariant(serves, kernel, b => new Grid(Grid.Div(rows(b), NativeF16DecodeTileRows), Grid.Div(n, columns)));
    }

    /// <summary>Token rows of the routing tiles of <c>gemm_moe_native.cu</c> (one warp row of 8 row fragments).</summary>
    private const int NativeMoeTileRows = 128;

    /// <summary>Warps of a <c>gemm_moe_native.cu</c> block (all side by side along the weight rows).</summary>
    private const int NativeMoeWarps = 8;

    /// <summary>Stages of the <c>cp.async</c> ring of <c>gemm_moe_native.cu</c> where they fit (else 3).</summary>
    private const int NativeMoeStages = 4;

    /// <summary>Static shared memory of <c>gemm_moe_native.cu</c> (its E8M0 factor table) next to the dynamic stages.</summary>
    private const int NativeMoeStaticShared = 1024;

    /// <summary>
    /// True when the expert products of <paramref name="w"/> ([E, <paramref name="n"/>, <paramref name="k"/>]) can run on the hand-written
    /// grouped MXFP4 GEMM (<see cref="DeviceFeatures.NativeGemmMoe"/>, <c>gemm_moe_native.cu</c>): MXFP4 experts in the 16-row groups of
    /// <see cref="WeightRepack.Mxfp4Split"/>, K a multiple of 64, on a target that takes CUDA C++ with room for three stages.
    /// </summary>
    private bool UseNativeMoeGemm(TensorRef w, long k, long n)
        => _opt.Has(DeviceFeatures.NativeGemmMoe) && _compiler.CompilesCuda(_opt.Target) && w.Type == ShaderTypes.MXFP4R && k % 64 == 0 && n % 32 == 0
           && DiagnosticSwitches.MoeSortedHalfActivations && _opt.MaxSharedMemory >= NativeMoeStageBytes(256, true) * 3 + NativeMoeStaticShared;

    /// <summary>
    /// Routed items (token, slot) from which the hand-written grouped MXFP4 GEMM serves a step of <paramref name="experts"/> experts:
    /// <see cref="NativeMoeDecodeMinItemsPerExpert"/> rows per expert on average (the expert-grouped skinny GEMM serves fewer), or <c>SHARPLIGHT_NATIVE_MOE_MIN_ROWS</c> per
    /// expert. RTX PRO 4500, gpt-oss-20b mixed serving steps (64 decode rows plus a prompt, GPU ms): 171 rows 33.7 → 25.8, 256 rows
    /// 40.2 → 29.6, 512 rows 58.2 → 45.6, 2048 rows 172.2 → 147.0 against gemm_cm2.
    /// </summary>
    private static long NativeMoeMinItems(long experts) => (DiagnosticSwitches.NativeMoeMinRows ?? NativeMoeDecodeMinItemsPerExpert) * experts;

    /// <summary>
    /// Routed items per expert from which the hand-written grouped MXFP4 GEMM takes a step over from the expert-grouped skinny GEMM
    /// (<see cref="NativeMoeMinItems"/>, on its <see cref="NativeMoeDecodeTileRows"/>-row tiles). RTX PRO 4500, gpt-oss-20b decode steps
    /// on text routing (BatchBench <c>--text</c>, GPU ms, two runs): from 12 → 3 per expert 24 sequences 15.92-16.07 → 15.67-15.71,
    /// 32: 17.13 → 16.56, 48: 19.16-19.32 → 18.19-18.22, 64: 20.32-21.06 → 19.52-19.55; from 3 → 1 per expert 8 sequences 10.91-10.92
    /// → 10.72-10.74, 16: 15.20-15.23 → 14.71-14.77, 20: 15.22-15.27 → 14.82-14.84 (likewise with fed-back random-id prompts and with
    /// fixed random ids). Fewer items than one per expert (up to 7 sequences of 4 slots) stay on the skinny GEMM or the matvec.
    /// </summary>
    private const long NativeMoeDecodeMinItemsPerExpert = 1;

    /// <summary>Bytes of one stage of <c>gemm_moe_native.cu</c>: activation rows, codes and scales of its weight rows.</summary>
    private static int NativeMoeStageBytes(int columns, bool dual, int tileRows = NativeMoeTileRows)
        => tileRows * 128 + 2 * columns * 16 + 2 * (dual ? 2 : 1) * (dual ? columns / 2 : columns) / 16 * 16;

    /// <summary>The stages of <c>gemm_moe_native.cu</c> that fit the device's shared memory (<see cref="NativeMoeStages"/> or 3).</summary>
    private int NativeMoeStagesFor(int columns, bool dual, int tileRows = NativeMoeTileRows)
        => _opt.MaxSharedMemory >= NativeMoeStageBytes(columns, dual, tileRows) * NativeMoeStages + NativeMoeStaticShared ? NativeMoeStages : 3;

    /// <summary>
    /// Routed items per expert up to which the hand-written grouped MXFP4 GEMM runs <see cref="NativeMoeShortTileRows"/>-row routing
    /// tiles (<c>SHARPLIGHT_NATIVE_MOE_SHORT=n</c>, 0 for none). The 128-row tile's activation stages are mostly empty there, so the
    /// short tile's smaller stages let it run two blocks per SM (<see cref="NativeMoeConfig"/>). RTX PRO 4500, gpt-oss-20b mixed steps
    /// (BatchBench, GPU ms): 64 + 107 rows (21 items per expert) 24.72 → 24.22, but 64 + 192 rows (32 per expert) 28.72 → 29.25.
    /// </summary>
    private const long NativeMoeShortItemsPerExpert = 24;

    /// <summary>Token rows of the short routing tiles of <c>gemm_moe_native.cu</c> (see <see cref="NativeMoeShortItemsPerExpert"/>).</summary>
    private const int NativeMoeShortTileRows = 64;

    /// <summary>
    /// Routed items per expert up to which the hand-written grouped MXFP4 GEMM runs <see cref="NativeMoeDecodeTileRows"/>-row routing tiles
    /// (batched decode steps; on when the short tiles are). An expert's 32-row tile holds the 4-30 items of a decode step's
    /// skewed routing in one pass over its weights, where the skinny GEMM's 16-row tiles read a busy expert two or three times.
    /// </summary>
    private const long NativeMoeDecodeItemsPerExpert = 12;

    /// <summary>Token rows of the decode routing tiles of <c>gemm_moe_native.cu</c> (see <see cref="NativeMoeDecodeItemsPerExpert"/>).</summary>
    private const int NativeMoeDecodeTileRows = 32;

    /// <summary>The routing tile rows the hand-written grouped MXFP4 GEMM is compiled for: the decode and short tiles when they are on, and the full one.</summary>
    private static int[] NativeMoeTiles
        => (DiagnosticSwitches.NativeMoeShortItems ?? NativeMoeShortItemsPerExpert) > 0
            ? [NativeMoeDecodeTileRows, NativeMoeShortTileRows, NativeMoeTileRows]
            : [NativeMoeTileRows];

    /// <summary>
    /// Routing tile rows of the hand-written grouped MXFP4 GEMM for a step of <paramref name="items"/> routed items over
    /// <paramref name="experts"/> experts: <see cref="NativeMoeDecodeTileRows"/> up to <see cref="NativeMoeDecodeItemsPerExpert"/> items per
    /// expert, <see cref="NativeMoeShortTileRows"/> up to <see cref="NativeMoeShortItemsPerExpert"/>, else <see cref="NativeMoeTileRows"/>.
    /// </summary>
    private static int NativeMoeTileRowsFor(long items, long experts)
    {
        long shortItems = DiagnosticSwitches.NativeMoeShortItems ?? NativeMoeShortItemsPerExpert;
        if (shortItems > 0 && items <= Math.Min(shortItems, NativeMoeDecodeItemsPerExpert) * experts)
        {
            return NativeMoeDecodeTileRows;
        }

        return items <= shortItems * experts ? NativeMoeShortTileRows : NativeMoeTileRows;
    }

    /// <summary>
    /// Weight rows per tile, stages and resident blocks per SM (<c>NM_BN</c>, <c>NM_STAGES</c>, <c>NM_MINB</c>) of <c>gemm_moe_native.cu</c>
    /// for <paramref name="n"/> outputs and routing tiles of <paramref name="tileRows"/> rows. A short tile runs two blocks of three stages
    /// per SM where they fit the shared memory of an SM (taken as the per-block limit plus the 1 KB the hardware reserves per block, as on
    /// compute capability 8.x-12.x; where fewer fit, the bound only caps the registers), narrowing a dual tile to 128 weight rows for it;
    /// otherwise one block with <see cref="NativeMoeTileColumns"/> and <see cref="NativeMoeStagesFor"/>. <c>SHARPLIGHT_NATIVE_MOE_BN</c>
    /// fixes the width, <c>SHARPLIGHT_NATIVE_MOE_BLOCKS=1</c> keeps one block. RTX PRO 4500, gpt-oss-20b shapes (bench/s56_moe, 171
    /// tokens, skewed routing): gate / up 0.510 → 0.492 ms (64 rows, 128-row dual tiles, two blocks), down 0.307 → 0.266 ms.
    /// </summary>
    private (int Columns, int Stages, int MinBlocks, bool Table) NativeMoeConfig(long n, bool dual, int tileRows)
    {
        int columns = DiagnosticSwitches.NativeMoeColumns ?? NativeMoeTileColumns(n, dual);
        bool TwoFit(int width, int stages = 3, int staticShared = NativeMoeStaticShared)
            => 2 * (NativeMoeStageBytes(width, dual, tileRows) * (long)stages + staticShared + 1024) <= _opt.MaxSharedMemory + 1024;
        if (tileRows < NativeMoeTileRows && DiagnosticSwitches.NativeMoeBlocks != 1)
        {
            // Decode tiles: dual tiles of 128 weight rows (twice the blocks: the step reads few weights per expert) and a fourth stage.
            if (dual && DiagnosticSwitches.NativeMoeColumns is null && TwoFit(128) && (!TwoFit(columns) || tileRows == NativeMoeDecodeTileRows))
            {
                columns = 128;
            }

            if (TwoFit(columns))
            {
                // A fourth stage where it fits, for a dual tile also without the factor table (NM_TABLE=0: the 64-row dual tile's four
                // stages take an SM's 100 KB exactly). RTX PRO 4500, gpt-oss-20b gate / up, 64-row tiles (bench/s56_moe, read eviction,
                // 32-256 tokens): 0.381-0.463 → 0.369-0.457 ms.
                if (TwoFit(columns, 4) && (tileRows == NativeMoeDecodeTileRows || dual))
                {
                    return (columns, 4, 2, true);
                }

                bool tableFree = dual && DiagnosticSwitches.NativeMoeTable != 1 && TwoFit(columns, 4, 0);
                return (columns, tableFree ? 4 : 3, 2, !tableFree);
            }
        }

        return (columns, NativeMoeStagesFor(columns, dual, tileRows), 1, true);
    }

    /// <summary>
    /// Weight rows per tile of <c>gemm_moe_native.cu</c> for <paramref name="n"/> outputs: the width (256, 192 or 128 rows; dual pairs
    /// 256 or 128, their 8 warps each holding whole gate / up fragment pairs) with the most useful columns per estimated time.
    /// RTX PRO 4500, gpt-oss-20b shapes at 2048 tokens: 192 and 256 rows equally fast per column, 128 rows ~10 % slower (more
    /// activation traffic per product).
    /// </summary>
    private static int NativeMoeTileColumns(long n, bool dual)
    {
        int[] widths = dual ? [256, 128] : [256, 192, 128];
        int best = widths[0];
        double bestScore = 0;
        foreach (int width in widths)
        {
            long outputs = dual ? width / 2 : width;
            double score = (double)n / (Grid.Div(n, outputs) * outputs) * (width == 128 ? 0.9 : 1.0);
            if (score > bestScore)
            {
                bestScore = score;
                best = width;
            }
        }

        return best;
    }

    /// <summary>
    /// The CUDA C++ form of the dual epilogue <c>epi2</c> (see <see cref="DualEpilogueSource"/>) for the hand-written dual GEMM, as the
    /// <c>F8N_EPI2</c> macro over a device function, or null when the group has an operation without a CUDA form here (the generated
    /// dual GEMM then serves the pair). The expressions are those of the Slang epilogue with single-precision literals and <c>__expf</c>.
    /// </summary>
    private static string? DualEpilogueCudaSource(Node a, Node bNode, List<Node> group, Value output)
    {
        string[] supported = ["Neg", "Exp", "Log", "Sqrt", "Rsqrt", "Sigmoid", "Silu", "Gelu", "GeluTanh", "Tanh", "Relu", "Square", "Recip",
            "Add", "Sub", "Mul", "Max", "Min", "Scale", "Cast", "Clip"];
        if (group.Any(node => !supported.Contains(node.Op.Name)))
        {
            return null;
        }

        var name = new Dictionary<Value, string> { [a.Output] = "m0", [bNode.Output] = "m1" };
        var src = new StringBuilder();
        src.AppendLine("static __device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }");
        src.AppendLine("static __device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }");
        src.AppendLine("static __device__ __forceinline__ float sl_epi2(float m0, float m1)");
        src.AppendLine("{");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            if (!name.ContainsKey(node.Inputs[0]) || node.Inputs.Skip(1).Any(i => !name.ContainsKey(i)))
            {
                // An input from outside the pair (a bias, a mask): not expressible in this epilogue.
                return null;
            }

            src.AppendLine(CultureInfo.InvariantCulture, $"    float {local} = {SinglePrecisionLiterals(OpExpr(node, [.. node.Inputs.Select(i => name[i])]))};");
            name[node.Output] = local;
        }

        src.AppendLine(CultureInfo.InvariantCulture, $"    return {name[output]};");
        src.AppendLine("}");
        src.AppendLine("#define F8N_EPI2(a, b) sl_epi2(a, b)");
        return src.ToString();
    }

    /// <summary>Floating-point literals of a generated expression as <c>float</c> literals (C++ reads <c>1.0</c> as a double).</summary>
    /// <remarks><c>exp</c> becomes <c>__expf</c>, as in the CUDA form of the Slang kernels (<c>common.slang</c>), so the fused epilogue
    /// rounds like the elementwise launch it replaces.</remarks>
    private static string SinglePrecisionLiterals(string expression)
        => CudaExpression(expression);

    /// <summary>A generated elementwise expression in its CUDA C++ form: <c>float</c> literals and <c>__expf</c> (see <see cref="SinglePrecisionLiterals"/>).</summary>
    internal static string CudaExpression(string expression)
        => ExpCall().Replace(FloatLiteral().Replace(expression, m => m.Value + "f"), "__expf(");

    [GeneratedRegex(@"(?<![\w])exp\(")]
    private static partial Regex ExpCall();

    [GeneratedRegex(@"(?<![\w.])(\d+\.\d*([eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")]
    private static partial Regex FloatLiteral();
}
