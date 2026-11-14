using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Smallest activation row count served by the skinny GEMM (single rows stay on the bandwidth-bound matvec).</summary>
    private const long SkinnyMinRows = 2;

    /// <summary>
    /// Largest activation row count the skinny GEMM serves at any output width. Up to 16 rows it reads the weights at the 1-row
    /// matvec's speed; above, its per-subgroup tiles make it slower than gemm_cm2's shared tiles (Blackwell, FP8 [5120→17408]:
    /// 80 vs 74 µs at 32 rows, 172 vs 106 µs at 64).
    /// </summary>
    private static long SkinnyMaxRows => DiagnosticSwitches.SkinnyGemmMaxRows ?? 16;

    /// <summary>
    /// Largest activation row count the skinny GEMM serves for <paramref name="n"/> output rows (below
    /// <see cref="SkinnyMinRows"/>: none). Its grid has one workgroup per 16-32 output rows and each walks all of K, so
    /// narrow matrices leave the GPU idle and take a fixed ~20 µs at K = 5120, where the split-K matvec (up to 8 rows) and
    /// the split-K gemm_cm2 (9+ rows) spread K over many workgroups. Blackwell, FP8 at K = 5120, skinny vs those: N = 48
    /// 24 vs 8 µs at 4 rows; N = 1024 23 vs 20 µs at 4 rows, 23 vs 35 µs at 8 rows, 21 vs 12 µs at 16 rows; N = 2048
    /// 21 vs 16 µs at 16 rows; N = 4096 even at 16 rows.
    /// </summary>
    private static long SkinnyMaxRowsFor(long n) => DiagnosticSwitches.SkinnyGemmMaxRows ?? (n >= 4096 ? SkinnyMaxRows : n >= 1024 ? 8 : 0);

    /// <summary>
    /// Subgroups a skinny GEMM launch aims for in total: below this, workgroups get 8 subgroups instead of 4, which keeps
    /// enough weight loads in flight when there are few output rows (Blackwell, FP8 [17408→5120] at 4-16 rows: 70-80 µs
    /// with 8 subgroups vs 84-92 µs with 4).
    /// </summary>
    private const long SkinnyTargetSubgroups = 1536;

    /// <summary>
    /// Output rows from which a single (not dual) skinny GEMM gives each workgroup two weight fragments (32 rows) instead of
    /// one: below, one fragment per workgroup doubles the grid (Blackwell, FP8 at 16 rows: [17408→5120] 80 vs 95 µs,
    /// [5120→6144] 31 vs 39 µs); from here the grid is large anyway and two fragments share the staged activations
    /// ([5120→12288] 56 vs 64 µs, [5120→10240] 46 vs 51 µs).
    /// </summary>
    private const long SkinnyTwoFragmentMinN = 10240;

    /// <summary>
    /// Shared memory the skinny GEMM's staged slices may use, at most the device's limit: 64-value K blocks with 8 subgroups
    /// need 37 KB.
    /// </summary>
    private long SkinnySharedBytes => Math.Min(_opt.MaxSharedMemory, 48 * 1024);

    /// <summary>
    /// Token fragments (16 rows each) of the skinny GEMM variants; each serves rows up to 16 x its count (2 and 4 are only
    /// reached with a larger <c>SHARPLIGHT_SKINNY_MAX_ROWS</c>).
    /// </summary>
    private static readonly int[] s_skinnyTokenFragments = [1, 2, 4];

    /// <summary>
    /// True when the skinny tensor-core GEMM (<c>matmul_skinny.slang</c>) can serve the small batches of a matmul with
    /// weights of this plan (for formats with an integer-dot matvec, the rows above its range): the f16 tensor-core path is
    /// usable (<see cref="MatMulPlan.Coop"/>, which needs <see cref="DeviceFeatures.MatrixF16"/>, <see cref="DeviceFeatures.Float16"/> and 32-wide subgroups) and K splits into
    /// 32-value blocks. Weights are decoded to f16 as in gemm_cm2, so every format gemm_cm2 reads works here too.
    /// </summary>
    private static bool UseSkinnyGemm(MatMulPlan p) => DiagnosticSwitches.SkinnyGemm && p.Coop && p.K % 32 == 0;

    /// <summary>
    /// True when <paramref name="rows"/> activation rows are in the skinny GEMM's range for <paramref name="n"/> output rows,
    /// starting at <paramref name="minRows"/>.
    /// </summary>
    private static bool InSkinnyRange(long rows, long n, long minRows) => rows >= minRows && rows <= SkinnyMaxRowsFor(n);

    /// <summary>
    /// Smallest row count for which the skinny GEMM reads an f16 copy of x (shared with the matmul's siblings) rather than
    /// converting the f32 rows while staging. Every workgroup re-reads x, about T / 8 of the weight bytes in f32, so the copy
    /// pays from 2 rows (Blackwell, FP8 [17408→5120] at 2-4 rows: 60 µs with the copy vs 75 µs from f32 rows).
    /// <c>SHARPLIGHT_SKINNY_X16=0</c> reads f32 rows at every row count.
    /// </summary>
    private static long SkinnyHalfMinRows => DiagnosticSwitches.SkinnyGemmHalfActivations == false ? long.MaxValue : SkinnyMinRows;

    /// <summary>
    /// Emits the skinny GEMM launches of a matmul (or of a dual matmul when <paramref name="weights"/> holds two matrices and
    /// <paramref name="compile"/> builds the kernel from a source with the <c>epi2</c> epilogue), conditioned on the row count
    /// being in the skinny range from <paramref name="minRows"/> (2, or above the int-dot matvec range): one reading f32 x
    /// below <see cref="SkinnyHalfMinRows"/>, one reading <paramref name="half"/>
    /// (an f16 copy of x covering its rows) or, when that is null, the shared f16 copy of <paramref name="source"/>'s rows.
    /// </summary>
    private void EmitSkinnyGemm(long minRows, string label, Node[] nodes, Value source, TensorRef x, long xStride, long rowsPerLead, TensorRef[] weights,
        TensorRef y, long n, long k, Dictionary<string, string> baseMacros, Func<Dictionary<string, string>, CompiledKernel> compile, TensorRef? half = null)
    {
        long rowsMax = x.Shape[0] * rowsPerLead;
        long maxRows = Math.Min(SkinnyMaxRowsFor(n), rowsMax);
        long halfMin = Math.Max(minRows, SkinnyHalfMinRows);
        if (minRows < halfMin && minRows <= maxRows)
        {
            EmitSkinnyLaunch(label, nodes, x, rowsPerLead, x, xStride, false, minRows, Math.Min(maxRows, halfMin - 1), weights, y, n, k, baseMacros, compile);
        }

        if (halfMin <= maxRows)
        {
            var activations = half ?? QuantizedActivations(label, nodes, source, x, xStride, rowsPerLead, k, halfMin, maxRows, ActivationCode.F16);
            EmitSkinnyLaunch(label, nodes, x, rowsPerLead, activations, k, true, halfMin, maxRows, weights, y, n, k, baseMacros, compile);
        }
    }

    /// <summary>One skinny GEMM launch for <paramref name="minRows"/> to <paramref name="maxRows"/> rows, with a variant per token-fragment count.</summary>
    private void EmitSkinnyLaunch(string label, Node[] nodes, TensorRef x, long rowsPerLead, TensorRef activations, long activationStride, bool half,
        long minRows, long maxRows, TensorRef[] weights, TensorRef y, long n, long k, Dictionary<string, string> baseMacros,
        Func<Dictionary<string, string>, CompiledKernel> compile)
    {
        Func<Bindings, long> rows = b => Rows(b, x) * rowsPerLead;
        var variants = new List<KernelVariant>();
        long previous = minRows - 1;
        foreach (int fm in s_skinnyTokenFragments)
        {
            long bucketMax = Math.Min(16L * fm, maxRows);
            if (bucketMax <= previous)
            {
                continue;
            }

            bool direct = UseDirectSkinny(half, fm, k);
            var (fn, sg, bk) = SkinnyConfig(n, k, fm, weights.Length, SkinnySharedBytes, direct);
            var macros = new Dictionary<string, string>(baseMacros)
            {
                ["X_STRIDE"] = activationStride.ToString(CultureInfo.InvariantCulture),
                ["X_F16"] = half ? "1" : "0",
                ["SK_FM"] = fm.ToString(CultureInfo.InvariantCulture),
                ["SK_FN"] = fn.ToString(CultureInfo.InvariantCulture),
                ["SK_SG"] = sg.ToString(CultureInfo.InvariantCulture),
                ["SK_BK"] = bk.ToString(CultureInfo.InvariantCulture),
                ["SK_DIRECT"] = direct ? "1" : "0",
            };
            long low = previous;
            long high = bucketMax;
            int rowsPerWorkgroup = fn * 16;
            variants.Add(new KernelVariant(b => rows(b) > low && rows(b) <= high, compile(macros), _ => new Grid(Grid.Div(n, rowsPerWorkgroup))));
            previous = bucketMax;
        }

        Emit($"{label} skinny{(half ? "" : " f32")}", nodes, variants,
            [Pointer(activations), .. weights.Select(Pointer), Pointer(y), TokenArg(x, rowsPerLead)],
            [activations, .. weights], [y]);
    }

    /// <summary>
    /// True when the skinny GEMM launch for <paramref name="fm"/> token fragments loads its operands straight into the fragment
    /// registers (<c>SK_DIRECT</c> in <c>matmul_skinny.slang</c>): the device's fragments have the MMA element layout
    /// (<see cref="DeviceFeatures.MatrixMmaLayout"/>), one token fragment (up to 16 rows) reading the f16 copy of x, K divisible
    /// by 64. Blackwell, FP8 at 16 rows: [5120→17408]x2 gate/up 141 → 127 µs, [17408→5120] 83 → 70 µs (with the aligned FP8
    /// layout). Without the feature, or with <c>SHARPLIGHT_SKINNY_DIRECT=0</c>, the operands are staged in shared memory.
    /// </summary>
    private bool UseDirectSkinny(bool half, int fm, long k)
        => DiagnosticSwitches.SkinnyGemmDirect && _opt.Has(DeviceFeatures.MatrixMmaLayout) && half && fm == 1 && k % 64 == 0;

    /// <summary>
    /// Skinny GEMM configuration for <paramref name="fm"/> token fragments: weight fragments per workgroup, subgroups splitting
    /// K inside the workgroup and the K block. One weight fragment per workgroup except for wide single matrices, more
    /// subgroups when there are few workgroups, so the launch keeps enough loads in flight; never more subgroups than K blocks.
    /// The K block shrinks (then the subgroups halve) until the staged slices fit <paramref name="sharedBytes"/>. With
    /// <paramref name="direct"/>, one fragment and 8 subgroups (the K block is not used).
    /// </summary>
    private static (int Fn, int Sg, int Bk) SkinnyConfig(long n, long k, int fm, int matrices, long sharedBytes, bool direct)
    {
        int bk = k % 64 == 0 ? 64 : 32;
        int fn = matrices > 1 || fm > 2 || n < SkinnyTwoFragmentMinN ? 1 : 2;
        long workgroups = (n + fn * 16 - 1) / (fn * 16);
        int sg = 4;
        if (direct)
        {
            // The direct path keeps no slices in shared memory, so it takes 8 subgroups for every shape: one fragment and 8
            // subgroups was the best or within 3 % of the best configuration for every 27B shape at 4 and 16 rows (Blackwell).
            fn = 1;
            sg = (int)Math.Clamp(k / 64, 1, 8);
        }

        while (!direct && sg < 8 && workgroups * sg < SkinnyTargetSubgroups && k / bk >= sg * 4)
        {
            sg *= 2;
        }

        if (DiagnosticSwitches.SkinnyGemmConfig is { Length: 3 } forced)
        {
            (fn, sg, bk) = (forced[0], forced[1], k % forced[2] == 0 ? forced[2] : 32);
        }

        // Shared memory: every subgroup stages fm + fn * matrices fragments of 16 rows x (bk + 8) halves (the direct path only
        // keeps the layout probe and the final reduction there).
        while (!direct && sg * (fm + fn * matrices) * 16 * (bk + 8) * 2 > sharedBytes)
        {
            if (bk > 32)
            {
                bk = 32;
            }
            else if (sg > 2)
            {
                sg /= 2;
            }
            else
            {
                break;
            }
        }

        return (fn, sg, bk);
    }
}
