using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>MatMulId (expert-indexed matmul): per-item matvec for few rows, grouped-by-expert GEMM for many.</summary>
public sealed class MoeTests
{
    // Grouped (tensor-core / tiled) path above 16 (token, slot) items may round activations/weights to f16 (K-quant
    // weights in two steps: the block scale, then the scaled code).
    private static double Atol(int items) => items > 16 ? 2.5e-1 : 2e-3;
    private static double Rtol(int items) => items > 16 ? 1e-2 : 1e-3;

    // Up to 128 items, q8_0 and K-quant experts with K % 256 == 0 may run on the integer-dot matvec, which quantizes the
    // activations to 8 bits per 32-value block: an absolute error proportional to the (large) norms of these random
    // weight rows. IntDotExactActivations checks its arithmetic tightly.
    private static bool IntDot(string fmt, int items, int k)
        => items <= 128 && (k % 256 == 0 && fmt is "q4_k" or "q5_k" or "q6_k" || k % 32 == 0 && fmt is "q8_0" or "q4_0" or "q5_0" or "iq4_nl" or "mxfp4");

    private static double Atol(string fmt, int items, int k) => IntDot(fmt, items, k) ? 3.0 : Atol(items);

    private static double Rtol(string fmt, int items, int k) => IntDot(fmt, items, k) ? 1e-2 : Rtol(items);

    /// <summary>ids pattern: "random" (all experts), "few" (only experts 1 and 5 get tokens), "one" (a single expert).</summary>
    private static int[] Ids(string pattern, int t, int s, int e)
    {
        var rng = new Random(11 + t * 31 + s);
        return pattern switch
        {
            "few" => [.. Enumerable.Range(0, t * s).Select(_ => rng.Next(2) == 0 ? 1 : 5)],
            "one" => [.. Enumerable.Range(0, t * s).Select(_ => e - 1)],
            _ => [.. Enumerable.Range(0, t * s).Select(_ => rng.Next(e))],
        };
    }

    private static void Run(string fmt, int t, int s, bool perSlot, string pattern, int E = 8, int N = 96, int K = 512, bool exactInt8 = false)
    {
        Gpu.Require();
        var tg = new TG();
        var x = perSlot ? tg.InF("x", TG.T, s, K) : tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, s);
        var w = tg.RandW("w", DataType.Get(fmt), 1f, E, N, K);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, w, ids]));
        int items = t * s;
        long xElements = (long)(perSlot ? items : t) * K;
        var xData = exactInt8 ? Rnd.ExactInt8(xElements, 3) : Rnd.F(xElements, 3);
        double atol = exactInt8 ? Atol(items) : Atol(fmt, items, K);
        double rtol = exactInt8 ? Rtol(items) : Rtol(fmt, items, K);
        Diff.Check("MatMulId." + fmt, tg.B.Build(), new Step(t, 0, (x, xData), (ids, Ids(pattern, t, s, E))),
            atol: atol, rtol: rtol, maxTokens: 128, what: $"T={t} S={s} perSlot={perSlot} ids={pattern}");
    }

    /// <summary>Activations exactly representable in 8 bits per block: the integer-dot expert matvec must match tightly.</summary>
    [Theory]
    [InlineData("q8_0", 1, 4, false)]
    [InlineData("q4_k", 3, 8, false)]
    [InlineData("q5_k", 2, 4, true)]
    [InlineData("q6_k", 4, 2, false)]
    [InlineData("q4_0", 2, 4, false)]
    [InlineData("q5_0", 3, 2, true)]
    [InlineData("mxfp4", 1, 4, false)]
    [InlineData("mxfp4", 5, 4, false)]
    public void IntDotExactActivations(string fmt, int t, int s, bool perSlot) => Run(fmt, t, s, perSlot, "random", exactInt8: true);

    public static IEnumerable<object[]> FormatCases()
    {
        foreach (var fmt in new[] { "f32", "f16", "bf16", "q8_0", "q4_0", "q5_0", "q4_k", "q5_k", "q6_k", "mxfp4" })
        {
            foreach (var t in new[] { 1, 3, 70 })
            {
                yield return [fmt, t];
            }
        }
    }

    [Theory]
    [MemberData(nameof(FormatCases))]
    public void WeightFormat(string fmt, int t) => Run(fmt, t, 4, false, "random");

    [Theory]
    [InlineData(1, 1)]
    [InlineData(1, 2)]
    [InlineData(1, 8)]
    [InlineData(3, 2)]
    [InlineData(3, 8)]
    [InlineData(5, 4)]
    [InlineData(70, 1)]
    [InlineData(70, 2)]
    [InlineData(70, 8)]
    public void TopKValues(int t, int s) => Run("q4_k", t, s, false, "random");

    [Theory]
    [InlineData(1, 4)]
    [InlineData(3, 4)]
    [InlineData(70, 4)]
    [InlineData(70, 8)]
    public void PerSlotInput(int t, int s) => Run("q4_k", t, s, true, "random");

    [Theory]
    [InlineData("f16", 70, "few")]
    [InlineData("q4_k", 70, "few")]
    [InlineData("q4_k", 70, "one")]
    [InlineData("q6_k", 20, "few")]
    [InlineData("f32", 40, "one")]
    public void ExpertsWithoutTokens(string fmt, int t, string pattern) => Run(fmt, t, 2, false, pattern);

    [Theory]
    [InlineData(64, 24, 256)]
    [InlineData(16, 130, 256)]
    [InlineData(32, 33, 512)]
    [InlineData(4, 200, 288)]
    public void ShapeVariants(int E, int N, int K) => Run(K % 256 == 0 ? "q4_k" : "q8_0", 70, 3, false, "random", E, N, K);

    /// <summary>
    /// Multi-head latent attention's absorbed projections: one "expert" per head, every head selected once per row (ids = head index),
    /// f16 per-head matrices with small output sizes.
    /// </summary>
    [Theory]
    [InlineData(16, 512, 128, 1)]
    [InlineData(16, 128, 512, 1)]
    [InlineData(16, 512, 128, 40)]
    [InlineData(16, 128, 512, 40)]
    [InlineData(4, 384, 64, 27)]
    [InlineData(4, 64, 384, 27)]
    public void LatentHeadProjections(int heads, int N, int K, int t)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, heads, K);
        var ids = tg.In("ids", DataType.I32, TG.T, heads);
        var w = tg.RandW("w", DataType.F16, 1f, heads, N, K);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, w, ids]));
        int[] headIds = [.. Enumerable.Range(0, t * heads).Select(i => i % heads)];
        int items = t * heads;
        Diff.Check("MatMulId.heads", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)items * K, 3)), (ids, headIds)),
            atol: Atol(items), rtol: Rtol(items), maxTokens: 128, what: $"H={heads} N={N} K={K} T={t}");
    }

    /// <summary>
    /// One ids tensor shared by a product that stays on the per-item matvec (K not a multiple of 32, no tensor-core path) and one that
    /// takes the grouped GEMM at the same item count: the GEMM must get its own routing launch.
    /// </summary>
    [Fact]
    public void SharedIdsWithDifferentPaths()
    {
        Gpu.Require();
        int E = 8;
        int K = 48;
        int H = 256;
        int S = 2;
        int t = 40;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, S);
        var up = tg.RandW("up", DataType.F16, 0.1f, E, H, K);
        var down = tg.RandW("down", DataType.F16, 0.1f, E, K, H);
        var h = tg.B.Emit1("MatMulId", [x, up, ids]);
        tg.B.Output("d", tg.B.Emit1("MatMulId", [h, down, ids]));
        Diff.Check("MatMulId.paths", tg.B.Build(), new Step(t, 0, (x, Rnd.F(t * K, 3)), (ids, Ids("random", t, S, E))),
            atol: Atol(t * S), rtol: Rtol(t * S), maxTokens: 128);
    }

    /// <summary>Two expert products on one ids tensor followed by a second block with other ids (routing is recomputed).</summary>
    [Fact]
    public void ChainedBlocks()
    {
        Gpu.Require();
        int E = 8;
        int K = 256;
        int H = 256;
        int S = 2;
        int t = 40;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, S);
        var ids2 = tg.In("ids2", DataType.I32, TG.T, S);
        var up = tg.RandW("up", DataType.Q4_K, 0.1f, E, H, K);
        var down = tg.RandW("down", DataType.Q4_K, 0.1f, E, K, H);
        var h = tg.B.Emit1("MatMulId", [x, up, ids]);                    // [T, S, H]
        var d = tg.B.Emit1("MatMulId", [h, down, ids]);                  // per-slot input -> [T, S, K]
        var h2 = tg.B.Emit1("MatMulId", [x, up, ids2]);
        tg.B.Output("d", d); tg.B.Output("h2", h2);
        // 80 items: the integer-dot expert matvec quantizes both blocks' activations to 8 bits; the second block's error
        // comes on top of the first's.
        Diff.Check("MatMulId.chain", tg.B.Build(), new Step(t, 0, (x, Rnd.F(t * K, 3)), (ids, Ids("random", t, S, E)), (ids2, Ids("few", t, S, E))),
            atol: 1.0, rtol: 5e-2, maxTokens: 128);
    }
}
