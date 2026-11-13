using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>MatMulId (expert-indexed matmul): per-item matvec for few rows, grouped-by-expert GEMM for many.</summary>
public class MoeTests
{
    // Grouped (tensor-core / tiled) path above 16 (token, slot) items may round activations/weights to f16.
    private static double Atol(int items) => items > 16 ? 1.5e-1 : 2e-3;
    private static double Rtol(int items) => items > 16 ? 1e-2 : 1e-3;

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

    private static void Run(string fmt, int t, int s, bool perSlot, string pattern, int E = 8, int N = 96, int K = 512)
    {
        Gpu.Require();
        var tg = new TG();
        var x = perSlot ? tg.InF("x", TG.T, s, K) : tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, s);
        var w = tg.RandW("w", DataType.Get(fmt), 1f, E, N, K);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, w, ids]));
        int items = t * s;
        Diff.Check("MatMulId." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)(perSlot ? items : t) * K, 3)), (ids, Ids(pattern, t, s, E))),
            atol: Atol(items), rtol: Rtol(items), maxTokens: 128, what: $"T={t} S={s} perSlot={perSlot} ids={pattern}");
    }

    public static IEnumerable<object[]> FormatCases()
    {
        foreach (var fmt in new[] { "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k", "q5_k", "q6_k" })
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
    [InlineData(1, 1)] [InlineData(1, 2)] [InlineData(1, 8)] [InlineData(3, 2)] [InlineData(3, 8)] [InlineData(5, 4)] [InlineData(70, 1)] [InlineData(70, 2)] [InlineData(70, 8)]
    public void TopKValues(int t, int s) => Run("q4_k", t, s, false, "random");

    [Theory]
    [InlineData(1, 4)] [InlineData(3, 4)] [InlineData(70, 4)] [InlineData(70, 8)]
    public void PerSlotInput(int t, int s) => Run("q4_k", t, s, true, "random");

    [Theory]
    [InlineData("f16", 70, "few")] [InlineData("q4_k", 70, "few")] [InlineData("q4_k", 70, "one")] [InlineData("q6_k", 20, "few")] [InlineData("f32", 40, "one")]
    public void ExpertsWithoutTokens(string fmt, int t, string pattern) => Run(fmt, t, 2, false, pattern);

    [Theory]
    [InlineData(64, 24, 256)] [InlineData(16, 130, 256)] [InlineData(32, 33, 512)] [InlineData(4, 200, 288)]
    public void ShapeVariants(int E, int N, int K) => Run(K % 256 == 0 ? "q4_k" : "q8_0", 70, 3, false, "random", E, N, K);

    /// <summary>Two expert products on one ids tensor followed by a second block with other ids (routing is recomputed).</summary>
    [Fact]
    public void ChainedBlocks()
    {
        Gpu.Require();
        int E = 8, K = 256, H = 256, S = 2, t = 40;
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
        Diff.Check("MatMulId.chain", tg.B.Build(), new Step(t, 0, (x, Rnd.F(t * K, 3)), (ids, Ids("random", t, S, E)), (ids2, Ids("few", t, S, E))),
            atol: 0.15, rtol: 5e-2, maxTokens: 128);
    }
}
