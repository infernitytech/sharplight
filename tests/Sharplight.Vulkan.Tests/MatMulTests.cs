using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class MatMulTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in new[] { "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k", "q5_k", "q6_k" })
        {
            foreach (var t in new[] { 1, 2, 8, 9, 20, 64 })
            {
                yield return [fmt, t];
            }
        }
    }

    // The T > 8 path may use reduced-precision (tensor core) products; allow ~1% there, tight bounds for the matvec path.
    private static double Atol(int t) => t > 8 ? 1.5e-1 : 2e-3;
    private static double Rtol(int t) => t > 8 ? 1e-2 : 1e-3;

    [Theory]
    [MemberData(nameof(Cases))]
    public void WeightFormat(string fmt, int t)
    {
        Gpu.Require();
        int K = 512;
        int N = 96;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        Diff.Check("MatMul." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(t), rtol: Rtol(t), what: $"T={t}");
    }

    [Theory]
    [InlineData("f32", 288, 100)] [InlineData("f16", 160, 33)] [InlineData("q8_0", 96, 130)] [InlineData("q4_0", 64, 7)] [InlineData("bf16", 40, 17)] [InlineData("f32", 16, 8)] [InlineData("f16", 64, 8)] [InlineData("f32", 24, 40)]
    public void OddShapes(string fmt, int K, int N)
    {
        Gpu.Require();
        foreach (var t in new[] { 1, 5, 17 })
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
            tg.B.Output("y", tg.B.MatMul(x, w));
            Diff.Check("MatMul.odd." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(t), rtol: Rtol(t), what: $"T={t} K={K} N={N}");
        }
    }

    [Theory] [InlineData(1)] [InlineData(12)]
    public void StridedAndHigherRankInputs(int t)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 8, 72);
        var xs = tg.B.Slice(x, 2, 8, 64);                      // [T,8,64] strided view, rows are not dense
        var w = tg.RandW("w", DataType.F32, 0.5f, 40, 64);
        tg.B.Output("y", tg.B.MatMul(xs, w));
        var xp = tg.B.Permute(x, 0, 2, 1);                     // [T,72,8]: inner axis not contiguous -> dense temp
        var w2 = tg.RandW("w2", DataType.F16, 0.5f, 24, 8);
        tg.B.Output("y2", tg.B.MatMul(xp, w2));
        Diff.Check("MatMul.strided", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * 576, 3))), atol: Atol(t * 8), rtol: Rtol(t * 8), what: $"T={t}");
    }
}
