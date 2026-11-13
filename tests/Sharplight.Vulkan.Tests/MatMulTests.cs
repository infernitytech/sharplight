using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class MatMulTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in new[] { "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k", "q5_k", "q6_k" })
        {
            foreach (var t in new[] { 1, 2, 8, 9, 20, 40, 64 })
            {
                yield return [fmt, t];
            }
        }
    }

    // The T > 8 path may use reduced-precision (tensor core) products; allow ~1% there, tight bounds for the matvec path.
    private static double Atol(int t) => t > 8 ? 1.5e-1 : 2e-3;
    private static double Rtol(int t) => t > 8 ? 1e-2 : 1e-3;

    // Formats whose matvec may run on the integer-dot path. Its activations are quantized to 8 bits per 32-value block,
    // an absolute error proportional to the weight row's norm (large for these random weights: K-quant rows with
    // offsets up to ~10 and Q6_K values up to ~50), not to the output's own size. IntDotExactActivations checks the
    // arithmetic itself tightly.
    private static readonly HashSet<string> s_intDotFormats = ["q8_0", "q4_0", "q4_k", "q5_k", "q6_k"];

    private static double Atol(string fmt, int t) => s_intDotFormats.Contains(fmt) ? 3.0 : Atol(t);
    private static double Rtol(string fmt, int t) => s_intDotFormats.Contains(fmt) ? Rtol(9) : Rtol(t);

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
        Diff.Check("MatMul." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(fmt, t), rtol: Rtol(fmt, t), what: $"T={t}");
    }

    /// <summary>
    /// Activations that 8-bit block quantization represents exactly (integers times a power of two, block maximum 127):
    /// the integer-dot matvec must then match the f32 reference as tightly as the f32 matvec.
    /// </summary>
    [Theory]
    [InlineData("q8_0", 1)]
    [InlineData("q8_0", 3)]
    [InlineData("q4_k", 1)]
    [InlineData("q4_k", 5)]
    [InlineData("q5_k", 2)]
    [InlineData("q5_k", 8)]
    [InlineData("q6_k", 1)]
    [InlineData("q6_k", 7)]
    [InlineData("q4_0", 1)]
    [InlineData("q4_0", 6)]
    [InlineData("q5_0", 1)]
    [InlineData("q5_0", 4)]
    [InlineData("iq4_nl", 2)]
    [InlineData("mxfp4", 1)]
    [InlineData("mxfp4", 8)]
    public void IntDotExactActivations(string fmt, int t)
    {
        Gpu.Require();
        int K = 1024;
        int N = 72;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        Diff.Check("MatMul.exact8." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.ExactInt8((long)t * K, 5))), atol: 2e-3, rtol: 1e-3, what: $"T={t}");
    }

    [Theory]
    [InlineData("f32", 288, 100)]
    [InlineData("f16", 160, 33)]
    [InlineData("q8_0", 96, 130)]
    [InlineData("q4_0", 64, 7)]
    [InlineData("bf16", 40, 17)]
    [InlineData("f32", 16, 8)]
    [InlineData("f16", 64, 8)]
    [InlineData("f32", 24, 40)]
    public void OddShapes(string fmt, int K, int N)
    {
        Gpu.Require();
        foreach (var t in new[] { 1, 5, 17 })
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
            tg.B.Output("y", tg.B.MatMul(x, w));
            Diff.Check("MatMul.odd." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(fmt, t), rtol: Rtol(fmt, t), what: $"T={t} K={K} N={N}");
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(12)]
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
