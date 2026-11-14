using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>IQ4_NL / IQ4_XS and the legacy Q4_1 / Q5_0 / Q5_1 weights on the GPU against the CPU oracle (matmul, GEMV and gather).</summary>
public sealed class Iq4QuantTests
{
    [Theory]
    [InlineData("iq4_nl", 1)]
    [InlineData("iq4_nl", 3)]
    [InlineData("iq4_nl", 40)]
    [InlineData("iq4_xs", 1)]
    [InlineData("iq4_xs", 3)]
    [InlineData("iq4_xs", 40)]
    [InlineData("q4_1", 1)]
    [InlineData("q4_1", 40)]
    [InlineData("q5_0", 1)]
    [InlineData("q5_0", 3)]
    [InlineData("q5_0", 40)]
    [InlineData("q5_1", 1)]
    [InlineData("q5_1", 40)]
    public void MatMul(string type, int tokens)
    {
        Gpu.Require();
        int K = 768;
        int N = 37;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var w = tg.RandW("w", DataType.Get(type), 1f, N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        // f16 tensor-core path for T > 1: random 6-bit scales span a wide weight range. From 33 rows IQ4_NL / IQ4_XS run on the
        // int8 GEMM (q8_1 activations), with the bound of MmqTests.
        double atol = tokens >= 33 && type is "iq4_nl" or "iq4_xs" ? 3.0 : 0.25;
        Diff.Check("MatMul." + type, tg.B.Build(), new Step(tokens, 0, (x, Rnd.F((long)tokens * K, 3))), atol: atol, rtol: 1e-2);
    }

    [Theory]
    [InlineData("iq4_nl")]
    [InlineData("iq4_xs")]
    [InlineData("q4_1")]
    [InlineData("q5_0")]
    [InlineData("q5_1")]
    public void Gather(string type)
    {
        Gpu.Require();
        int D = 512;
        int V = 11;
        var tg = new TG();
        var idx = tg.In("idx", DataType.I32, TG.T);
        var tbl = tg.RandW("tbl", DataType.Get(type), 1f, V, D);
        tg.B.Output("y", tg.B.Gather(tbl, idx));
        Diff.Check("Gather." + type, tg.B.Build(), new Step(5, 0, (idx, new[] { 3, 0, 10, 10, 7 })), atol: 1e-6, rtol: 1e-6);
    }
}
