using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>Group-wise 1-bit / ternary weight formats on the GPU against the CPU oracle (matmul, GEMV and gather).</summary>
public sealed class BonsaiQuantTests
{
    [Theory]
    [InlineData("q1_0", 1), InlineData("q1_0", 3), InlineData("q1_0", 40)]
    [InlineData("q2_0", 1), InlineData("q2_0", 3), InlineData("q2_0", 40)]
    [InlineData("q2_0_g128", 1), InlineData("q2_0_g128", 40)]
    [InlineData("tern1_g128", 1), InlineData("tern1_g128", 3), InlineData("tern1_g128", 40)]
    public void MatMul(string type, int tokens)
    {
        Gpu.Require();
        int K = 640;
        int N = 37;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var w = tg.RandW("w", DataType.Get(type), 1f, N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        Diff.Check("MatMul." + type, tg.B.Build(), new Step(tokens, 0, (x, Rnd.F((long)tokens * K, 3))), atol: 2e-3, rtol: 1e-3);
    }

    [Theory]
    [InlineData("q1_0"), InlineData("q2_0"), InlineData("q2_0_g128"), InlineData("tern1_g128")]
    public void Gather(string type)
    {
        Gpu.Require();
        int D = 256;
        int V = 11;
        var tg = new TG();
        var idx = tg.In("idx", DataType.I32, TG.T);
        var tbl = tg.RandW("tbl", DataType.Get(type), 1f, V, D);
        tg.B.Output("y", tg.B.Gather(tbl, idx));
        Diff.Check("Gather." + type, tg.B.Build(), new Step(5, 0, (idx, new[] { 3, 0, 10, 10, 7 })), atol: 1e-6, rtol: 1e-6);
    }
}
