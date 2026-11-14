namespace Sharplight.Cpu.Tests;

/// <summary>
/// MatMul output epilogues (bias vector over the last axis, residual Add, activation) on the GEMM and the row paths,
/// against the unfused reference, with and without prepacked static weights.
/// </summary>
public class MatMulEpilogueTests
{
    [Theory]
    [InlineData(3)]     // float rows: epilogue pass after the product
    [InlineData(37)]    // GEMM with edge tiles: epilogue per tile
    public void BiasResidualActivation(int tokens)
    {
        const int K = 64;
        const int N = 40;
        var x = Kit.Rand(tokens * K, 41);
        var w = Kit.Rand(N * K, 42, 0.2f);
        var b = Kit.Rand(N, 43);
        var r = Kit.Rand(tokens * N, 44);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, K);
        var rin = tg.InF("r", TestGraph.T, N);
        var wv = tg.WF("w", w, N, K);
        var bv = tg.WF("b", b, N);
        var brow = tg.WF("brow", b, 1, N);
        tg.B.Output("full", tg.B.Unary("Relu", tg.B.Add(rin, tg.B.Add(tg.B.MatMul(xin, wv), bv))));
        tg.B.Output("bias", tg.B.Add(brow, tg.B.MatMul(xin, wv)));
        tg.B.Output("act", tg.B.Unary("Tanh", tg.B.MatMul(xin, wv)));
        var shared = tg.B.MatMul(xin, wv);
        var sharedBias = tg.B.Add(shared, bv);
        tg.B.Output("shared", tg.B.Add(sharedBias, shared));   // the product feeds two nodes: nothing is fused
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 64);
        Kit.Set(m, xin, x);
        Kit.Set(m, rin, r);
        var p = Kit.P(tokens);
        m.Run(p);
        m.Run(p);   // second run uses the prepacked weight
        var mm = Ref.MatMul(x, tokens, K, w, N);
        Kit.AssertClose(mm.Select((v, i) => Math.Max(0f, v + b[i % N] + r[i])).ToArray(), Kit.Out(m, "full", p), 1e-4, 1e-5, "full");
        Kit.AssertClose(mm.Select((v, i) => v + b[i % N]).ToArray(), Kit.Out(m, "bias", p), 1e-4, 1e-5, "bias");
        Kit.AssertClose(mm.Select(MathF.Tanh).ToArray(), Kit.Out(m, "act", p), 1e-4, 1e-5, "act");
        Kit.AssertClose(mm.Select((v, i) => 2 * v + b[i % N]).ToArray(), Kit.Out(m, "shared", p), 1e-4, 1e-5, "shared");
    }
}
