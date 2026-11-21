using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Elementwise groups consuming a weight matmul (bias, GELU, per-column scale, residual) applied in the split-f16 GEMM's store
/// (<see cref="Shaders.Lowering.GraphLowerer"/> matmul epilogues), on the plain and split-K GEMMs, and as a separate launch
/// after the matvec for a few rows, all against the CPU.
/// </summary>
public sealed class MatMulEpilogueTests
{
    /// <summary>y = Gelu(x·Wᵀ + b) (a transformer MLP's first projection) for matvec, split-K and plain GEMM row counts.</summary>
    [Theory]
    [InlineData(768, 3072)]
    [InlineData(3072, 768)]
    [InlineData(96, 384)]
    public void BiasGelu(int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var w = tg.Const("w", Rnd.F((long)n * k, 2, 0.03f), n, k);
        var bias = tg.Const("b", Rnd.F(n, 3, 0.5f), n);
        tg.B.Output("y", tg.B.Unary("Gelu", tg.B.Add(tg.B.MatMul(x, w), bias)));
        var g = tg.B.Build();
        Step Rows(int t) => new(t, 0, (x, Rnd.F((long)t * k, 1 + t)));
        Diff.Check("MatMul+bias+Gelu", g, [Rows(4), Rows(300), Rows(1576)], atol: 2e-5, rtol: 2e-5, maxTokens: 1576, what: $"{k}->{n}");
    }

    /// <summary>
    /// q / k / v = x·Wᵀ + b with f16 weights reading the same rows (gpt-oss's attention projections, one multi-output matmul) and a
    /// 32-wide router with its bias, for single-row, skinny, split-K and plain GEMM row counts: the biases stay in the split-K
    /// reduction for these weights (a change that ran them as their own launch broke gpt-oss-20b on CUDA).
    /// </summary>
    [Fact]
    public void HalfWeightsBias()
    {
        const int k = 2880;
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        foreach (var (name, n) in new[] { ("q", 1024), ("k", 128), ("v", 128), ("router", 32) })
        {
            var w = tg.RandW("w" + name, DataType.F16, 0.03f, n, k);
            var bias = tg.Const("b" + name, Rnd.F(n, 3 + n, 0.5f), n);
            tg.B.Output(name, tg.B.Add(tg.B.MatMul(x, w), bias));
        }

        var g = tg.B.Build();
        Step Rows(int t) => new(t, 0, (x, Rnd.F((long)t * k, 1 + t)));
        Diff.Check("MatMul(f16) q/k/v/router + bias", g, [Rows(1), Rows(5), Rows(40), Rows(300)], atol: 1e-3, rtol: 1e-3, maxTokens: 300);
    }

    /// <summary>y = (x·Wᵀ + b)·γ + r (a ConvNeXt block's last projection with its layer scale and the residual).</summary>
    [Theory]
    [InlineData(1536, 384)]
    [InlineData(384, 96)]
    public void BiasScaleResidual(int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var r = tg.InF("r", TG.T, n);
        var w = tg.Const("w", Rnd.F((long)n * k, 2, 0.03f), n, k);
        var bias = tg.Const("b", Rnd.F(n, 3, 0.5f), n);
        var gamma = tg.Const("gamma", Rnd.F(n, 4, 0.1f), n);
        tg.B.Output("y", tg.B.Add(tg.B.Mul(tg.B.Add(tg.B.MatMul(x, w), bias), gamma), r));
        var g = tg.B.Build();
        Step Rows(int t) => new(t, 0, (x, Rnd.F((long)t * k, 1 + t)), (r, Rnd.F((long)t * n, 7 + t)));
        Diff.Check("MatMul+bias*gamma+residual", g, [Rows(3), Rows(197), Rows(2000)], atol: 2e-5, rtol: 2e-5, maxTokens: 2000, what: $"{k}->{n}");
    }
}
