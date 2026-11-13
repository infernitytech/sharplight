using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public class ConvTests
{
    private static void Run(string label, int[] batches, int C, int H, int W, int O, int KH, int KW, int[] stride, int[] pads, int[] dil, int groups,
        bool bias, string wtype = "f32", bool sliceInput = false, double atol = 2e-4, double rtol = 1e-4)
    {
        Gpu.Require();
        var tg = new TG();
        int Cin = sliceInput ? C + 5 : C;
        var x = tg.InF("x", TG.T, Cin, H, W);
        var xv = sliceInput ? tg.B.Slice(x, 1, 2, C) : x;
        var wt = DataType.Get(wtype);
        var w = tg.RandW("w", wt, 0.3f, O, C / groups, KH, KW);
        var args = new List<Value> { xv, w };
        if (bias)
        {
            args.Add(tg.RandW("b", DataType.F32, 1f, O));
        }

        var attrs = Attributes.From(("strides", stride.Select(v => (long)v).ToArray()), ("pads", pads.Select(v => (long)v).ToArray()),
            ("dilations", dil.Select(v => (long)v).ToArray()), ("groups", (long)groups));
        tg.B.Output("y", tg.B.Emit1("Conv", args, attrs));
        var g = tg.B.Build();
        var steps = batches.Select((n, i) => new Step(n, 0, (x, Rnd.F((long)n * Cin * H * W, 10 + i)))).ToList();
        Diff.Check("Conv", g, steps, atol, rtol, maxTokens: batches.Max(), what: label);
    }

    [Theory]
    [InlineData(1, 16, 14, 14, 32, 3, 3, 1, 1, 1)]
    [InlineData(2, 16, 14, 14, 32, 3, 3, 1, 1, 1)]
    [InlineData(1, 3, 32, 32, 16, 7, 7, 2, 3, 1)]       // stem, K=147 (non-vector weight load)
    [InlineData(1, 64, 13, 13, 96, 1, 1, 1, 0, 1)]
    [InlineData(3, 64, 13, 13, 96, 1, 1, 2, 0, 1)]
    [InlineData(1, 8, 17, 19, 12, 3, 3, 2, 1, 1)]
    [InlineData(2, 24, 10, 10, 40, 5, 5, 1, 2, 1)]
    [InlineData(1, 32, 9, 9, 8, 3, 3, 1, 0, 1)]
    public void Dense(int n, int c, int h, int w, int o, int kh, int kw, int s, int p, int g)
        => Run($"{c}->{o} {kh}x{kw} s{s} p{p} N{n}", [n], c, h, w, o, kh, kw, [s, s], [p, p, p, p], [1, 1], g, bias: true);

    [Fact] public void NoBias() => Run("nobias", [2], 12, 11, 11, 20, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, bias: false);
    [Fact] public void Dilated() => Run("dil2", [1], 8, 16, 16, 8, 3, 3, [1, 1], [2, 2, 2, 2], [2, 2], 1, bias: true);
    [Fact] public void DilatedAsymmetric() => Run("dil(2,3) s(1,2)", [2], 6, 15, 17, 10, 3, 2, [1, 2], [2, 3, 2, 3], [2, 3], 1, bias: true);
    [Fact] public void AsymmetricPads() => Run("pads 1,0,2,3", [1], 8, 12, 14, 16, 3, 3, [1, 1], [1, 0, 2, 3], [1, 1], 1, bias: true);
    [Fact] public void NonSquareKernel() => Run("1x7", [1], 16, 12, 20, 16, 1, 7, [1, 1], [0, 3, 0, 3], [1, 1], 1, bias: true);
    [Fact] public void Pointwise1x1Large() => Run("1x1 256->64", [1], 256, 28, 28, 64, 1, 1, [1, 1], [0, 0, 0, 0], [1, 1], 1, bias: true);
    [Fact] public void StridedInputView() => Run("sliced input", [2], 16, 9, 9, 24, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, bias: true, sliceInput: true);

    [Theory]
    [InlineData(32, 4)] [InlineData(64, 2)] [InlineData(16, 16)] [InlineData(12, 3)]
    public void Grouped(int c, int g) => Run($"groups {g}", [2], c, 12, 12, c * 2, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], g, bias: true);

    [Theory]
    [InlineData(32, 1, 1, 1)] [InlineData(32, 1, 2, 1)] [InlineData(16, 2, 1, 1)] [InlineData(24, 1, 1, 2)]
    public void Depthwise(int c, int mult, int stride, int dil)
        => Run($"dw c{c} m{mult} s{stride} d{dil}", [1, 2], c, 15, 15, c * mult, 3, 3, [stride, stride], [dil, dil, dil, dil], [dil, dil], c, bias: true);

    [Fact] public void Depthwise5x5NoBias() => Run("dw5x5", [2], 20, 14, 14, 20, 5, 5, [1, 1], [2, 2, 2, 2], [1, 1], 20, bias: false);
    [Fact] public void DepthwiseF16() => Run("dw f16", [1], 24, 14, 14, 24, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 24, bias: true, wtype: "f16");

    [Theory]
    [InlineData("f16")] [InlineData("bf16")] [InlineData("q8_0")] [InlineData("q4_0")]
    public void WeightFormats(string t) => Run("wfmt " + t, [2], 32, 10, 10, 48, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, bias: true, wtype: t, atol: 1e-3, rtol: 1e-3);

    [Theory]
    [InlineData("q4_k")] [InlineData("q5_k")] [InlineData("q6_k")]
    public void KQuantWeightFormats(string t) => Run("wfmt " + t, [1], 64, 8, 8, 40, 2, 2, [1, 1], [0, 0, 0, 0], [1, 1], 1, bias: false, wtype: t, atol: 1e-3, rtol: 1e-3);

    /// <summary>One compiled model run with different batch sizes exercises the per-binding kernel variants (incl. split-K).</summary>
    [Fact] public void BatchVariants() => Run("batch sweep", [1, 2, 3, 5, 8, 17], 64, 7, 7, 64, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, bias: true);
    [Fact] public void SplitKSmallSpatial() => Run("split-k", [1], 256, 7, 7, 128, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, bias: true, atol: 5e-4);
    [Fact] public void SplitK1x1() => Run("split-k 1x1", [1], 512, 7, 7, 192, 1, 1, [1, 1], [0, 0, 0, 0], [1, 1], 1, bias: true, atol: 5e-4);
    [Fact] public void SplitKNoBias() => Run("split-k nobias", [2], 128, 5, 5, 64, 3, 3, [1, 1], [1, 1, 1, 1], [1, 1], 1, bias: false, atol: 5e-4);
}
