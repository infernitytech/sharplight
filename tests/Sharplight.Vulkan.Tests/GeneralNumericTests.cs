using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class GeneralNumericTests
{
    [Theory]
    [InlineData("prod")]
    [InlineData("l1")]
    [InlineData("l2")]
    [InlineData("sumsquare")]
    [InlineData("logsumexp")]
    public void ReduceModes(string mode)
    {
        Gpu.Require();
        foreach (var (axes, keep) in new[] { (new long[] { 2, 3 }, true), (new long[] { 1 }, false) })
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, 6, 5, 4);
            tg.B.Output("y", tg.B.Emit1("Reduce", [x], Attributes.From(("mode", mode), ("axes", axes), ("keepdims", keep))));
            Diff.Check("Reduce." + mode, tg.B.Build(), new Step(2, 0, (x, Rnd.F(240, 1, mode == "prod" ? 1.2f : 3f))), atol: 1e-4, rtol: 1e-4, what: $"axes {string.Join(",", axes)}");
        }
    }

    [Fact]
    public void ReduceLogSumExpLargeAxis()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 5000);
        tg.B.Output("y", tg.B.Emit1("Reduce", [x], Attributes.From(("mode", "logsumexp"), ("axes", new long[] { 1 }), ("keepdims", false))));
        Diff.Check("Reduce.logsumexp.large", tg.B.Build(), new Step(2, 0, (x, Rnd.F(10000, 1, 20f))), atol: 1e-4, rtol: 1e-5);
    }

    [Fact]
    public void LogSoftmax()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 3, 50);
        tg.B.Output("y", tg.B.Emit1("Softmax", [x], Attributes.From(("log", true), ("scale", 0.5f))));
        Diff.Check("Softmax.log", tg.B.Build(), new Step(4, 0, (x, Rnd.F(600, 1, 8f))), atol: 1e-5, rtol: 1e-5);
    }

    [Theory]
    [InlineData(4, 1, 6, true, true)]
    [InlineData(4, 4, 6, true, true)]
    [InlineData(12, 3, 5, true, false)]
    [InlineData(8, 2, 7, false, true)]
    [InlineData(32, 8, 3, false, false)]
    public void GroupNorm(int c, int groups, int s, bool scale, bool bias)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, s, 3);
        var args = new List<Value> { x };
        if (scale)
        {
            args.Add(tg.RandW("s", DataType.F32, 1f, c));
        }

        if (bias)
        {
            args.Add(tg.RandW("b", DataType.F32, 1f, c));
        }

        tg.B.Output("y", tg.B.Emit1("GroupNorm", args, Attributes.From(("groups", (long)groups), ("eps", 1e-5f))));
        Diff.Check("GroupNorm", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * c * s * 3, 1, 2f))), atol: 5e-5, rtol: 1e-4, what: $"c={c} g={groups}");
    }

    [Theory]
    [InlineData(1, 8, 5, 5, 6, 3, 1, 0, 0, 1, 1)]
    [InlineData(2, 8, 5, 6, 6, 3, 2, 1, 1, 1, 1)]
    [InlineData(1, 4, 4, 4, 4, 4, 2, 1, 0, 1, 1)]
    [InlineData(1, 6, 5, 5, 6, 3, 1, 1, 0, 2, 3)]
    [InlineData(2, 8, 4, 4, 8, 2, 2, 0, 0, 1, 4)]
    [InlineData(1, 4, 5, 7, 4, 3, 3, 2, 2, 1, 1)]
    public void ConvTranspose(int n, int c, int h, int w, int o, int k, int stride, int pad, int outPad, int dil, int groups)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, h, w);
        var wt = tg.RandW("w", DataType.F32, 0.4f, c, o / groups, k, k);
        var b = tg.RandW("b", DataType.F32, 1f, o);
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", [x, wt, b], Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }),
            ("dilations", new long[] { dil, dil }), ("groups", (long)groups), ("output_padding", new long[] { outPad, outPad }))));
        Diff.Check("ConvTranspose", tg.B.Build(), new Step(n, 0, (x, Rnd.F(n * c * h * w, 1))), atol: 1e-4, rtol: 1e-4, what: $"k{k} s{stride} p{pad} op{outPad} d{dil} g{groups}");
    }

    [Fact]
    public void ConvTransposeNoBiasF16Weights()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 6, 5, 5);
        var wt = tg.RandW("w", DataType.F16, 0.4f, 6, 4, 3, 3);
        tg.B.Output("y", tg.B.Emit1("ConvTranspose", [x, wt], Attributes.From(("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }))));
        Diff.Check("ConvTranspose.f16", tg.B.Build(), new Step(2, 0, (x, Rnd.F(300, 1))), atol: 1e-4, rtol: 1e-4);
    }

    [Theory]
    [InlineData("linear", "zeros", false)]
    [InlineData("linear", "zeros", true)]
    [InlineData("linear", "border", false)]
    [InlineData("linear", "reflection", true)]
    [InlineData("linear", "reflection", false)]
    [InlineData("nearest", "zeros", false)]
    [InlineData("nearest", "border", true)]
    [InlineData("cubic", "zeros", false)]
    [InlineData("cubic", "border", true)]
    public void GridSample(string mode, string padding, bool align)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 3, 6, 7);
        var grid = tg.InF("g", TG.T, 4, 5, 2);
        tg.B.Output("y", tg.B.Emit1("GridSample", [x, grid], Attributes.From(("mode", mode), ("padding", padding), ("align_corners", align))));
        Diff.Check("GridSample", tg.B.Build(), new Step(2, 0, (x, Rnd.F(2 * 126, 1)), (grid, Rnd.F(2 * 40, 2, 1.4f))), atol: 2e-4, rtol: 1e-4, what: $"{mode}/{padding}/align={align}");
    }
}
