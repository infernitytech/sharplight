using Sharplight.Graph;

namespace Sharplight.Vulkan.Tests;

/// <summary>Max-pooling argmax indices (ONNX MaxPool's Indices output) on the GPU against the CPU oracle.</summary>
public sealed class PoolArgMaxTests
{
    [Theory]
    [InlineData(2, 6, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, false, 0, false)]   // padding and strides
    [InlineData(1, 4, 16, 16, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, false, 1, false)]   // storage_order 1 (column-major indices)
    [InlineData(1, 3, 17, 17, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, false, 0, false)]   // dilation
    [InlineData(2, 5, 13, 17, 3, 2, 2, 1, 1, 0, 2, 1, 1, 1, true, 1, false)]    // asymmetric pads, ceil mode
    [InlineData(1, 4, 12, 12, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, false, 0, true)]    // ties: the first maximum wins
    public void Indices(int n, int c, int h, int w, int kh, int kw, int sh, int sw, int top, int left, int bottom, int right, int dh, int dw, bool ceil,
        long storageOrder, bool ties)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, c + 1, h, w);
        var input = tg.B.Slice(x, 1, 1, c);   // strided input
        var attributes = Attributes.From(("kernel", new long[] { kh, kw }), ("strides", new long[] { sh, sw }), ("pads", new long[] { top, left, bottom, right }),
            ("dilations", new long[] { dh, dw }), ("ceil_mode", ceil), ("storage_order", storageOrder));
        tg.B.Output("indices", tg.B.Emit1("PoolArgMax", [input], attributes));
        tg.B.Output("values", tg.B.Emit1("Pool", [input], attributes.With("mode", "max")));
        var data = Rnd.F((long)n * (c + 1) * h * w, 7);
        if (ties)
        {
            data = [.. data.Select(v => MathF.Round(v * 2) / 2)];
        }

        Diff.Check("PoolArgMax", tg.B.Build(), new Step(n, 0, (x, data)), atol: 0, rtol: 0, what: $"k{kh}x{kw} s{sh}x{sw} d{dh}x{dw} order={storageOrder} ties={ties}");
    }
}
