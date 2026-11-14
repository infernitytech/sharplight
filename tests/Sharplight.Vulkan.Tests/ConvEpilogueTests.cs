using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Elementwise consumers of a convolution computed in its store (conv epilogue fusion): an activation alone, and a
/// residual Add followed by an activation, on every conv kernel (cooperative-matrix and SIMT implicit GEMM with and
/// without split-K, depthwise).
/// </summary>
public sealed class ConvEpilogueTests
{
    [Theory]
    [InlineData(64, 20, 96, 3, 1, 1, true)]    // conv_cm (SIMT conv_gemm without matrix features), split-K reduce applies the epilogue
    [InlineData(256, 14, 128, 1, 1, 1, false)] // 1x1, activation only
    [InlineData(7, 24, 64, 5, 1, 1, true)]     // small K: SIMT conv_gemm
    [InlineData(32, 17, 32, 3, 2, 32, true)]   // depthwise, stride 2
    public void ResidualAddAndActivation(int c, int hw, int o, int k, int stride, int groups, bool residual)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, hw, hw);
        var w = tg.RandW("w", DataType.F32, 0.1f, o, c / groups, k, k);
        var bias = tg.RandW("b", DataType.F32, 1f, o);
        int pad = k / 2;
        var attrs = Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }),
            ("dilations", new long[] { 1, 1 }), ("groups", (long)groups));
        var y = tg.B.Emit1("Conv", [x, w, bias], attrs);
        int ho = (hw + 2 * pad - k) / stride + 1;
        var r = tg.InF("r", TG.T, o, ho, ho);
        if (residual)
        {
            y = tg.B.Add(y, r);
        }

        tg.B.Output("y", tg.B.Unary(residual ? "Relu" : "Silu", y));
        var g = tg.B.Build();
        int[] batches = [1, 2];
        var steps = batches.Select((n, i) => new Step(n, 0, (x, Rnd.F((long)n * c * hw * hw, 70 + i)), (r, Rnd.F((long)n * o * ho * ho, 80 + i)))).ToArray();
        Diff.Check("Conv.epilogue", g, steps, atol: 2e-5, rtol: 1e-4, maxTokens: 2, what: $"C={c} {hw}x{hw} O={o} k{k} s{stride} g{groups}");
    }
}
