using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Convolutions large enough for the cooperative-matrix implicit GEMM (conv_cm): bf16x3 products under strict
/// precision must match the f32 CPU oracle tightly (also for values far outside the f16 range), the f16 form under
/// reduced precision to about 1e-2. Devices without matrix features run the SIMT conv for the same graphs.
/// </summary>
public sealed class ConvMatrixTests
{
    private static ComputeGraph Graph(TG tg, Value x, int c, int o, int k, int stride, int pad, int groups, string weightType, bool bias)
    {
        var w = tg.RandW("w", DataType.Get(weightType), 0.1f, o, c / groups, k, k);
        var args = new List<Value> { x, w };
        if (bias)
        {
            args.Add(tg.RandW("b", DataType.F32, 1f, o));
        }

        var attrs = Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }),
            ("dilations", new long[] { 1, 1 }), ("groups", (long)groups));
        tg.B.Output("y", tg.B.Emit1("Conv", args, attrs));
        return tg.B.Build();
    }

    [Theory]
    [InlineData(64, 20, 96, 3, 1, 1, 1, "f32", true)]     // K = 576, split-K
    [InlineData(7, 24, 64, 5, 1, 2, 1, "f32", true)]      // K = 175: scalar weight loads, partial K step
    [InlineData(64, 17, 64, 3, 2, 1, 2, "f32", false)]    // groups, stride 2
    [InlineData(256, 14, 128, 1, 1, 0, 1, "f32", true)]   // 1x1
    [InlineData(32, 33, 48, 3, 1, 1, 1, "f16", true)]     // f16 weights, partial channel tile
    [InlineData(256, 7, 128, 3, 1, 1, 1, "f32", true)]    // 7x7 map: 49 pixels, less than one 64-pixel tile
    public void StrictMatchesF32(int c, int hw, int o, int k, int stride, int pad, int groups, string weightType, bool bias)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, hw, hw);
        var g = Graph(tg, x, c, o, k, stride, pad, groups, weightType, bias);
        int[] batches = [1, 2];
        var steps = batches.Select((n, i) => new Step(n, 0, (x, Rnd.F((long)n * c * hw * hw, 40 + i)))).ToArray();
        Diff.Check("Conv.matrix", g, steps, atol: 2e-5, rtol: 1e-4, maxTokens: 2, what: $"C={c} {hw}x{hw} O={o} k{k} s{stride} g{groups} {weightType}");
    }

    /// <summary>Input as a channel slice of a larger tensor (strided batch and channel offsets).</summary>
    [Theory]
    [InlineData(1, 1, 0)]
    [InlineData(3, 2, 1)]
    public void StrictChannelSliceInput(int k, int stride, int pad)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, 96, 20, 20);
        var xs = tg.B.Slice(x, 1, 32, 64);
        var g = Graph(tg, xs, 64, 64, k, stride, pad, 1, "f32", true);
        int[] batches = [1, 2];
        var steps = batches.Select((n, i) => new Step(n, 0, (x, Rnd.F((long)n * 96 * 400, 60 + i)))).ToArray();
        Diff.Check("Conv.matrix.slice", g, steps, atol: 2e-5, rtol: 1e-4, maxTokens: 2, what: $"k{k} s{stride}");
    }

    /// <summary>
    /// Activations around 1e6 (beyond the f16 range) stay accurate with the bf16 split: about 16 mantissa bits per product
    /// (TF32 has 10), so errors relative to the summed term magnitudes (~1e7 here) stay near 1e-6.
    /// </summary>
    [Fact]
    public void StrictLargeMagnitudes()
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, 64, 16, 16);
        var g = Graph(tg, x, 64, 64, 3, 1, 1, 1, "f32", false);
        var data = Rnd.F(64 * 16 * 16, 50, 1e6f);
        Diff.Check("Conv.matrix.large", g, new Step(1, 0, (x, data)), atol: 20, rtol: 1e-4, maxTokens: 1);
    }

    /// <summary>Reduced precision: 128-pixel tiles for large maps, 64-pixel tiles for small grids, a 7x7 map below one tile.</summary>
    [Theory]
    [InlineData(64, 20, 96)]
    [InlineData(64, 56, 64)]
    [InlineData(256, 7, 128)]
    public void ReducedPrecisionF16(int c, int hw, int o)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, c, hw, hw);
        var g = Graph(tg, x, c, o, 3, 1, 1, 1, "f32", true);
        Diff.Check("Conv.matrix.f16", g, new Step(1, 0, (x, Rnd.F((long)c * hw * hw, 51))), atol: 2e-2, rtol: 1e-2, maxTokens: 1,
            allowReducedPrecision: true, what: $"C={c} {hw}x{hw} O={o}");
    }
}
