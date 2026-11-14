using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>NonZero on the GPU against the CPU oracle: coordinates in row-major order and the data-dependent count.</summary>
public sealed class NonZeroTests
{
    private static Value NonZero(TG tg, Value x)
    {
        string count = tg.B.DynamicSymbol("nonzero", x.Shape.Dims.Aggregate((Dim)1, (a, b) => a * b));
        return tg.B.Emit1("NonZero", [x], Attributes.From(("count_symbol", count)));
    }

    /// <summary>f32 values with zeros, negative zeros and NaN (non-zero) over ranks 1 to 3.</summary>
    [Theory]
    [InlineData(new long[] { 1000 })]
    [InlineData(new long[] { 37, 11 })]
    [InlineData(new long[] { 5, 7, 300 })]
    public void Floats(long[] dims)
    {
        var tg = new TG();
        var x = tg.InF("x", [.. dims.Select(d => (Dim)d)]);
        tg.B.Output("y", NonZero(tg, x));
        long n = dims.Aggregate(1L, (a, b) => a * b);
        var data = Rnd.F(n, 3);
        for (long i = 0; i < n; i++)
        {
            data[i] = (i % 7) switch { 0 => 0f, 1 => -0f, 2 when i % 5 == 0 => float.NaN, _ => data[i] };
        }

        Diff.Check("NonZero", tg.B.Build(), new Step(1, 0, (x, data)), atol: 0, rtol: 0, what: string.Join("x", dims));
    }

    /// <summary>A comparison result (bool) with a symbolic outer dim, and a consumer of the count in the next segment.</summary>
    [Theory]
    [InlineData(40, 0.5f)]
    [InlineData(64, 2f)]   // nothing selected
    public void Mask(int rows, float threshold)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, 30);
        var mask = tg.B.Binary("Greater", x, tg.Const("threshold", [threshold], 1));
        var selected = NonZero(tg, mask);
        tg.B.Output("y", selected);
        var column = tg.B.Emit1("Reshape", [tg.B.Slice(selected, 1, 1, 1)], Attributes.From(("shape", new long[] { 0 })));
        var table = tg.Const("table", [.. Enumerable.Range(0, 30).Select(i => i * 0.5f)], 30);
        tg.B.Output("picked", tg.B.Emit1("GatherAxis", [table, column], Attributes.From(("axis", 0L))));
        Diff.Check("NonZero", tg.B.Build(), new Step(rows, 0, (x, Rnd.F(rows * 30L, 5))), atol: 0, rtol: 0, maxTokens: 64, what: $"mask rows={rows} threshold={threshold}");
    }

    /// <summary>Integer input.</summary>
    [Fact]
    public void Integers()
    {
        var tg = new TG();
        var x = tg.In("x", DataType.I32, 9, 1000);
        tg.B.Output("y", NonZero(tg, x));
        Diff.Check("NonZero", tg.B.Build(), new Step(1, 0, (x, Rnd.I(9000, 7, 3))), atol: 0, rtol: 0, what: "i32");
    }
}
