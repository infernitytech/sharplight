using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class CastAndIntegerTests
{
    [Theory]
    [InlineData("f16")]
    [InlineData("bf16")]
    [InlineData("i32")]
    [InlineData("i64")]
    public void CastRoundTrip(string t)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, 19);
        var c = tg.B.Emit1("Cast", [x], Attributes.From(("type", t)));
        tg.B.Output("back", tg.B.Emit1("Cast", [c], Attributes.From(("type", "f32"))));
        if (t == "i32" || t == "i64")
        {
            tg.B.Output("raw", c);
        }

        var data = Rnd.F(3 * 19, 5, 100f);
        data[0] = -7.9f;
        data[1] = 7.9f;
        data[2] = 0.5f;
        Diff.Check("Cast." + t, tg.B.Build(), new Step(3, 0, (x, data)), atol: 0, rtol: 0);
    }

    [Fact]
    public void CastBetweenIntegerTypes()
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.In("x", DataType.I64, TG.T, 5);
        var i = tg.B.Emit1("Cast", [x], Attributes.From(("type", "i32")));
        tg.B.Output("i32", i);
        tg.B.Output("f", tg.B.Emit1("Cast", [i], Attributes.From(("type", "f32"))));
        tg.B.Output("i64", tg.B.Emit1("Cast", [i], Attributes.From(("type", "i64"))));
        Diff.Check("Cast.int", tg.B.Build(), new Step(3, 0, (x, Enumerable.Range(-7, 15).Select(v => (long)v * 3).ToArray())), atol: 0, rtol: 0);
    }

    [Theory]
    [InlineData("i32")]
    [InlineData("i64")]
    public void IntegerTensorsThroughViewsConcatGatherWhereExpand(string t)
    {
        Gpu.Require();
        var dt = DataType.Get(t);
        var tg = new TG();
        var x = tg.In("x", dt, TG.T, 6);
        var y = tg.In("y", dt, TG.T, 6);
        var s = tg.B.Slice(x, 1, 1, 4);                             // [T,4] strided
        var pm = tg.B.Permute(x, 0, 1);
        tg.B.Output("slice", s);                                     // view as graph output
        tg.B.Output("cat", tg.B.Concat(1, s, tg.B.Slice(y, 1, 0, 2)));
        tg.B.Output("contig", tg.B.Contiguous(tg.B.Slice(y, 1, 2, 3)));
        var cond = tg.B.Binary("Greater", tg.B.Emit1("Cast", [x], Attributes.From(("type", "f32"))), tg.Const("zero", [0f], 1));
        tg.B.Output("where", tg.B.Emit1("Where", [cond, x, y]));
        tg.B.Output("expand", tg.B.Emit1("Expand", [tg.B.Slice(x, 1, 0, 1)], Attributes.From(("shape", new long[] { 1, 7 }))));
        var idx = tg.In("idx", DataType.I32, 3);
        tg.B.Output("gather", tg.B.Emit1("GatherAxis", [x, idx], Attributes.From(("axis", 1L))));
        Array Ints(int n, int seed) => t == "i32" ? Rnd.I(n, seed, 50).Select(v => v - 25).ToArray() : Rnd.I(n, seed, 50).Select(v => (long)v - 25).ToArray();
        Diff.Check("Integer.flow." + t, tg.B.Build(), new Step(4, 0, (x, Ints(24, 1)), (y, Ints(24, 2)), (idx, new[] { 5, 0, 3 })), atol: 0, rtol: 0);
    }
}
