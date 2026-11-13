using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class CastAndIntegerFlowTests
{
    [Fact]
    public void CastBetweenTypes()
    {
        float[] x = [0f, 1f, -1f, 2.7f, -2.7f, 1000f, 65504f, 0.5f];
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, 4);
        foreach (var t in new[] { "f16", "i32", "i64", "bool" })
        {
            tg.B.Output(t, tg.B.Emit1("Cast", [xin], GeneralKit.A(("type", t))));
        }

        var f16 = tg.B.Emit1("Cast", [xin], GeneralKit.A(("type", "f16")));
        tg.B.Output("f16_back", tg.B.Emit1("Cast", [f16], GeneralKit.A(("type", "f32"))));
        tg.B.Output("i64_back", tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [xin], GeneralKit.A(("type", "i64")))], GeneralKit.A(("type", "f32"))));
        tg.B.Output("i32_i64", tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [xin], GeneralKit.A(("type", "i32")))], GeneralKit.A(("type", "i64"))));
        tg.B.Output("bool_back", tg.B.Emit1("Cast", [tg.B.Emit1("Cast", [xin], GeneralKit.A(("type", "bool")))], GeneralKit.A(("type", "f32"))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        var p = Kit.P(2);
        m.Run(p);
        Kit.AssertClose(x.Select(v => (float)(Half)v).ToArray(), Kit.Out(m, "f16_back", p), 0, 0);
        Kit.AssertClose(x.Select(v => (float)(long)v).ToArray(), Kit.Out(m, "i64_back", p), 0, 0);
        Assert.Equal(x.Select(v => (long)(int)v).ToArray(), GeneralKit.OutL(m, "i32_i64", p));
        Kit.AssertClose(x.Select(v => v != 0 ? 1f : 0f).ToArray(), Kit.Out(m, "bool_back", p), 0, 0);
        Assert.Equal(x.Select(v => (int)v).ToArray(), Kit.OutI(m, "i32", p));
    }

    [Fact]
    public void IntegerTensorsFlowThroughViewsAndData()
    {
        // i64 shape-like tensor: reshape -> slice -> concat -> gather -> expand
        var tg = new TestGraph();
        var x = tg.In("x", DataType.I64, 2, 6);
        var r = tg.B.Reshape(x, 3, 4);                       // [3,4]
        var s1 = tg.B.Slice(r, 1, 1, 2);
        var s2 = tg.B.Slice(r, 1, 0, 2);
        var cc = tg.B.Concat(1, s1, s2);                     // [3,4]
        var iv = tg.In("i", DataType.I32, 2);
        var g = tg.B.Emit1("GatherAxis", [cc, iv], GeneralKit.A(("axis", 0L)));   // [2,4]
        tg.B.Output("g", g);
        tg.B.Output("e", tg.B.Emit1("Expand", [tg.B.Slice(g, 0, 0, 1)], GeneralKit.A(("shape", new long[] { 3, 4 }))));
        tg.B.Output("perm", tg.B.Contiguous(tg.B.Permute(cc, 1, 0)));
        using var m = Kit.Compile(tg.B.Build());
        long[] data = Enumerable.Range(0, 12).Select(i => (long)i * 3_000_000_000L).ToArray();
        GeneralKit.SetI64(m, x, data);
        Kit.Set(m, iv, new[] { 2, 0 });
        var p = Kit.P(1);
        m.Run(p);
        var cat = new long[12];
        for (int i = 0; i < 3; i++)
        {
            cat[i * 4] = data[i * 4 + 1];
            cat[i * 4 + 1] = data[i * 4 + 2];
            cat[i * 4 + 2] = data[i * 4];
            cat[i * 4 + 3] = data[i * 4 + 1];
        }
        var gexp = new long[8];
        for (int j = 0; j < 4; j++)
        {
            gexp[j] = cat[2 * 4 + j];
            gexp[4 + j] = cat[j];
        }
        Assert.Equal(gexp, GeneralKit.OutL(m, "g", p));
        Assert.Equal(Enumerable.Range(0, 12).Select(i => gexp[i % 4]).ToArray(), GeneralKit.OutL(m, "e", p));
        var perm = new long[12];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                perm[j * 3 + i] = cat[i * 4 + j];
            }
        }

        Assert.Equal(perm, GeneralKit.OutL(m, "perm", p));
    }
}
