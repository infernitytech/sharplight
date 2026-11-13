using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

public class PoolTests
{
    public static IEnumerable<object[]> Cases()
    {
        // name, mode, H, W, kernel, stride, pad(4), dil, ceil, includePad
        object[][] c =
        [
            ["max-3x3-s2-p1", "max", 14, 15, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, false, false],
            ["max-2x2-s2", "max", 8, 9, new[] { 2, 2 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, false, false],
            ["max-ceil-odd", "max", 7, 9, new[] { 2, 2 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, false],
            ["max-ceil-pad-drop", "max", 6, 6, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, true, false],
            ["max-ceil-s3", "max", 10, 11, new[] { 3, 3 }, new[] { 3, 3 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, false],
            ["max-dil2", "max", 13, 12, new[] { 3, 3 }, new[] { 1, 1 }, new[] { 2, 2, 2, 2 }, new[] { 2, 2 }, false, false],
            ["max-dil-ceil-asym", "max", 13, 12, new[] { 2, 3 }, new[] { 2, 2 }, new[] { 1, 0, 1, 2 }, new[] { 2, 1 }, true, false],
            ["avg-3x3-s2-p1-excl", "avg", 14, 15, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, false, false],
            ["avg-3x3-s2-p1-incl", "avg", 14, 15, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, false, true],
            ["avg-ceil-excl", "avg", 7, 9, new[] { 2, 2 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, false],
            ["avg-ceil-incl-pad", "avg", 6, 6, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1, 1, 1 }, new[] { 1, 1 }, true, true],
            ["avg-ceil-incl-nopad", "avg", 8, 8, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, true, true],
            ["avg-7x7", "avg", 7, 7, new[] { 7, 7 }, new[] { 1, 1 }, new[] { 0, 0, 0, 0 }, new[] { 1, 1 }, false, false],
        ];
        foreach (var o in c)
        {
            foreach (var n in new[] { 1, 3 })
            {
                yield return [.. o, n];
            }
        }
    }

    [Theory, MemberData(nameof(Cases))]
    public void Pool(string name, string mode, int H, int W, int[] k, int[] s, int[] pad, int[] dil, bool ceil, bool includePad, int N)
    {
        int C = 4;
        var x = Kit.Rand(N * C * H * W, 31);
        var tg = new TestGraph();
        var xin = tg.InF("x", SpatialKit.DSym(C, H, W));
        var attrs = Attributes.From(("mode", mode), ("kernel", k.Select(v => (long)v).ToArray()), ("strides", s.Select(v => (long)v).ToArray()),
            ("pads", pad.Select(v => (long)v).ToArray()), ("dilations", dil.Select(v => (long)v).ToArray()), ("ceil_mode", ceil), ("count_include_pad", includePad));
        tg.B.Output("y", tg.B.Emit1("Pool", [xin], attrs));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 3);
        Kit.Set(m, xin, x);
        var p = Kit.P(N);
        m.Run(p);
        var exp = SRef.Pool(x, N, C, H, W, mode == "max", k, s, pad, dil, ceil, includePad, out int ho, out int wo);
        var shape = m.Graph.Outputs["y"].Shape;
        Assert.Equal(ho, (int)shape[2].Fixed);
        Assert.Equal(wo, (int)shape[3].Fixed);
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-5, 1e-5, name);
    }
}
