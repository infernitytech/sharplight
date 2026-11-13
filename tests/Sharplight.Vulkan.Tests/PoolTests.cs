using Sharplight.Graph;

namespace Sharplight.Vulkan.Tests;

public sealed class PoolTests
{
    private static void Run(string label, int n, int c, int h, int w, string mode, int[] k, int[] s, int[] pads, int[] dil, bool ceil, bool countPad, bool slice = false)
    {
        Gpu.Require();
        var tg = new TG();
        int cin = slice ? c + 3 : c;
        var x = tg.InF("x", TG.T, cin, h, w);
        var xv = slice ? tg.B.Slice(x, 1, 1, c) : x;
        var attrs = Attributes.From(("mode", mode), ("kernel", k.Select(v => (long)v).ToArray()), ("strides", s.Select(v => (long)v).ToArray()),
            ("pads", pads.Select(v => (long)v).ToArray()), ("dilations", dil.Select(v => (long)v).ToArray()), ("ceil_mode", ceil), ("count_include_pad", countPad));
        tg.B.Output("y", tg.B.Emit1("Pool", [xv], attrs));
        Diff.Check("Pool." + mode, tg.B.Build(), new Step(n, 0, (x, Rnd.F((long)n * cin * h * w, 3))), what: label);
    }

    [Theory]
    [InlineData("max", 2, 8, 15, 15, 3, 2, 1, false, false)]
    [InlineData("max", 1, 8, 16, 16, 2, 2, 0, false, false)]
    [InlineData("max", 1, 8, 15, 15, 3, 2, 0, true, false)]
    [InlineData("avg", 2, 8, 15, 15, 3, 2, 1, false, false)]
    [InlineData("avg", 2, 8, 15, 15, 3, 2, 1, false, true)]
    [InlineData("avg", 1, 8, 14, 14, 2, 2, 0, true, true)]
    [InlineData("avg", 1, 8, 15, 15, 3, 2, 1, true, false)]
    [InlineData("avg", 1, 8, 15, 15, 3, 2, 1, true, true)]
    [InlineData("max", 1, 4, 7, 7, 7, 1, 0, false, false)]
    public void Square(string mode, int n, int c, int h, int w, int k, int s, int p, bool ceil, bool cp)
        => Run($"{mode} k{k} s{s} p{p} ceil{ceil} cp{cp}", n, c, h, w, mode, [k, k], [s, s], [p, p, p, p], [1, 1], ceil, cp);

    [Fact] public void MaxDilated() => Run("max dil2", 1, 6, 17, 17, "max", [3, 3], [1, 1], [2, 2, 2, 2], [2, 2], false, false);
    [Fact] public void MaxAsymmetric() => Run("max asym", 2, 5, 13, 17, "max", [3, 2], [2, 1], [1, 0, 2, 1], [1, 1], false, false);
    [Fact] public void AvgAsymmetricCountPad() => Run("avg asym cp", 2, 5, 13, 17, "avg", [3, 2], [2, 1], [1, 0, 2, 1], [1, 1], false, true);
    [Fact] public void AvgAsymmetricCeil() => Run("avg asym ceil", 2, 5, 13, 17, "avg", [3, 2], [2, 2], [1, 0, 1, 1], [1, 1], true, true);
    [Fact] public void StridedInput() => Run("sliced", 2, 6, 11, 11, "max", [3, 3], [2, 2], [1, 1, 1, 1], [1, 1], false, false, slice: true);
}
