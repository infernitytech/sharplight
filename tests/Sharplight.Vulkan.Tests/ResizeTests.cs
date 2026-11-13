using Sharplight.Graph;

namespace Sharplight.Vulkan.Tests;

public sealed class ResizeTests
{
    private static void Run(string label, string mode, string coord, string nearest, int n, int c, int h, int w, int oh, int ow, bool slice = false)
    {
        Gpu.Require();
        var tg = new TG();
        int cin = slice ? c + 2 : c;
        var x = tg.InF("x", TG.T, cin, h, w);
        var xv = slice ? tg.B.Slice(x, 1, 1, c) : x;
        tg.B.Output("y", tg.B.Emit1("Resize", [xv], Attributes.From(("size", new long[] { oh, ow }), ("mode", mode), ("coord", coord), ("nearest_mode", nearest))));
        Diff.Check("Resize." + mode, tg.B.Build(), new Step(n, 0, (x, Rnd.F((long)n * cin * h * w, 4))), atol: 2e-5, rtol: 2e-5, what: label);
    }

    public static IEnumerable<object[]> NearestCases()
    {
        foreach (var coord in new[] { "half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel" })
        {
            foreach (var nm in new[] { "round_prefer_floor", "round_prefer_ceil", "floor", "ceil" })
            {
                foreach (var (oh, ow) in new[] { (16, 14), (5, 6), (7, 9) })
                {
                    yield return [coord, nm, oh, ow];
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(NearestCases))]
    public void Nearest(string coord, string nm, int oh, int ow) => Run($"{coord}/{nm} {oh}x{ow}", "nearest", coord, nm, 2, 3, 8, 7, oh, ow);

    public static IEnumerable<object[]> LinearCases()
    {
        foreach (var coord in new[] { "half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel" })
        {
            foreach (var (oh, ow) in new[] { (16, 14), (5, 6), (7, 9), (1, 1), (8, 7) })
            {
                yield return [coord, oh, ow];
            }
        }
    }

    [Theory]
    [MemberData(nameof(LinearCases))]
    public void Linear(string coord, int oh, int ow) => Run($"{coord} {oh}x{ow}", "linear", coord, "round_prefer_floor", 2, 3, 8, 7, oh, ow);

    [Fact] public void Upsample2x() => Run("2x", "nearest", "asymmetric", "floor", 1, 16, 14, 14, 28, 28);
    [Fact] public void BilinearUpsample2xStrided() => Run("2x bilinear sliced", "linear", "half_pixel", "round_prefer_floor", 2, 4, 10, 10, 20, 20, slice: true);
}
