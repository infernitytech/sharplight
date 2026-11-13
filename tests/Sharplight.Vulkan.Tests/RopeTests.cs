using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class RopeTests
{
    public static IEnumerable<object[]> Cases() =>
    [
        ["neox", 16, 1, null!, false, 1f],
        ["gptj", 16, 1, null!, false, 1f],
        ["neox", 8, 1, null!, false, 0.5f],
        ["gptj", 8, 1, null!, false, 1f],
        ["neox", 16, 4, new[] { 2, 3, 3, 0 }, false, 1f],
        ["neox", 16, 4, new[] { 3, 2, 2, 1 }, true, 1f],
        ["neox", 16, 4, new[] { 4, 2, 2, 0 }, true, 1f],
        ["neox", 8, 4, new[] { 1, 1, 1, 1 }, true, 1f],
    ];

    [Theory]
    [MemberData(nameof(Cases))]
    public void Variants(string style, int dims, int axes, int[]? sections, bool interleaved, float freqScale)
    {
        Gpu.Require();
        int T = 6;
        int H = 3;
        int D = 16;
        float @base = 10000f;
        var tg = new TG();
        var x = tg.InF("x", TG.T, H, D);
        var pos = tg.In("pos", DataType.I32, TG.T, axes);
        var attrs = Attributes.From(("dims", dims), ("base", @base), ("style", style), ("interleaved", interleaved), ("freq_scale", freqScale))
            .With("sections", sections ?? [0, 0, 0, 0]);
        tg.B.Output("y", tg.B.Emit1("Rope", [x, pos], attrs));
        Diff.Check("Rope", tg.B.Build(), new Step(T, 0, (x, Rnd.F(T * H * D, 1)), (pos, Rnd.I(T * axes, 5, 300))), atol: 1e-4, rtol: 1e-4, what: $"{style} dims={dims} axes={axes} il={interleaved}");
    }

    [Fact]
    public void FrequencyFactors()
    {
        Gpu.Require();
        int T = 5;
        int H = 2;
        int D = 16;
        float @base = 500000f;
        var tg = new TG();
        var x = tg.InF("x", TG.T, H, D);
        var pos = tg.In("pos", DataType.I32, TG.T, 1);
        var ff = tg.InF("ff", D / 2);
        tg.B.Output("y", tg.B.Emit1("Rope", [x, pos, ff], Attributes.From(("dims", D), ("base", @base), ("style", "neox"))));
        var factors = Sharplight.Models.RopeScaling.Llama3(D, @base, 32f, 1f, 4f, 8192).Factors;
        Diff.Check("Rope.ff", tg.B.Build(), new Step(T, 0, (x, Rnd.F(T * H * D, 1)), (pos, Enumerable.Range(100, T).ToArray()), (ff, factors)), atol: 1e-4, rtol: 1e-4);
    }

    [Fact]
    public void OnStridedSlice()
    {
        Gpu.Require();
        int T = 4;
        int H = 2;
        int D = 8;
        var tg = new TG();
        var x = tg.InF("x", TG.T, H * 2 * D);
        var pos = tg.In("pos", DataType.I32, TG.T, 1);
        var q = tg.B.Slice(tg.B.Reshape(x, 0, H, 2 * D), 2, 0, D);
        tg.B.Output("y", tg.B.Emit1("Rope", [q, pos], Attributes.From(("dims", D), ("base", 1000f), ("style", "neox"))));
        Diff.Check("Rope.strided", tg.B.Build(), new Step(T, 0, (x, Rnd.F(T * H * 2 * D, 3)), (pos, Enumerable.Range(10, T).ToArray())), atol: 1e-5, rtol: 1e-4);
    }
}
