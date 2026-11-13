using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class RopeTests
{
    public static IEnumerable<object[]> Cases() =>
    [
        ["neox", 16, 1, null!, false, 1f],
        ["gptj", 16, 1, null!, false, 1f],
        ["neox", 8, 1, null!, false, 0.5f],                // partial rotation, freq_scale
        ["gptj", 8, 1, null!, false, 1f],
        ["neox", 16, 4, new[] { 2, 3, 3, 0 }, false, 1f],  // M-RoPE contiguous sections
        ["neox", 16, 4, new[] { 3, 2, 2, 1 }, true, 1f],   // IMROPE
        ["neox", 16, 4, new[] { 4, 2, 2, 0 }, true, 1f],   // IMROPE, axis 3 absent
        ["neox", 8, 4, new[] { 1, 1, 1, 1 }, true, 1f],
    ];

    [Theory, MemberData(nameof(Cases))]
    public void RopeMatchesReference(string style, int dims, int axes, int[]? sections, bool interleaved, float freqScale)
    {
        int T = 6;
        int H = 3;
        int D = 16;
        float @base = 10000f;
        var x = Kit.Rand(T * H * D, 1);
        var rng = new Random(5);
        var pos = new int[T * axes];
        for (int t = 0; t < T; t++)
        {
            for (int a = 0; a < axes; a++)
            {
                pos[t * axes + a] = rng.Next(0, 300);
            }
        }

        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H, D);
        var pin = tg.In("pos", DataType.I32, TestGraph.T, axes);
        var attrs = Attributes.From(("dims", dims), ("base", @base), ("style", style), ("interleaved", interleaved), ("freq_scale", freqScale))
            .With("sections", sections ?? [0, 0, 0, 0]);
        tg.B.Output("y", tg.B.Emit1("Rope", [xin, pin], attrs));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, pin, pos);
        var p = Kit.P(T);
        m.Run(p);
        Kit.AssertClose(Ref.Rope(x, T, H, D, pos, axes, dims, @base, style, sections, interleaved, freqScale), Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>Per-frequency divisors as third input (llama3 / yarn scaling).</summary>
    [Fact]
    public void RopeWithFrequencyFactors()
    {
        int T = 5;
        int H = 2;
        int D = 16;
        float @base = 500000f;
        var x = Kit.Rand(T * H * D, 1);
        var pos = Enumerable.Range(100, T).ToArray();
        var ff = Sharplight.Models.RopeScaling.Llama3(D, @base, 32f, 1f, 4f, 8192).Factors;
        Assert.Contains(ff, f => f > 1.5f);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H, D);
        var pin = tg.In("pos", DataType.I32, TestGraph.T, 1);
        var fin = tg.InF("ff", D / 2);
        tg.B.Output("y", tg.B.Emit1("Rope", [xin, pin, fin], Attributes.From(("dims", D), ("base", @base), ("style", "neox"))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, pin, pos);
        Kit.Set(m, fin, ff);
        var p = Kit.P(T);
        m.Run(p);
        Kit.AssertClose(Ref.Rope(x, T, H, D, pos, 1, D, @base, "neox", null, false, 1f, ff), Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>The YaRN magnitude (<c>attn_factor</c>) multiplies cos and sin, so it scales the rotated dimensions and leaves the others alone.</summary>
    [Theory]
    [InlineData("neox")]
    [InlineData("gptj")]
    public void AttentionFactorScalesTheRotatedDimensions(string style)
    {
        int T = 5;
        int H = 2;
        int D = 16;
        int dims = 8;
        float factor = 1.25f;
        float @base = 10000f;
        var x = Kit.Rand(T * H * D, 1);
        var pos = Enumerable.Range(7, T).ToArray();
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H, D);
        var pin = tg.In("pos", DataType.I32, TestGraph.T, 1);
        tg.B.Output("y", tg.B.Emit1("Rope", [xin, pin], Attributes.From(("dims", dims), ("base", @base), ("style", style), ("attn_factor", factor))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, x);
        Kit.Set(m, pin, pos);
        var p = Kit.P(T);
        m.Run(p);
        var expected = Ref.Rope(x, T, H, D, pos, 1, dims, @base, style, null, false, 1f);
        for (int row = 0; row < T * H; row++)
        {
            for (int i = 0; i < dims; i++)
            {
                expected[row * D + i] *= factor;
            }
        }

        Kit.AssertClose(expected, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>Rope applied to a strided Slice view (as in Qwen3.5 q after the joint q/gate projection).</summary>
    [Fact]
    public void RopeOnStridedInput()
    {
        int T = 4;
        int H = 2;
        int D = 8;
        var full = Kit.Rand(T * H * 2 * D, 3);
        var pos = Enumerable.Range(10, T).ToArray();
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, H * 2 * D);
        var pin = tg.In("pos", DataType.I32, TestGraph.T, 1);
        var q = tg.B.Slice(tg.B.Reshape(xin, 0, H, 2 * D), 2, 0, D);
        tg.B.Output("y", tg.B.Emit1("Rope", [q, pin], Attributes.From(("dims", D), ("base", 1000f), ("style", "neox"))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xin, full);
        Kit.Set(m, pin, pos);
        var p = Kit.P(T);
        m.Run(p);
        var qs = new float[T * H * D];
        for (int r = 0; r < T * H; r++)
        {
            for (int i = 0; i < D; i++)
            {
                qs[r * D + i] = full[r * 2 * D + i];
            }
        }

        Kit.AssertClose(Ref.Rope(qs, T, H, D, pos, 1, D, 1000f, "neox", null, false, 1f), Kit.Out(m, "y", p), 1e-5, 1e-4);
    }
}
