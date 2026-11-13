using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Packed-sequence ops (embedding graphs): Attention with i32 row spans and SequencePool, against naive references.</summary>
public class PackedSequenceTests
{
    private static (int[] Spans, int Total) RowSpans(int[] lengths)
    {
        int total = lengths.Sum(), first = 0;
        var spans = new int[total * 2];
        foreach (int len in lengths)
        {
            for (int t = first; t < first + len; t++) { spans[t * 2] = first; spans[t * 2 + 1] = len; }
            first += len;
        }
        return (spans, total);
    }

    private static float[] Slice(float[] a, int row0, int rows, int rowLen) => a.AsSpan(row0 * rowLen, rows * rowLen).ToArray();

    [Theory]
    [InlineData(false, 4, 2), InlineData(true, 4, 2), InlineData(false, 3, 3), InlineData(true, 2, 1)]
    public void AttentionWithinSpans(bool causal, int Hq, int Hkv)
    {
        int D = 16, Dv = 8; float scale = 0.25f;
        int[] lengths = [5, 1, 9, 3];
        var (spans, T) = RowSpans(lengths);
        var q = Kit.Rand(T * Hq * D, 1); var k = Kit.Rand(T * Hkv * D, 2); var v = Kit.Rand(T * Hkv * Dv, 3);
        var tg = new TG(); var Tm = TG.T;
        var qi = tg.InF("q", Tm, Hq, D); var ki = tg.InF("k", Tm, Hkv, D); var vi = tg.InF("v", Tm, Hkv, Dv);
        var si = tg.In("spans", DataType.I32, Tm, 2);
        tg.B.Output("o", tg.B.Emit1("Attention", [qi, ki, vi, si], Attributes.From(("scale", scale), ("causal", causal))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, qi, q); Kit.Set(m, ki, k); Kit.Set(m, vi, v); Kit.Set(m, si, spans);
        var p = Kit.P(T); m.Run(p);
        var got = Kit.Out(m, "o", p);
        var exp = new List<float>();
        int first = 0;
        foreach (int len in lengths)
        {
            exp.AddRange(Ref.Attention(Slice(q, first, len, Hq * D), len, Hq, D, Slice(k, first, len, Hkv * D), Slice(v, first, len, Hkv * Dv), len, Hkv, Dv, scale, 0, causal));
            first += len;
        }
        Kit.AssertClose([.. exp], got, 2e-5, 2e-4);
    }

    [Fact]
    public void AttentionEmptySpanGivesZeros()
    {
        int Hq = 2, D = 8;
        var tg = new TG(); var Tm = TG.T;
        var qi = tg.InF("q", Tm, Hq, D); var ki = tg.InF("k", Tm, Hq, D); var vi = tg.InF("v", Tm, Hq, D);
        var si = tg.In("spans", DataType.I32, Tm, 2);
        tg.B.Output("o", tg.B.Emit1("Attention", [qi, ki, vi, si], Attributes.From(("scale", 1f))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, qi, Kit.Rand(3 * Hq * D, 1)); Kit.Set(m, ki, Kit.Rand(3 * Hq * D, 2)); Kit.Set(m, vi, Kit.Rand(3 * Hq * D, 3));
        Kit.Set(m, si, new[] { 0, 3, 1, 0, 0, 3 });
        var p = Kit.P(3); m.Run(p);
        var o = Kit.Out(m, "o", p);
        Assert.All(o.AsSpan(Hq * D, Hq * D).ToArray(), x => Assert.Equal(0f, x));
        Assert.Contains(o.AsSpan(0, Hq * D).ToArray(), x => x != 0);
    }

    [Theory]
    [InlineData("mean"), InlineData("sum"), InlineData("max"), InlineData("first"), InlineData("last")]
    public void SequencePool(string mode)
    {
        int D = 7;
        int[] spans = [0, 3, 3, 1, 4, 0, 4, 5];   // last two rows of x are not pooled; one empty span
        int T = 11, B = spans.Length / 2;
        var x = Kit.Rand(T * D, 5);
        var tg = new TG(); var Tm = TG.T; var Bm = Dim.Sym("R");
        var xi = tg.InF("x", Tm, D); var si = tg.In("spans", DataType.I32, Bm, 2);
        tg.B.Output("y", tg.B.Emit1("SequencePool", [xi, si], Attributes.From(("mode", mode))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, xi, x); Kit.Set(m, si, spans);
        var p = Kit.P(T); p.Symbols["R"] = B; m.Run(p);
        var exp = new float[B * D];
        for (int b = 0; b < B; b++)
        {
            int first = spans[b * 2], cnt = spans[b * 2 + 1];
            for (int d = 0; d < D; d++)
            {
                if (cnt == 0)
                {
                    continue;
                }

                var col = Enumerable.Range(first, cnt).Select(r => x[r * D + d]).ToArray();
                exp[b * D + d] = mode switch { "mean" => col.Average(), "sum" => col.Sum(), "max" => col.Max(), "first" => col[0], _ => col[^1] };
            }
        }
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-5, 1e-5, mode);
    }
}
