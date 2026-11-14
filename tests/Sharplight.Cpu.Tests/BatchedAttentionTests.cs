using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// The batched form of Attention (equal-length sequences, <see cref="BatchedAttention"/>) and its additive masks: one bias per key
/// row or a per-query mask, on the tiled and the general kernel, against the naive reference per sequence.
/// </summary>
public class BatchedAttentionTests
{
    /// <summary>mask: 0 none, 1 one bias per key row, 2 a [rows, keys] mask per query (batch 1).</summary>
    [Theory]
    [InlineData(1, 1, 1, 4, 0)]
    [InlineData(3, 7, 7, 4, 0)]
    [InlineData(2, 37, 37, 3, 0)]
    [InlineData(4, 20, 9, 2, 0)]
    [InlineData(3, 7, 7, 4, 1)]
    [InlineData(2, 37, 37, 3, 1)]
    [InlineData(5, 1, 6, 2, 1)]
    [InlineData(1, 40, 33, 2, 2)]
    [InlineData(1, 2, 5, 2, 2)]
    public void MatchesPerSequenceReference(int batch, int rows, int keys, int heads, int mask)
    {
        const int D = 16;
        const int Dv = 8;
        float scale = 0.3f;
        int t = batch * rows;
        int s = batch * keys;
        var q = Kit.Rand(t * heads * D, 1);
        var k = Kit.Rand(s * heads * D, 2);
        var v = Kit.Rand(s * heads * Dv, 3);
        float[]? maskData = mask switch
        {
            1 => [.. Enumerable.Range(0, s).Select(j => j % 4 == 3 ? -10000f : 0.2f * (j % 3))],
            2 => [.. Enumerable.Range(0, rows * keys).Select(i => 0.1f * (i % 5) - (i % 7 == 0 ? 50f : 0f))],
            _ => null,
        };
        var tg = new TestGraph();
        var qi = tg.InF("q", TestGraph.T, heads, D);
        var ki = tg.InF("k", Dim.Of(s), heads, D);
        var vi = tg.InF("v", Dim.Of(s), heads, Dv);
        Value[] operands = [qi, ki, vi];
        Value? mi = null;
        if (maskData is not null)
        {
            mi = mask == 1 ? tg.InF("mask", Dim.Of(s)) : tg.InF("mask", Dim.Of(rows), Dim.Of(keys));
            operands = [.. operands, mi];
        }

        var attributes = Attributes.From(("scale", scale), (BatchedAttention.RowsAttribute, (long)rows), (BatchedAttention.KeysAttribute, (long)keys));
        tg.B.Output("o", tg.B.Emit1("Attention", operands, attributes));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: Math.Max(64, t));
        Kit.Set(m, qi, q);
        Kit.Set(m, ki, k);
        Kit.Set(m, vi, v);
        if (mi is not null)
        {
            Kit.Set(m, mi, maskData!);
        }

        var p = Kit.P(t);
        m.Run(p);
        var got = Kit.Out(m, "o", p);
        var expected = new List<float>();
        for (int b = 0; b < batch; b++)
        {
            float[]? sequenceMask = mask switch
            {
                1 => [.. Enumerable.Range(0, rows * keys).Select(i => maskData![b * keys + i % keys])],
                2 => maskData,
                _ => null,
            };
            expected.AddRange(Ref.Attention(Slice(q, b * rows, rows, heads * D), rows, heads, D, Slice(k, b * keys, keys, heads * D),
                Slice(v, b * keys, keys, heads * Dv), keys, heads, Dv, scale, 0, causal: false, mask: sequenceMask));
        }

        Kit.AssertClose([.. expected], got, 2e-5, 2e-4);
    }

    private static float[] Slice(float[] a, int row0, int count, int rowLength) => a.AsSpan(row0 * rowLength, count * rowLength).ToArray();
}
