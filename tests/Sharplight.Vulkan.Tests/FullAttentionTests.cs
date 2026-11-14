using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Bidirectional Attention over f32 K/V (encoders): the batched form, additive masks, and the strict-precision accuracy of the
/// split-f16 flash kernel, against the CPU oracle.
/// </summary>
public sealed class FullAttentionTests
{
    /// <summary>
    /// Under strict precision f32 attention keeps near-f32 accuracy: the flash kernel splits Q, K, V and P into two f16 parts.
    /// Reduced precision runs single f16 products and is allowed f16-level error.
    /// </summary>
    [Theory]
    [InlineData(1, 197, 12, 64, false)]
    [InlineData(1, 1500, 6, 64, false)]
    [InlineData(3, 50, 4, 64, false)]
    [InlineData(2, 77, 4, 32, false)]
    [InlineData(2, 77, 4, 80, false)]
    [InlineData(1, 197, 12, 64, true)]
    [InlineData(3, 50, 4, 64, true)]
    public void BatchedStrictAccuracy(int batch, int rows, int heads, int d, bool reduced)
    {
        var (g, step) = Graph(batch, rows, rows, heads, d, mask: 0, seed: 1, magnitude: 2f);
        double tol = reduced ? 2e-2 : 5e-6;
        Diff.Check("Attention.full", g, step, atol: tol, rtol: tol, maxTokens: batch * rows, allowReducedPrecision: reduced,
            what: $"b={batch} rows={rows} H={heads} D={d} reduced={reduced}");
    }

    /// <summary>Key-padding biases (one per key row) and per-query masks, on the scalar and flash kernels.</summary>
    [Theory]
    [InlineData(1, 3, 1)]
    [InlineData(4, 3, 1)]
    [InlineData(1, 40, 1)]
    [InlineData(4, 40, 1)]
    [InlineData(1, 3, 2)]
    [InlineData(1, 100, 2)]
    public void Masks(int batch, int rows, int mask)
    {
        var (g, step) = Graph(batch, rows, rows, 4, 64, mask, seed: 7, magnitude: 1f);
        Diff.Check("Attention.mask", g, step, atol: 5e-6, rtol: 5e-6, maxTokens: batch * rows, what: $"b={batch} rows={rows} mask={mask}");
    }

    /// <summary>Queries and keys of different lengths per sequence (cross-attention within each batch entry).</summary>
    [Theory]
    [InlineData(2, 5, 9)]
    [InlineData(3, 70, 33)]
    public void BatchedCross(int batch, int rows, int keys)
    {
        var (g, step) = Graph(batch, rows, keys, 4, 64, mask: 0, seed: 3, magnitude: 1f);
        Diff.Check("Attention.batched-cross", g, step, atol: 5e-6, rtol: 5e-6, maxTokens: batch * Math.Max(rows, keys), what: $"b={batch} {rows}x{keys}");
    }

    /// <summary>
    /// An Attention graph over <paramref name="batch"/> sequences of <paramref name="rows"/> queries and <paramref name="keys"/> keys;
    /// mask 1 = one bias per key row (every third key masked), 2 = a per-query [rows, keys] mask (batch 1).
    /// </summary>
    private static (ComputeGraph Graph, Step Step) Graph(int batch, int rows, int keys, int heads, int d, int mask, int seed, float magnitude)
    {
        var tg = new TG();
        var q = tg.InF("q", TG.T, heads, d);
        var k = tg.InF("k", Dim.Of((long)batch * keys), heads, d);
        var v = tg.InF("v", Dim.Of((long)batch * keys), heads, d);
        var attributes = new List<(string, object)> { ("scale", 1f / MathF.Sqrt(d)) };
        if (batch > 1 || keys != rows)
        {
            attributes.Add((BatchedAttention.RowsAttribute, (long)rows));
            attributes.Add((BatchedAttention.KeysAttribute, (long)keys));
        }

        Value[] operands = [q, k, v];
        float[]? maskData = null;
        if (mask == 1)
        {
            maskData = [.. Enumerable.Range(0, batch * keys).Select(j => j % 3 == 2 ? -10000f : 0.25f * (j % 5))];
            operands = [.. operands, tg.Const("mask", maskData, batch * keys)];
        }
        else if (mask == 2)
        {
            maskData = [.. Enumerable.Range(0, rows * keys).Select(i => (i % keys) > (i / keys) + 1 ? float.NegativeInfinity : 0.1f * (i % 7))];
            operands = [.. operands, tg.Const("mask", maskData, rows, keys)];
        }

        tg.B.Output("o", tg.B.Emit1("Attention", operands, Attributes.From([.. attributes])));
        int t = batch * rows;
        int s = batch * keys;
        var step = new Step(t, 0, (q, Rnd.F(t * heads * d, seed, magnitude)), (k, Rnd.F(s * heads * d, seed + 1, magnitude)),
            (v, Rnd.F(s * heads * d, seed + 2)));
        return (tg.B.Build(), step);
    }
}
