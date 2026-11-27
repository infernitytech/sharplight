using Sharplight.Graph;
using Sharplight.Runtime;
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

    /// <summary>
    /// Queries, keys and values far outside f16's range (about 6.5e4) or far below it keep near-f32 accuracy in the split-f16 flash
    /// kernels (fe, and fa for the windowed form): each query row, key row and V tile is brought into f16's range by a power of two
    /// before the split (XTTS's Perceiver resampler attends over latents of 1e10).
    /// </summary>
    [Theory]
    [InlineData(1e6f, 1e-6f, 1e6f, false)]
    [InlineData(1e9f, 1e-8f, 1e10f, false)]
    [InlineData(3e-6f, 2e5f, 1e-7f, false)]
    [InlineData(1e6f, 1e-6f, 1e6f, true)]
    [InlineData(2f, 2f, 1f, true)]
    public void SplitHalfLargeMagnitudes(float qMagnitude, float kMagnitude, float vMagnitude, bool windowed)
    {
        int batch = windowed ? 1 : 2;
        var (g, step) = Graph(batch, 97, 97, 4, 64, mask: 0, seed: 5, magnitude: qMagnitude, kMagnitude: kMagnitude, vMagnitude: vMagnitude,
            window: windowed ? 40 : 0);
        Diff.Check("Attention.full.magnitudes", g, step, atol: 5e-6 * vMagnitude, rtol: 5e-6, maxTokens: batch * 97,
            what: $"q={qMagnitude} k={kMagnitude} v={vMagnitude} windowed={windowed}");
    }

    /// <summary>Fast matrix precision runs f32 attention on the single-product f16 flash kernel, with f16-level error.</summary>
    [Theory]
    [InlineData(1, 197, 12, 64)]
    [InlineData(1, 1500, 6, 64)]
    [InlineData(3, 50, 4, 64)]
    public void FastPrecision(int batch, int rows, int heads, int d)
    {
        var (g, step) = Graph(batch, rows, rows, heads, d, mask: 0, seed: 1, magnitude: 2f);
        Diff.Check("Attention.full.fast", g, step, atol: 2e-2, rtol: 2e-2, maxTokens: batch * rows, precision: MatrixPrecision.Fast,
            what: $"b={batch} rows={rows} H={heads} D={d}");
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

    /// <summary>
    /// A key-padding bias together with T5's relative position bias (the Chronos encoder's mask of missing values, #93), on the scalar and flash
    /// kernels and without matrix units.
    /// </summary>
    [Theory]
    [InlineData(3, false)]
    [InlineData(40, false)]
    [InlineData(300, false)]
    [InlineData(40, true)]
    public void MaskWithPositionBias(int rows, bool withoutMatrixUnits)
    {
        var (g, step) = Graph(1, rows, rows, 4, 64, mask: 1, seed: 13, magnitude: 1f, positionBiasRange: 16);
        Diff.Check("Attention.mask-position-bias", g, step, atol: 5e-6, rtol: 5e-6, maxTokens: rows,
            disabledFeatures: withoutMatrixUnits ? new HashSet<string> { DeviceFeatures.MatrixF16, DeviceFeatures.MatrixWorkgroup } : null,
            what: $"rows={rows} withoutMatrixUnits={withoutMatrixUnits}");
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
    /// Without matrix units the f32 encoder flash kernel (fes) serves these forms: plain, batched, cross, both mask kinds and
    /// key ranges (1500 rows).
    /// </summary>
    [Theory]
    [InlineData(1, 197, 197, 12, 64, 0)]
    [InlineData(3, 50, 50, 4, 64, 1)]
    [InlineData(1, 100, 100, 4, 64, 2)]
    [InlineData(2, 77, 77, 4, 32, 0)]
    [InlineData(3, 70, 33, 4, 64, 0)]
    [InlineData(1, 1500, 1500, 6, 64, 0)]
    [InlineData(2, 77, 77, 4, 80, 0)]
    public void WithoutMatrixUnits(int batch, int rows, int keys, int heads, int d, int mask)
    {
        var (g, step) = Graph(batch, rows, keys, heads, d, mask, seed: 11, magnitude: 2f);
        Diff.Check("Attention.simt", g, step, atol: 5e-6, rtol: 5e-6, maxTokens: batch * Math.Max(rows, keys),
            disabledFeatures: new HashSet<string> { DeviceFeatures.MatrixF16, DeviceFeatures.MatrixWorkgroup },
            what: $"b={batch} {rows}x{keys} H={heads} D={d} mask={mask}");
    }

    /// <summary>
    /// An Attention graph over <paramref name="batch"/> sequences of <paramref name="rows"/> queries and <paramref name="keys"/> keys;
    /// mask 1 = one bias per key row (every third key masked), 2 = a per-query [rows, keys] mask (batch 1); <paramref name="positionBiasRange"/> &gt; 0 adds a
    /// relative position bias over the distances ±range (self-attention, batch 1).
    /// </summary>
    private static (ComputeGraph Graph, Step Step) Graph(int batch, int rows, int keys, int heads, int d, int mask, int seed, float magnitude,
        float? kMagnitude = null, float vMagnitude = 1f, long window = 0, int positionBiasRange = 0)
    {
        var tg = new TG();
        var q = tg.InF("q", TG.T, heads, d);
        // A window or a position bias needs self-attention: keys share the queries' row symbol.
        var keyRows = window > 0 || positionBiasRange > 0 ? TG.T : Dim.Of((long)batch * keys);
        var k = tg.InF("k", keyRows, heads, d);
        var v = tg.InF("v", keyRows, heads, d);
        var attributes = new List<(string, object)> { ("scale", 1f / MathF.Sqrt(d)) };
        if (window > 0)
        {
            attributes.Add(("window", window));
        }

        if (positionBiasRange > 0)
        {
            attributes.Add(("position_bias", Rnd.F(heads * (2 * positionBiasRange + 1), seed + 3, 1f)));
            attributes.Add(("position_bias_range", positionBiasRange));
        }

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
        var step = new Step(t, 0, (q, Rnd.F(t * heads * d, seed, magnitude)), (k, Rnd.F(s * heads * d, seed + 1, kMagnitude ?? magnitude)),
            (v, Rnd.F(s * heads * d, seed + 2, vMagnitude)));
        return (tg.B.Build(), step);
    }
}
