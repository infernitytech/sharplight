using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>DeepSeek sparse attention: the lightning-indexer selection (<c>IndexerTopK</c>) and attention over selected positions, Vulkan vs the CPU oracle.</summary>
public sealed class SparseAttentionTests
{
    /// <summary>Prefill chunks and decode rows; rows that see at most k positions list all of them, the others the k best (exact positions).</summary>
    [Theory]
    [InlineData("f32", 4, 16, 8)]
    [InlineData("f16", 16, 64, 8)]
    [InlineData("f32", 64, 128, 32)]
    public void IndexerTopKAcrossSteps(string cacheType, int heads, int dims, int k)
    {
        Gpu.Require();
        int[] chunks = [5, 3, 1, 1, 40, 1, 7];
        int total = chunks.Sum();
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, heads, dims);
        var ki = tg.InF("k", T, 1, dims);
        var wi = tg.InF("w", T, heads);
        var keys = tg.State("keys", DataType.Get(cacheType), StateKind.KeyCache, Dim.Sym("ctx"), 1, dims);
        tg.B.CacheWrite(keys, ki);
        tg.B.RuntimeScalar("n_past");
        tg.B.Output("selection", tg.B.Emit1("IndexerTopK", [qi, keys, wi], Attributes.From(("k", (long)k), ("offset", "n_past"))));
        var q = Rnd.F(total * heads * dims, 1);
        var key = Rnd.F(total * dims, 2);
        var w = Rnd.F(total * heads, 3);
        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * heads * dims)..((past + c) * heads * dims)]), (ki, key[(past * dims)..((past + c) * dims)]),
                (wi, w[(past * heads)..((past + c) * heads)])));
            past += c;
        }

        Diff.Check("IndexerTopK", tg.B.Build(), steps, maxTokens: chunks.Max(), ctx: 128, what: $"{cacheType} H={heads} D={dims} k={k}");
    }

    /// <summary>
    /// Attention over given key positions (with skipped entries: below 0 or beyond the row's position), on a latent-attention-like shared K/V cache
    /// with many heads and on a GQA cache with sinks.
    /// </summary>
    [Theory]
    [InlineData("f16", 40, 32, 8, 1, false)]
    [InlineData("f32", 64, 64, 4, 2, true)]
    public void CachedAttentionWithSelection(string cacheType, int D, int Dv, int Hq, int Hkv, bool sinks)
    {
        Gpu.Require();
        int[] chunks = [6, 3, 1, 30, 1];
        int width = 8;
        int total = chunks.Sum();
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, Dv);
        var si = tg.In("sel", DataType.I32, T, width);
        var ct = DataType.Get(cacheType);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, Dv);
        tg.B.CacheWrite(kc, ki);
        tg.B.CacheWrite(vc, vi);
        tg.B.RuntimeScalar("n_past");
        var attributes = Attributes.From(("scale", 1f / MathF.Sqrt(D)), ("offset", "n_past"), ("selection", true));
        Value[] inputs = [qi, kc, vc, si];
        if (sinks)
        {
            inputs = [qi, kc, vc, tg.WF("sinks", Rnd.F(Hq, 9), Hq), si];
            attributes = attributes.With("sinks", true);
        }

        tg.B.Output("o", tg.B.Emit1("CachedAttention", inputs, attributes));
        var q = Rnd.F(total * Hq * D, 1).Select(x => x * 4).ToArray();
        var k = Rnd.F(total * Hkv * D, 2);
        var v = Rnd.F(total * Hkv * Dv, 3);
        var random = new Random(5);
        var selection = new int[total * width];
        for (int t = 0; t < total; t++)
        {
            for (int i = 0; i < width; i++)
            {
                // Mostly visible positions, some skipped entries (-1, or beyond the row's position).
                int r = random.Next(10);
                selection[t * width + i] = r == 0 ? -1 : r == 1 ? t + 1 + random.Next(5) : random.Next(t + 1);
            }
        }

        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]),
                (vi, v[(past * Hkv * Dv)..((past + c) * Hkv * Dv)]), (si, selection[(past * width)..((past + c) * width)])));
            past += c;
        }

        Diff.Check("CachedAttention.selection", tg.B.Build(), steps, atol: 3e-4, rtol: 3e-3, maxTokens: chunks.Max(), ctx: 64, what: $"{cacheType} D={D} Hq={Hq} Hkv={Hkv}");
    }
}
