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
    /// Group selection: per-head block selection by the best position score with forced local blocks (MiniMax), and pooled groups whose keys are
    /// softmax-gated averages, with the incomplete group's positions appended (GLM-5.3 k-pool); exact positions against the CPU oracle.
    /// </summary>
    [Theory]
    [InlineData("f32", 2, 16, 4, 12, "max", 1, false, true)]
    [InlineData("f16", 4, 32, 8, 24, "max", 2, false, true)]
    [InlineData("f32", 4, 16, 4, 8, "pooled", 0, true, false)]
    [InlineData("f16", 8, 32, 8, 16, "pooled", 0, true, false)]
    [InlineData("f32", 4, 16, 4, 8, "max", 0, false, false)]
    public void IndexerTopKGroupsAcrossSteps(string cacheType, int heads, int dims, int group, int k, string score, int local, bool tail, bool perHead)
    {
        Gpu.Require();
        int[] chunks = [5, 3, 1, 1, 40, 1, 7, 1];
        int total = chunks.Sum();
        bool pooled = score == "pooled";
        int rowValues = pooled ? 2 * dims : dims;
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, heads, dims);
        var ki = tg.InF("k", T, 1, rowValues);
        var wi = perHead ? null : tg.InF("w", T, heads);
        var keys = tg.State("keys", DataType.Get(cacheType), StateKind.KeyCache, Dim.Sym("ctx"), 1, rowValues);
        tg.B.CacheWrite(keys, ki);
        tg.B.RuntimeScalar("n_past");
        List<Value> inputs = [qi, keys];
        if (wi is not null)
        {
            inputs.Add(wi);
        }

        if (pooled)
        {
            inputs.Add(tg.WF("gate", Rnd.F(group * dims, 11), group, dims));
        }

        var attributes = Attributes.From(("k", (long)k), ("offset", "n_past"), ("group", (long)group), ("group_score", score), ("local", (long)local),
            ("tail", tail), ("per_head", perHead));
        tg.B.Output("selection", tg.B.Emit1("IndexerTopK", inputs, attributes));
        var q = Rnd.F(total * heads * dims, 1);
        var key = Rnd.F(total * rowValues, 2);
        var w = Rnd.F(total * heads, 3);
        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            var feeds = new List<(Value, Array)> { (qi, q[(past * heads * dims)..((past + c) * heads * dims)]), (ki, key[(past * rowValues)..((past + c) * rowValues)]) };
            if (wi is not null)
            {
                feeds.Add((wi, w[(past * heads)..((past + c) * heads)]));
            }

            steps.Add(new Step(c, past, [.. feeds]));
            past += c;
        }

        Diff.Check("IndexerTopK.groups", tg.B.Build(), steps, maxTokens: chunks.Max(), ctx: 128,
            what: $"{cacheType} H={heads} D={dims} G={group} k={k} {score} local={local} tail={tail} per_head={perHead}");
    }

    /// <summary>
    /// Attention over given key positions (with skipped entries: below 0 or beyond the row's position), on a latent-attention-like shared K/V cache
    /// with many heads and on a GQA cache with sinks.
    /// </summary>
    [Theory]
    [InlineData("f16", 40, 32, 8, 1, false, false)]
    [InlineData("f32", 64, 64, 4, 2, true, false)]
    [InlineData("f32", 64, 64, 8, 2, false, true)]
    [InlineData("f16", 32, 32, 4, 4, true, true)]
    public void CachedAttentionWithSelection(string cacheType, int D, int Dv, int Hq, int Hkv, bool sinks, bool perHead)
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
        int selectionHeads = perHead ? Hkv : 1;
        var si = perHead ? tg.In("sel", DataType.I32, T, Hkv, width) : tg.In("sel", DataType.I32, T, width);
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
        int rowWidth = selectionHeads * width;
        var selection = new int[total * rowWidth];
        for (int t = 0; t < total; t++)
        {
            for (int i = 0; i < rowWidth; i++)
            {
                // Mostly visible positions, some skipped entries (-1, or beyond the row's position).
                int r = random.Next(10);
                selection[t * rowWidth + i] = r == 0 ? -1 : r == 1 ? t + 1 + random.Next(5) : random.Next(t + 1);
            }
        }

        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]),
                (vi, v[(past * Hkv * Dv)..((past + c) * Hkv * Dv)]), (si, selection[(past * rowWidth)..((past + c) * rowWidth)])));
            past += c;
        }

        Diff.Check("CachedAttention.selection", tg.B.Build(), steps, atol: 3e-4, rtol: 3e-3, maxTokens: chunks.Max(), ctx: 64, what: $"{cacheType} D={D} Hq={Hq} Hkv={Hkv} per_head={perHead}");
    }
}
