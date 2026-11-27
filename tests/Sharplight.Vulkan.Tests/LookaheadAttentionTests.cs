using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// CachedAttention with <c>lookahead</c> (issue #98, Gemma 3's image rows): rows of a block see the rest of their block in the same pass, other
/// rows stay causal, and a window still counts back from each row's own position. The CPU oracle is checked against a direct computation, the
/// GPU against the CPU, on one sequence and on a paged multi-sequence cache, with passes that bind the lookahead input to one entry (the plain
/// kernels: decode rows, prefill chunks long enough for flash attention) between the passes with blocks.
/// </summary>
public sealed class LookaheadAttentionTests
{
    private const int Hq = 4;
    private const int Hkv = 2;
    private const int D = 32;

    /// <summary>One pass of one sequence: its row count and the blocks (first row, rows) whose rows see each other.</summary>
    private sealed record Pass(int Slot, int Rows, params (int First, int Count)[] Blocks);

    [Theory]
    [InlineData("f32", 0)]
    [InlineData("f32", 8)]
    [InlineData("f16", 0)]
    [InlineData("f16", 8)]
    public void SingleSequence(string cacheType, int window)
    {
        Gpu.Require();
        Pass[][] script =
        [
            [new Pass(0, 12, (3, 5))],
            [new Pass(0, 1)],
            [new Pass(0, 1)],
            [new Pass(0, 30, (0, 10), (15, 15))],   // a block at each end of the pass
            [new Pass(0, 40)],                      // no block: the plain (flash) kernels
            [new Pass(0, 1)],
            [new Pass(0, 9, (4, 5))],
        ];
        var (g, q, k, v, look) = Graph(cacheType, window);
        var steps = new List<Step>();
        var feeds = Feeds(script, slots: 1);
        foreach (var (pass, f) in script.Zip(feeds))
        {
            int rows = pass.Sum(p => p.Rows);
            steps.Add(new Step(rows, f.Past[0], (q, f.Q), (k, f.K), (v, f.V), (look, f.Look))
            {
                Symbols = new Dictionary<string, long> { [WellKnown.LookaheadSymbol] = f.Look.Length },
            });
        }

        CheckOracle(g, steps, script, feeds, window, cacheType, maxTokens: 48, ctx: 128);
        Diff.Check("CachedAttention.lookahead", g, steps, atol: 2e-3, rtol: 5e-3, maxTokens: 48, ctx: 128, what: $"{cacheType} window={window}");
    }

    [Theory]
    [InlineData("f32", 0)]
    [InlineData("f16", 8)]
    public void PagedMultiSequence(string cacheType, int window)
    {
        Gpu.Require();
        const int slots = 2;
        const int blockSize = 16;
        const int slotContext = 64;
        Pass[][] script =
        [
            [new Pass(0, 12, (2, 7)), new Pass(1, 5)],
            [new Pass(0, 1), new Pass(1, 1)],
            [new Pass(1, 20, (0, 20)), new Pass(0, 3, (0, 3))],
            [new Pass(0, 1), new Pass(1, 1)],
            [new Pass(0, 24), new Pass(1, 2)],
            [new Pass(1, 6, (1, 4)), new Pass(0, 1)],
        ];
        var (single, _, _, _, _) = Graph(cacheType, window);
        var g = MultiSequence.Transform(single, slots, blockSize, slotContext / blockSize).Graph;
        Value In(string name) => g.Inputs.Single(x => x.Name == name);
        var feeds = Feeds(script, slots);
        var length = new int[slots];
        // Slot s owns blocks s, s + slots, s + 2 * slots, ... of the pool.
        var table = new int[slots * (slotContext / blockSize)];
        for (int s = 0; s < slots; s++)
        {
            for (int j = 0; j < slotContext / blockSize; j++)
            {
                table[s * (slotContext / blockSize) + j] = s + j * slots;
            }
        }

        var steps = new List<Step>();
        foreach (var (pass, f) in script.Zip(feeds))
        {
            int rows = pass.Sum(p => p.Rows);
            var seqRows = new int[rows * 4];
            var runs = new int[pass.Length * 4];
            int r = 0;
            for (int u = 0; u < pass.Length; u++)
            {
                var (slot, n) = (pass[u].Slot, pass[u].Rows);
                for (int i = 0; i < n; i++, r++)
                {
                    seqRows[r * 4] = slot;
                    seqRows[r * 4 + 1] = length[slot] + i;
                    seqRows[r * 4 + 2] = n;
                    seqRows[r * 4 + 3] = u;
                }

                runs[u * 4] = slot;
                runs[u * 4 + 1] = r - n;
                runs[u * 4 + 2] = n;
                runs[u * 4 + 3] = length[slot];
                length[slot] += n;
            }

            steps.Add(new Step(rows, 0, (In("q"), f.Q), (In("k"), f.K), (In("v"), f.V), (In("look"), f.Look), (In(MultiSequence.RowsInput), seqRows),
                (In("sequence_runs"), runs), (In(MultiSequence.BlockTableInput), table))
            {
                Symbols = new Dictionary<string, long> { [WellKnown.RunSymbol] = pass.Length, [WellKnown.LookaheadSymbol] = f.Look.Length },
            });
        }

        int poolRows = slots * slotContext;
        CheckOracle(g, steps, script, feeds, window, cacheType, maxTokens: 32, ctx: poolRows);
        Diff.Check("CachedAttention.lookahead.paged", g, steps, atol: 2e-3, rtol: 5e-3, maxTokens: 32, ctx: poolRows, what: $"{cacheType} window={window}");
    }

    private static (ComputeGraph Graph, Value Q, Value K, Value V, Value Look) Graph(string cacheType, int window)
    {
        var tg = new TG();
        var T = TG.T;
        var q = tg.InF("q", T, Hq, D);
        var k = tg.InF("k", T, Hkv, D);
        var v = tg.InF("v", T, Hkv, D);
        var look = tg.B.Input("look", InputRole.AttentionLookahead, DataType.I32, new Shape(Dim.Sym(WellKnown.LookaheadSymbol)));
        var type = DataType.Get(cacheType);
        var kc = tg.State("kc", type, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", type, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, k);
        tg.B.CacheWrite(vc, v);
        tg.B.RuntimeScalar("n_past");
        var attrs = new List<(string, object)> { ("scale", 1f / MathF.Sqrt(D)), ("offset", "n_past"), ("lookahead", true) };
        if (window > 0)
        {
            attrs.Add(("window", window));
        }

        tg.B.Output("o", tg.B.Emit1("CachedAttention", [q, kc, vc, look], Attributes.From([.. attrs])));
        return (tg.B.Build(), q, k, v, look);
    }

    /// <summary>Per pass: random q / k / v rows, the lookahead entries (one ignored entry for a pass without blocks) and each slot's length before it.</summary>
    private static List<(float[] Q, float[] K, float[] V, int[] Look, int[] Past)> Feeds(Pass[][] script, int slots)
    {
        var length = new int[slots];
        var feeds = new List<(float[], float[], float[], int[], int[])>();
        int seed = 40;
        foreach (var pass in script)
        {
            int rows = pass.Sum(p => p.Rows);
            var look = new int[rows];
            int r = 0;
            foreach (var p in pass)
            {
                foreach (var (first, count) in p.Blocks)
                {
                    for (int i = 0; i < count; i++)
                    {
                        look[r + first + i] = count - 1 - i;
                    }
                }

                r += p.Rows;
            }

            var past = (int[])length.Clone();
            foreach (var p in pass)
            {
                length[p.Slot] += p.Rows;
            }

            feeds.Add((Rnd.F((long)rows * Hq * D, seed++, 4f), Rnd.F((long)rows * Hkv * D, seed++), Rnd.F((long)rows * Hkv * D, seed++),
                look.Any(x => x > 0) ? look : [0], past));
        }

        return feeds;
    }

    /// <summary>
    /// The CPU oracle against softmax attention computed directly: the row at position p of a slot sees that slot's keys
    /// [max(0, p − window + 1), p + lookahead] (the cache rounded to f16 when the cache is f16).
    /// </summary>
    private static void CheckOracle(ComputeGraph g, List<Step> steps, Pass[][] script, List<(float[] Q, float[] K, float[] V, int[] Look, int[] Past)> feeds,
        int window, string cacheType, long maxTokens, long ctx)
    {
        var cpu = Diff.RunCpu(g, steps, maxTokens, ctx);
        var keys = Enumerable.Range(0, 4).Select(_ => new List<float[]>()).ToArray();     // per slot, per position: [Hkv * D]
        var values = Enumerable.Range(0, 4).Select(_ => new List<float[]>()).ToArray();
        float Round(float x) => cacheType == "f16" ? (float)(Half)x : x;
        double maxDiff = 0;
        for (int s = 0; s < script.Length; s++)
        {
            var (q, k, v, look, _) = feeds[s];
            int rows = script[s].Sum(p => p.Rows);
            var slotOf = new int[rows];
            var posOf = new int[rows];
            int r = 0;
            foreach (var p in script[s])
            {
                for (int i = 0; i < p.Rows; i++, r++)
                {
                    slotOf[r] = p.Slot;
                    posOf[r] = keys[p.Slot].Count;
                    keys[p.Slot].Add([.. k.AsSpan(r * Hkv * D, Hkv * D).ToArray().Select(Round)]);
                    values[p.Slot].Add([.. v.AsSpan(r * Hkv * D, Hkv * D).ToArray().Select(Round)]);
                }
            }

            for (int t = 0; t < rows; t++)
            {
                int pos = posOf[t];
                int hi = pos + (look.Length == rows ? look[t] : 0);
                int lo = window > 0 ? Math.Max(0, pos - window + 1) : 0;
                for (int h = 0; h < Hq; h++)
                {
                    int hk = h / (Hq / Hkv);
                    var scores = new double[hi - lo + 1];
                    for (int j = lo; j <= hi; j++)
                    {
                        double dot = 0;
                        for (int d = 0; d < D; d++)
                        {
                            dot += q[(t * Hq + h) * D + d] * keys[slotOf[t]][j][hk * D + d];
                        }

                        scores[j - lo] = dot / Math.Sqrt(D);
                    }

                    double max = scores.Max();
                    double sum = scores.Sum(x => Math.Exp(x - max));
                    for (int d = 0; d < D; d++)
                    {
                        double o = 0;
                        for (int j = lo; j <= hi; j++)
                        {
                            o += Math.Exp(scores[j - lo] - max) / sum * values[slotOf[t]][j][hk * D + d];
                        }

                        maxDiff = Math.Max(maxDiff, Math.Abs(o - cpu[s][(t * Hq + h) * D + d]));
                    }
                }
            }
        }

        Assert.True(maxDiff < 1e-4, $"CPU lookahead attention differs from the direct computation by {maxDiff}");
    }
}
