using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Compressed KV caches (<see cref="StateDefinition.Compression"/>), Vulkan against the CPU oracle: WindowPool (overlapping and plain windows,
/// ring state across passes), the compressed CacheWrite, IndexerTopK over compressed entries and CachedAttention over a window of raw keys plus
/// the compressed entries (all or the selected ones) with sinks, as one shared-KV attention layer. Single sequence, multi-sequence (equal slot
/// ranges) and paged (shuffled blocks of one pool, a slot that starts over); the steps mix prefill chunks of several windows, chunks shorter than a
/// window and single decode rows. With candidates, a group selection (blocks of entries scored by their best entry, the newest block kept) limits
/// the second selection to the entries of its blocks; ratio 1 stores one entry per position.
/// </summary>
public sealed class CompressedKvTests
{
    private const int Heads = 4;
    private const int Dims = 32;
    private const int IndexHeads = 4;
    private const int Window = 8;
    private const int TopK = 4;
    private const int Slots = 3;
    private const int SlotContext = 128;
    private const int BlockSize = 16;
    private const int PoolBlocks = 20;
    private const int CandidateBlock = 4;
    private const int CandidateBlocks = 3;

    [Theory]
    [InlineData("f32", 4, true, true, "single")]
    [InlineData("f16", 4, true, true, "single")]
    [InlineData("f32", 8, false, false, "single")]
    [InlineData("f16", 8, true, false, "single")]
    [InlineData("f16", 4, true, true, "multi")]
    [InlineData("f32", 8, false, false, "multi")]
    [InlineData("f16", 4, true, true, "paged")]
    [InlineData("f32", 8, false, false, "paged")]
    [InlineData("f32", 1, false, true, "single", true)]
    [InlineData("f16", 2, true, true, "single", true)]
    [InlineData("f32", 2, false, true, "multi", true)]
    [InlineData("f16", 1, false, true, "paged", true)]
    public void CompressedAttentionMatchesCpu(string cacheType, int ratio, bool overlap, bool selection, string mode, bool candidates = false)
    {
        Gpu.Require();
        var type = DataType.Get(cacheType);
        int channels = overlap ? 2 * Dims : Dims;
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, Heads, Dims);
        var ki = tg.InF("k", T, 1, Dims);
        var vals = tg.InF("values", T, channels);
        var scores = tg.InF("scores", T, channels);
        var pos = tg.In("pos", DataType.I32, T, 1);
        var keys = tg.State("keys", type, StateKind.KeyCache, Dim.Sym("ctx"), 1, Dims);
        tg.B.CacheWrite(keys, ki);
        tg.B.RuntimeScalar("n_past");

        var ring = tg.B.State(new StateDefinition
        {
            Name = "pool", Type = DataType.F32, Kind = StateKind.Recurrent, Shape = new Shape((overlap ? 2 : 1) * ratio, 2 * channels),
        });
        var pooled = tg.B.Emit("WindowPool", [vals, scores, tg.WF("bias", Rnd.F(ratio * channels, 7, 0.5f), ratio, channels), pos, ring],
            Attributes.From(("ratio", (long)ratio), ("overlap", overlap)), writes: [ring])[0];
        tg.B.Output("pooled", pooled);
        var entries = tg.B.State(new StateDefinition
        {
            Name = "entries", Type = type, Kind = StateKind.KeyCache, Compression = ratio,
            Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol).CeilDiv(ratio), 1, Dims),
        });
        tg.B.Emit("CacheWrite", [entries, tg.B.Reshape(pooled, 0, 1, Dims)], Attributes.From(("offset", "n_past"), ("compression", (long)ratio)),
            writes: [entries]);

        var attributes = Attributes.From(("scale", 1f / MathF.Sqrt(Dims)), ("offset", "n_past"), ("window", (long)Window), ("sinks", true),
            ("compression", (long)ratio));
        var inputs = new List<Value> { qi, keys, keys, tg.WF("sinks", Rnd.F(Heads, 9), Heads), entries };
        Value? iq = null, iw = null;
        if (selection)
        {
            iq = tg.InF("iq", T, IndexHeads, Dims);
            iw = tg.InF("iw", T, IndexHeads);
            var topK = Attributes.From(("k", (long)TopK), ("offset", "n_past"), ("compression", (long)ratio));
            Value picked;
            if (candidates)
            {
                var blocks = tg.B.Emit1("IndexerTopK", [iq, entries, iw], Attributes.From(("k", (long)CandidateBlocks * CandidateBlock),
                    ("offset", "n_past"), ("compression", (long)ratio), ("group", (long)CandidateBlock), ("group_score", "max"), ("local", 1L)));
                tg.B.Output("candidates", blocks);
                picked = tg.B.Emit1("IndexerTopK", [iq, entries, iw, blocks], topK.With("candidate_group", (long)CandidateBlock));
            }
            else
            {
                picked = tg.B.Emit1("IndexerTopK", [iq, entries, iw], topK);
            }

            tg.B.Output("selection", picked);
            inputs.Add(picked);
            attributes = attributes.With("selection", true);
        }

        tg.B.Output("o", tg.B.Emit1("CachedAttention", inputs, attributes));
        var graph = tg.B.Build();
        var steps = mode == "single" ? SingleSteps(graph, channels, selection) : MultiSteps(ref graph, channels, selection, mode == "paged");
        long ctx = mode switch { "single" => SlotContext, "multi" => Slots * SlotContext, _ => PoolBlocks * BlockSize };
        Diff.Check("CompressedKv", graph, steps, atol: 2e-3, rtol: 5e-3, maxTokens: 96, ctx: ctx,
            what: $"{cacheType} r={ratio} overlap={overlap} selection={selection} candidates={candidates} {mode}");
    }

    // Chunks of one sequence: several windows in one pass, windows spanning passes, single rows.
    private static List<Step> SingleSteps(ComputeGraph g, int channels, bool selection)
    {
        int[] chunks = [5, 3, 1, 1, 40, 1, 7, 9, 1, 2];
        var steps = new List<Step>();
        int past = 0, seed = 20;
        foreach (int c in chunks)
        {
            var positions = Enumerable.Range(past, c).ToArray();
            steps.Add(new Step(c, past, [.. Feeds(g, c, channels, selection, positions, ref seed)]));
            past += c;
        }

        return steps;
    }

    // Steps of several sequences (runs) in one batch; paged: blocks from a shuffled free list, slot 0 starts over at step 4.
    private static List<Step> MultiSteps(ref ComputeGraph g, int channels, bool selection, bool paged)
    {
        g = paged ? MultiSequence.Transform(g, Slots, BlockSize, SlotContext / BlockSize).Graph : MultiSequence.Transform(g, Slots).Graph;
        var graph = g;
        Value In(string name) => graph.Inputs.Single(v => v.Name == name);
        var script = new List<(int Slot, int Tokens)[]>
        {
            new[] { (0, 70), (1, 5) },
            new[] { (0, 1), (1, 1), (2, 40) },
            new[] { (0, 1), (1, 3), (2, 1) },
            new[] { (0, 2), (1, 50), (2, 1) },
            new[] { (0, 33), (2, 1) },
            new[] { (0, 1), (1, 1), (2, 7) },
        };
        const int ResetStep = 4;
        var free = new Stack<int>(Enumerable.Range(0, PoolBlocks).OrderBy(b => (b * 7) % PoolBlocks));
        var owned = Enumerable.Range(0, Slots).Select(_ => new List<int>()).ToArray();
        var length = new int[Slots];
        var steps = new List<Step>();
        int seed = 20;
        foreach (var (step, si) in script.Select((x, i) => (x, i)))
        {
            if (si == ResetStep)
            {
                owned[0].ForEach(free.Push);
                owned[0].Clear();
                length[0] = 0;
            }

            int rows = step.Sum(x => x.Tokens);
            var seqRows = new int[rows * 4];
            var runs = new int[step.Length * 4];
            var positions = new int[rows];
            int r = 0;
            for (int u = 0; u < step.Length; u++)
            {
                var (slot, n) = step[u];
                while (owned[slot].Count * BlockSize < length[slot] + n)
                {
                    owned[slot].Add(free.Pop());
                }

                for (int i = 0; i < n; i++, r++)
                {
                    positions[r] = length[slot] + i;
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

            var inputs = Feeds(g, rows, channels, selection, positions, ref seed);
            inputs.Add((In(MultiSequence.RowsInput), seqRows));
            inputs.Add((In("sequence_runs"), runs));
            if (paged)
            {
                var table = new int[Slots * (SlotContext / BlockSize)];
                Array.Fill(table, -1);
                for (int s = 0; s < Slots; s++)
                {
                    for (int j = 0; j < owned[s].Count; j++)
                    {
                        table[s * (SlotContext / BlockSize) + j] = owned[s][j];
                    }
                }

                inputs.Add((In(MultiSequence.BlockTableInput), table));
            }

            steps.Add(new Step(rows, 0, [.. inputs]) { Symbols = new Dictionary<string, long> { [WellKnown.RunSymbol] = step.Length } });
        }

        return steps;
    }

    // Random rows of the graph's per-token inputs, with the given positions.
    private static List<(Value, Array)> Feeds(ComputeGraph g, int rows, int channels, bool selection, int[] positions, ref int seed)
    {
        Value In(string name) => g.Inputs.Single(v => v.Name == name);
        var feeds = new List<(Value, Array)>
        {
            (In("q"), Rnd.F((long)rows * Heads * Dims, seed++, 2f)),
            (In("k"), Rnd.F((long)rows * Dims, seed++)),
            (In("values"), Rnd.F((long)rows * channels, seed++)),
            (In("scores"), Rnd.F((long)rows * channels, seed++, 2f)),
            (In("pos"), positions),
        };
        if (selection)
        {
            feeds.Add((In("iq"), Rnd.F((long)rows * IndexHeads * Dims, seed++)));
            feeds.Add((In("iw"), Rnd.F((long)rows * IndexHeads, seed++)));
        }

        return feeds;
    }
}
