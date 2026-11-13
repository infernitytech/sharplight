using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Paged KV caches (<see cref="MultiSequence.Transform(ComputeGraph, int, int, int)"/> with a block size): CacheWrite and
/// CachedAttention of several sequences whose positions live in shuffled, non-contiguous blocks of one pool, Vulkan against the
/// CPU oracle. The steps mix prefill chunks of several tiles (flash kernels), short chunks and single decode rows (split-K),
/// and a slot that starts over in blocks another slot released.
/// </summary>
public sealed class PagedAttentionTests
{
    private const int BlockSize = 16;
    private const int Slots = 3;
    private const int SlotContext = 256;
    private const int PoolBlocks = 40;   // < Slots * SlotContext / BlockSize: the slots share the pool

    [Theory]
    [InlineData("f16", 64, 64, 8, 2)]
    [InlineData("q8_0", 64, 64, 8, 2)]
    [InlineData("f32", 64, 64, 4, 4)]
    [InlineData("f16", 576, 512, 16, 1)]   // latent attention (fa_mqa)
    [InlineData("q4_0", 288, 256, 40, 1)]
    public void PagedAttentionMatchesCpu(string cacheType, int D, int valueDims, int Hq, int Hkv)
    {
        Gpu.Require();
        bool latent = valueDims != D;
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var type = DataType.Get(cacheType);
        var kc = tg.State("kc", type, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki);
        tg.B.RuntimeScalar("n_past");
        var attrs = new List<(string, object)> { ("scale", 1f / MathF.Sqrt(D)), ("offset", "n_past") };
        Value vc = kc;
        if (latent)
        {
            attrs.Add(("v_dims", (long)valueDims));
        }
        else
        {
            var vi = tg.InF("v", T, Hkv, D);
            vc = tg.State("vc", type, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
            tg.B.CacheWrite(vc, vi);
        }

        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc], Attributes.From([.. attrs])));
        var g = MultiSequence.Transform(tg.B.Build(), Slots, BlockSize, SlotContext / BlockSize).Graph;
        Value In(string name) => g.Inputs.Single(v => v.Name == name);

        // Blocks are handed out from a shuffled free list; slot 0 starts over at step 4 (its blocks return to the list).
        var free = new Stack<int>(Enumerable.Range(0, PoolBlocks).OrderBy(b => (b * 17) % PoolBlocks));
        var owned = Enumerable.Range(0, Slots).Select(_ => new List<int>()).ToArray();
        var length = new int[Slots];
        var script = new List<(int Slot, int Tokens)[]>
        {
            new[] { (0, 70), (1, 5) },
            new[] { (0, 1), (1, 1), (2, 40) },
            new[] { (0, 1), (1, 1), (2, 1) },
            new[] { (0, 2), (1, 130), (2, 1) },
            new[] { (0, 33), (2, 1) },   // slot 0: new sequence
            new[] { (0, 1), (1, 1), (2, 1) },
        };
        const int ResetStep = 4;
        var steps = new List<Step>();
        int seed = 10;
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

            var table = new int[Slots * (SlotContext / BlockSize)];
            Array.Fill(table, -1);
            for (int s = 0; s < Slots; s++)
            {
                for (int j = 0; j < owned[s].Count; j++)
                {
                    table[s * (SlotContext / BlockSize) + j] = owned[s][j];
                }
            }

            var inputs = new List<(Value, Array)>
            {
                (In("q"), Rnd.F((long)rows * Hq * D, seed++, 4f)),
                (In("k"), Rnd.F((long)rows * Hkv * D, seed++)),
                (In(MultiSequence.RowsInput), seqRows),
                (In("sequence_runs"), runs),
                (In(MultiSequence.BlockTableInput), table),
            };
            if (!latent)
            {
                inputs.Add((In("v"), Rnd.F((long)rows * Hkv * D, seed++)));
            }

            steps.Add(new Step(rows, 0, [.. inputs]) { Symbols = new Dictionary<string, long> { [WellKnown.RunSymbol] = step.Length } });
        }

        var (atol, rtol) = cacheType switch { "q8_0" => (5e-3, 1e-2), "q4_0" => (3e-2, 5e-2), _ => (2e-3, 5e-3) };
        Diff.Check("CachedAttention.paged", g, steps, atol: atol, rtol: rtol, maxTokens: 192, ctx: PoolBlocks * BlockSize,
            what: $"{cacheType} D={D} v_dims={valueDims} Hq={Hq} Hkv={Hkv}");
    }
}
