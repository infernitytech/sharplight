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
    private const int ThreeSlots = 3;
    private const int ThreeSlotContext = 256;
    private const int ThreeSlotPoolBlocks = 40;   // < slots * context / BlockSize: the slots share the pool

    [Theory]
    [InlineData("f16", 64, 64, 8, 2)]
    [InlineData("f16", 64, 64, 16, 2)]   // 8 heads per KV head: the tensor-core decode kernel on CUDA
    [InlineData("f16", 128, 128, 16, 4)]
    [InlineData("f16", 256, 256, 24, 4)]   // head sizes 256 / 512: the wide form of the tensor-core decode kernel on CUDA
    [InlineData("f16", 512, 512, 32, 4)]
    [InlineData("q8_0", 64, 64, 8, 2)]
    [InlineData("f32", 64, 64, 4, 4)]
    [InlineData("f16", 576, 512, 16, 1)]   // latent attention (fa_mqa)
    [InlineData("q4_0", 288, 256, 40, 1)]
    [InlineData("f16", 12, 12, 4, 2)]   // head size not a multiple of 8
    public void PagedAttentionMatchesCpu(string cacheType, int D, int valueDims, int Hq, int Hkv)
    {
        var script = new List<(int Slot, int Tokens)[]>
        {
            new[] { (0, 70), (1, 5) },
            new[] { (0, 1), (1, 1), (2, 40) },
            new[] { (0, 1), (1, 1), (2, 1) },
            new[] { (0, 2), (1, 130), (2, 1) },
            new[] { (0, 33), (2, 1) },   // slot 0: new sequence
            new[] { (0, 1), (1, 1), (2, 1) },
        };
        Check(cacheType, D, valueDims, Hq, Hkv, ThreeSlots, ThreeSlotContext, ThreeSlotPoolBlocks, script, resetStep: 4, maxTokens: 192);
    }

    /// <summary>
    /// More sequences than the decode kernel's rows (64 on CUDA): short prompts of all of them (the mixed form, every run short), a
    /// decode row each (no run long enough for flash attention: its launch runs no blocks), decode rows next to one prompt chunk of 40
    /// rows (both halves of the mixed form), then decode rows again.
    /// </summary>
    [Theory]
    [InlineData("f16", 128, 128, 16, 4)]
    [InlineData("f16", 64, 64, 8, 2)]
    [InlineData("f16", 256, 256, 8, 2)]
    [InlineData("q8_0", 64, 64, 8, 2)]
    public void ManySequencesDecodeOneRowEach(string cacheType, int D, int valueDims, int Hq, int Hkv)
    {
        const int slots = 72;
        var prompts = Enumerable.Range(0, slots).Select(s => (s, 3 + s % 5)).ToArray();
        var decode = Enumerable.Range(0, slots).Select(s => (s, 1)).ToArray();
        var mixed = Enumerable.Range(0, slots).Select(s => (s, s == slots - 1 ? 40 : 1)).ToArray();
        var script = new List<(int Slot, int Tokens)[]> { prompts, decode, mixed, decode };
        Check(cacheType, D, valueDims, Hq, Hkv, slots, 64, slots * 4, script, resetStep: -1, maxTokens: 512);
    }

    private static void Check(string cacheType, int D, int valueDims, int Hq, int Hkv, int slots, int slotContext, int poolBlocks,
        List<(int Slot, int Tokens)[]> script, int resetStep, int maxTokens)
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
        var g = MultiSequence.Transform(tg.B.Build(), slots, BlockSize, slotContext / BlockSize).Graph;
        Value In(string name) => g.Inputs.Single(v => v.Name == name);

        // Blocks are handed out from a shuffled free list; slot 0 starts over at resetStep (its blocks return to the list).
        var free = new Stack<int>(Enumerable.Range(0, poolBlocks).OrderBy(b => (b * 17) % poolBlocks));
        var owned = Enumerable.Range(0, slots).Select(_ => new List<int>()).ToArray();
        var length = new int[slots];
        var steps = new List<Step>();
        int seed = 10;
        foreach (var (step, si) in script.Select((x, i) => (x, i)))
        {
            if (si == resetStep)
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

            var table = new int[slots * (slotContext / BlockSize)];
            Array.Fill(table, -1);
            for (int s = 0; s < slots; s++)
            {
                for (int j = 0; j < owned[s].Count; j++)
                {
                    table[s * (slotContext / BlockSize) + j] = owned[s][j];
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
        Diff.Check("CachedAttention.paged", g, steps, atol: atol, rtol: rtol, maxTokens: maxTokens, ctx: poolBlocks * BlockSize,
            what: $"{cacheType} D={D} v_dims={valueDims} Hq={Hq} Hkv={Hkv}");
    }
}
