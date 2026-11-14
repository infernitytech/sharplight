using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// States in a mixed compilation (Vulkan with CPU fallback), against the CPU oracle: the nodes of one state split into several segments of the
/// same device by the other device's nodes must all see one copy of it. Here an indexer key cache whose rows Vulkan does not support stays on
/// the CPU (its CacheWrite and IndexerTopK, a Vulkan op between them), and the attention key cache stays on Vulkan (its CacheWrite and the
/// attention over the selection, the CPU nodes between them). Prefill chunks and decode rows, so later steps read what earlier ones wrote.
/// </summary>
public sealed class StateHandoffTests
{
    private const int Heads = 2;
    private const int Dims = 32;
    private const int IndexDims = 12;   // not a multiple of 8: the Vulkan indexer leaves it to the CPU
    private const int TopK = 4;

    [Fact]
    public void SplitStatesMatchCpu()
    {
        Gpu.Require();
        var tg = new TG();
        var T = TG.T;
        var q = tg.InF("q", T, Heads, Dims);
        var k = tg.InF("k", T, 1, Dims);
        var ik = tg.InF("ik", T, 1, IndexDims);
        var iq = tg.InF("iq", T, Heads, IndexDims);
        var iw = tg.InF("iw", T, Heads);
        tg.B.RuntimeScalar("n_past");
        var keys = tg.State("keys", DataType.F32, StateKind.KeyCache, Dim.Sym("ctx"), 1, Dims);
        tg.B.CacheWrite(keys, k);
        var indexKeys = tg.State("index_keys", DataType.F32, StateKind.KeyCache, Dim.Sym("ctx"), 1, IndexDims);
        tg.B.CacheWrite(indexKeys, ik);
        var scaled = tg.B.Scale(iq, 0.5f);
        var picked = tg.B.Emit1("IndexerTopK", [scaled, indexKeys, iw], Attributes.From(("k", (long)TopK), ("offset", "n_past")));
        tg.B.Output("selection", picked);
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [q, keys, keys, picked],
            Attributes.From(("scale", 1f / MathF.Sqrt(Dims)), ("offset", "n_past"), ("selection", true))));
        var graph = tg.B.Build();

        int[] chunks = [6, 1, 1, 9, 1, 1];
        var steps = new List<Step>();
        int past = 0, seed = 40;
        Value In(string name) => graph.Inputs.Single(v => v.Name == name);
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past,
                (In("q"), Rnd.F((long)c * Heads * Dims, seed++, 2f)),
                (In("k"), Rnd.F((long)c * Dims, seed++)),
                (In("ik"), Rnd.F((long)c * IndexDims, seed++)),
                (In("iq"), Rnd.F((long)c * Heads * IndexDims, seed++)),
                (In("iw"), Rnd.F((long)c * Heads, seed++))));
            past += c;
        }

        Diff.Check("StateHandoff", graph, steps, atol: 1e-4, rtol: 1e-4, ctx: 64, cpuFallback: true, inspect: m =>
        {
            var kinds = m.Plan.Segments.Select(s => m.Plan.Slots[s.DeviceSlot].Device.Info.Kind).ToList();
            Assert.True(kinds.Count(x => x == DeviceKind.Cpu) >= 2 && kinds.Count(x => x != DeviceKind.Cpu) >= 2,
                "the placement should split both states: " + m.Plan.Describe());
        });
    }
}
