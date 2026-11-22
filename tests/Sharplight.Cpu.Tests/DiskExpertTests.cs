using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Runtime.Experts;
using Sharplight.Tensors;
using Sharplight.Tests;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Routed experts left on disk (<see cref="CompileOptions.DiskExperts"/>): the placement leaves the experts it puts on the CPU fallback on disk
/// when they do not fit the host budget or the cache asked for, and the CPU's expert products (a gated gate / up pair and the down product
/// with its weighted slot sum) give bit for bit the outputs of the resident experts, for decode-sized passes (all experts held at once) and
/// for passes that select more experts than the cache holds (chunks).
/// </summary>
public sealed class DiskExpertTests
{
    private const int Hidden = 64;
    private const int Inner = 64;
    private const int Slots = 2;
    private const int Layers = 2;

    private static ComputeGraph BuildGraph(FileWeights file, int experts, bool quantized, out Value x, out Value ids, out Value slotWeights, out long denseBytes)
    {
        var tg = new TestGraph();
        Value Weight(string name, int seed, params long[] dims)
        {
            var data = Kit.Rand((int)dims.Aggregate(1L, (a, b) => a * b), seed, 0.2f);
            var info = quantized ? file.Add(name, DataType.Q8_0, Quantizers.Q8_0(data), dims) : file.AddF32(name, data, dims);
            tg.Weights[name] = info;
            return tg.B.Weight(name);
        }

        x = tg.InF("x", TestGraph.T, Hidden);
        ids = tg.In("ids", DataType.I32, TestGraph.T, Slots);
        slotWeights = tg.InF("slot_weights", TestGraph.T, Slots);
        var h = x;
        for (int layer = 0; layer < Layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, tg.WF($"dense{layer}", Kit.Rand(Hidden * Hidden, layer, 0.2f), Hidden, Hidden));
            var slotWeightRows = tg.B.Reshape(slotWeights, 0, Slots, 1);
            var gate = tg.B.Emit1("MatMulId", [h, Weight($"gate{layer}", 10 + layer, experts, Inner, Hidden), ids]);
            var up = tg.B.Emit1("MatMulId", [h, Weight($"up{layer}", 20 + layer, experts, Inner, Hidden), ids]);
            var act = tg.B.Mul(tg.B.Emit1("Silu", [gate]), up);
            var down = tg.B.Emit1("MatMulId", [act, Weight($"down{layer}", 30 + layer, experts, Hidden, Inner), ids]);
            var summed = tg.B.Emit1("ReduceSum", [tg.B.Permute(tg.B.Mul(down, slotWeightRows), 0, 2, 1)]);
            h = tg.B.Add(h, tg.B.Reshape(summed, 0, Hidden));
        }

        tg.B.Output("y", h);
        denseBytes = Layers * Hidden * Hidden * sizeof(float);
        return tg.B.Build();
    }

    /// <summary>The dense layers on the device, every layer's experts on the fallback (the device's limit leaves no room for them).</summary>
    private static PlacementOptions Options(IComputeDevice device, IComputeDevice fallback, long denseBytes, bool intDot, ExpertStore? store, long? fallbackLimit = null) => new()
    {
        Devices = [new DeviceSlot { Device = device, MemoryLimit = denseBytes + (64 << 10) }],
        Fallback = new DeviceSlot { Device = fallback, MemoryLimit = fallbackLimit },
        Compile = new CompileOptions
        {
            MaxTokens = 64,
            ContextLength = 4,
            DisabledFeatures = intDot ? new HashSet<string>() : new HashSet<string> { DeviceFeatures.IntDot },
            DiskExperts = store,
        },
        ReservePerDevice = 0,
        ReservePerSegment = 0,
    };

    [Theory]
    [InlineData(8, 1, false, false)]
    [InlineData(8, 3, false, false)]
    [InlineData(8, 40, false, false)]
    [InlineData(160, 1, true, true)]
    [InlineData(160, 4, true, true)]
    [InlineData(160, 40, true, true)]   // up to 80 experts per product: more than a chunk (16) of the smallest cache
    [InlineData(160, 40, false, false)]
    [InlineData(160, 40, true, false)]
    public void DiskExpertsMatchTheResidentExperts(int experts, int rows, bool quantized, bool intDot)
    {
        using var file = new FileWeights();
        var graph = BuildGraph(file, experts, quantized, out var x, out var ids, out var slotWeights, out long denseBytes);
        using var device = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        using var fallback = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        // The smallest cache: 64 slots per pool, fewer than one layer's experts.
        using var store = new ExpertStore(cacheBytes: 1, ExpertReaders.CreateThreads(4));

        var input = Kit.Rand(rows * Hidden, 3);
        var rowIds = Enumerable.Range(0, rows * Slots).Select(i => (i * 37 + i / Slots * 11) % experts).ToArray();
        for (int t = 0; t < rows; t++)
        {
            if (rowIds[t * Slots] == rowIds[t * Slots + 1])
            {
                rowIds[t * Slots + 1] = (rowIds[t * Slots + 1] + 1) % experts;
            }
        }

        var weights = Enumerable.Range(0, rows * Slots).Select(i => 0.25f + 0.5f * (i % 2)).ToArray();
        float[] Run(CompiledModel m)
        {
            Kit.Set(m, x, input);
            Kit.Set(m, ids, rowIds);
            Kit.Set(m, slotWeights, weights);
            var p = Kit.P(rows);
            m.Run(p);
            return Kit.Out(m, "y", p);
        }

        using var resident = CompiledModel.Compile(graph, Options(device, fallback, denseBytes, intDot, null));
        using var disk = CompiledModel.Compile(graph, Options(device, fallback, denseBytes, intDot, store));
        Assert.Empty(resident.Plan.DiskExpertGroups);
        Assert.Equal(Layers, disk.Plan.DiskExpertGroups.Count);
        Assert.True(store.IsActive);
        Assert.Equal(Layers * 3, store.DiskBackedWeights);
        Assert.True(disk.ResidentMemoryBytes < resident.ResidentMemoryBytes, "the experts on disk are not resident");

        var expected = Run(resident);
        Assert.Equal(expected, Run(disk));
        var first = store.Statistics;
        Assert.True(first.Misses > 0);
        // A second pass finds what the cache kept (or, cycling through more experts than it holds, what it read ahead) and gives the same result.
        Assert.Equal(expected, Run(disk));
        var second = store.Statistics;
        Assert.True(second.Hits + second.ReadAhead > first.Hits + first.ReadAhead);
    }

    [Theory]
    [InlineData(null, null, false)]          // the experts fit the host
    [InlineData(1L, null, true)]             // a cache smaller than the experts
    [InlineData(null, 1L << 30, true)]       // a host budget below the reserve
    public void ThePlacementLeavesTheCpuExpertsOnDiskWhenTheyDoNotFit(long? cache, long? fallbackLimit, bool disk)
    {
        using var file = new FileWeights();
        var graph = BuildGraph(file, 8, quantized: false, out _, out _, out _, out long denseBytes);
        using var device = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        using var fallback = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        using var store = new ExpertStore(cache, ExpertReaders.CreateThreads(1));

        var plan = PlacementPlanner.Plan(graph, Options(device, fallback, denseBytes, intDot: false, store, fallbackLimit));

        Assert.All(Enumerable.Range(0, Layers), l => Assert.Equal(1, plan.GroupDevice[$"layers.{l}{PlacementPlanner.ExpertGroupSuffix}"]));
        Assert.Equal(disk ? Layers : 0, plan.DiskExpertGroups.Count);
        Assert.Equal(disk, store.IsActive);
        Assert.Empty(plan.StreamedSegments);
    }
}
