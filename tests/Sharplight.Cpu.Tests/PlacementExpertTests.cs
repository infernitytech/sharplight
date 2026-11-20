using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Placement of routed experts: with a fallback device the experts of a mixture-of-experts layer (the expert-indexed products, the
/// activation between them and the weighted sum over the slots) form a group of their own; when the model does not fit, the experts of
/// the first layers go to the fallback, as few as needed (and the experts that then still do not fit), and the dense weights stay on the
/// device.
/// </summary>
public sealed class PlacementExpertTests
{
    private const int Hidden = 8;
    private const int Experts = 4;
    private const int Slots = 2;
    private const int Inner = 16;
    private const int Layers = 3;
    private const long DenseBytes = Hidden * Hidden * sizeof(float);
    private const long ExpertBytes = 2L * Experts * Inner * Hidden * sizeof(float);

    private static ComputeGraph BuildGraph(out Value x, out Value ids, out Value slotWeights)
    {
        var tg = new TestGraph();
        x = tg.InF("x", TestGraph.T, Hidden);
        ids = tg.In("ids", DataType.I32, TestGraph.T, Slots);
        slotWeights = tg.InF("slot_weights", TestGraph.T, Slots);
        var h = x;
        for (int layer = 0; layer < Layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, tg.WF($"dense{layer}", Kit.Rand(Hidden * Hidden, layer, 0.5f), Hidden, Hidden));
            var slotWeightRows = tg.B.Reshape(slotWeights, 0, Slots, 1);   // before the experts, as the decoder's MoE block builds it
            var up = tg.B.Emit1("MatMulId", [h, tg.WF($"up{layer}", Kit.Rand(Experts * Inner * Hidden, 10 + layer, 0.5f), Experts, Inner, Hidden), ids]);
            var act = tg.B.Mul(up, up);
            var down = tg.B.Emit1("MatMulId", [act, tg.WF($"down{layer}", Kit.Rand(Experts * Hidden * Inner, 20 + layer, 0.5f), Experts, Hidden, Inner), ids]);
            var weighted = tg.B.Mul(down, slotWeightRows);
            var summed = tg.B.Emit1("ReduceSum", [tg.B.Permute(weighted, 0, 2, 1)]);
            h = tg.B.Add(h, tg.B.Reshape(summed, 0, Hidden));
        }

        tg.B.Output("y", h);
        return tg.B.Build();
    }

    private static PlacementOptions Options(IComputeDevice device, IComputeDevice fallback, long memoryLimit, bool splitExperts = true) => new()
    {
        Devices = [new DeviceSlot { Device = device, MemoryLimit = memoryLimit }],
        Fallback = new DeviceSlot { Device = fallback },
        Compile = new CompileOptions { MaxTokens = 4, ContextLength = 4, DisabledFeatures = new HashSet<string> { DeviceFeatures.IntDot } },
        ReservePerDevice = 0,
        SplitExperts = splitExperts,
    };

    [Fact]
    public void ExpertsOfTheFirstLayersGoToTheFallback_AndTheDenseWeightsStayOnTheDevice()
    {
        var graph = BuildGraph(out var x, out var ids, out var slotWeights);
        using var device = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        using var fallback = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        var options = Options(device, fallback, Layers * DenseBytes + ExpertBytes + 2048);   // the dense layers, one layer of experts and the transfer buffers

        var plan = PlacementPlanner.Plan(graph, options);

        // The first layer's experts are moved out; the last layer's experts then are the ones that no longer fit.
        string Experts(int layer) => $"layers.{layer}{PlacementPlanner.ExpertGroupSuffix}";
        Assert.Equal(1, plan.GroupDevice[Experts(0)]);
        Assert.Equal(0, plan.GroupDevice[Experts(1)]);
        Assert.Equal(1, plan.GroupDevice[Experts(2)]);
        Assert.All(Enumerable.Range(0, Layers), l => Assert.Equal(0, plan.GroupDevice[$"layers.{l}"]));
        // The fallback returns the summed rows: the slot weighting and its sum run there.
        Assert.All(plan.Segments.Where(s => s.DeviceSlot == 1), s => Assert.Contains(s.Nodes, n => n.Op.Name == "ReduceSum"));
        Assert.Equal(5, plan.Segments.Count);   // device, experts 0, device, experts 2, device (the last residual add)

        float[] input = Kit.Rand(3 * Hidden, 3);
        int[] rows = [0, 3, 1, 1, 2, 0];
        float[] weights = [0.75f, 0.25f, 0.5f, 0.5f, 1f, 0f];
        float[] Run(CompiledModel m)
        {
            Kit.Set(m, x, input);
            Kit.Set(m, ids, rows);
            Kit.Set(m, slotWeights, weights);
            var p = Kit.P(3);
            m.Run(p);
            return Kit.Out(m, "y", p);
        }

        using var split = CompiledModel.Compile(graph, options);
        using var whole = Kit.Compile(graph);
        Kit.AssertClose(Run(whole), Run(split));
    }

    [Fact]
    public void ModelThatFits_KeepsEveryGroupOnTheDevice()
    {
        var graph = BuildGraph(out _, out _, out _);
        using var device = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        using var fallback = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);

        var plan = PlacementPlanner.Plan(graph, Options(device, fallback, Layers * (DenseBytes + ExpertBytes)));

        Assert.All(plan.GroupDevice.Values, slot => Assert.Equal(0, slot));
        Assert.Single(plan.Segments);
    }

    [Fact]
    public void WithoutExpertSplitting_WholeLayersGoToTheFallback()
    {
        var graph = BuildGraph(out _, out _, out _);
        using var device = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        using var fallback = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);

        var plan = PlacementPlanner.Plan(graph, Options(device, fallback, Layers * DenseBytes + ExpertBytes, splitExperts: false));

        Assert.DoesNotContain(plan.GroupDevice.Keys, g => g.EndsWith(PlacementPlanner.ExpertGroupSuffix, StringComparison.Ordinal));
        Assert.Equal(0, plan.GroupDevice["layers.0"]);
        Assert.Equal(1, plan.GroupDevice["layers.2"]);
    }
}
