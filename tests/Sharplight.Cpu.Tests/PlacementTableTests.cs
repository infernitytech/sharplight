using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// Placement of lookup tables (weights only <c>Gather</c> reads): with a fallback device a table that does not fit the device goes to
/// the fallback on its own, and the layers after it keep filling the device.
/// </summary>
public sealed class PlacementTableTests
{
    private const int Hidden = 8;
    private const int TableRows = 1024;

    private static ComputeGraph BuildGraph(out Value x, out Value ids)
    {
        var tg = new TestGraph();
        x = tg.InF("x", TestGraph.T, Hidden);
        ids = tg.In("ids", DataType.I32, TestGraph.T);
        var h = x;
        for (int layer = 0; layer < 3; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            if (layer == 1)
            {
                var table = tg.WF("table", Kit.Rand(TableRows * Hidden, 7), TableRows, Hidden);
                h = tg.B.Add(h, tg.B.Gather(table, ids));
            }

            h = tg.B.MatMul(h, tg.WF($"w{layer}", Kit.Rand(Hidden * Hidden, layer), Hidden, Hidden));
        }

        tg.B.Output("y", h);
        return tg.B.Build();
    }

    [Fact]
    public void TableThatDoesNotFit_GoesToTheFallback_AndTheLayersAfterItStayOnTheDevice()
    {
        var graph = BuildGraph(out var x, out var ids);
        using var device = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        using var fallback = Kit.Provider.OpenDevice(Kit.Provider.GetDevices()[0]);
        var compile = new CompileOptions { MaxTokens = 4, ContextLength = 4, DisabledFeatures = new HashSet<string> { DeviceFeatures.IntDot } };
        var options = new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device, MemoryLimit = 4096 }],   // the three 256-byte layers, not the 32 KiB table
            Fallback = new DeviceSlot { Device = fallback },
            Compile = compile,
            ReservePerDevice = 0,
        };

        var plan = PlacementPlanner.Plan(graph, options);

        Assert.Equal(1, plan.GroupDevice["layers.1" + PlacementPlanner.TableGroupSuffix]);
        Assert.All(new[] { "layers.0", "layers.1", "layers.2" }, g => Assert.Equal(0, plan.GroupDevice[g]));

        float[] input = Kit.Rand(2 * Hidden, 3);
        int[] rows = [5, 1000];
        float[] Run(CompiledModel m)
        {
            Kit.Set(m, x, input);
            Kit.Set(m, ids, rows);
            var p = Kit.P(2);
            m.Run(p);
            return Kit.Out(m, "y", p);
        }

        using var split = CompiledModel.Compile(graph, options);
        using var whole = Kit.Compile(graph);
        Kit.AssertClose(Run(whole), Run(split));
    }
}
