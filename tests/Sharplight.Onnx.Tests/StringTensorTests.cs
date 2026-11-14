using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// String tensors (DataType.Str): the CompiledModel string API, the run-owned string heap, the CPU string kernels and the planner keeping
/// string nodes on the CPU when the primary device is a GPU.
/// </summary>
public class StringTensorTests
{
    [Fact]
    public void StringGraph_RunsOnTheCpu()
    {
        var (concat, equal, ids) = Run(Providers.Cpu, ["red", "blue", "x"]);
        Assert.Equal(["red!", "blue!", "x!"], concat);
        Assert.Equal([1f, 0f, 0f], equal);
        Assert.Equal([1, 2, -1], ids);
    }

    [Fact]
    public void StringNodes_StayOnTheCpuWithAGpuDevice()
    {
        var device = Providers.VulkanOrSkip();
        var (concat, equal, ids) = Run(device, ["blue", "red", "red"]);
        Assert.Equal(["blue!", "red!", "red!"], concat);
        Assert.Equal([0f, 1f, 1f], equal);
        Assert.Equal([2, 1, 1], ids);
    }

    [Fact]
    public void StringHeap_InternsAndClears()
    {
        var heap = new StringHeap();
        int a = heap.Add("a");
        Assert.Equal(a, heap.Add("a"));
        Assert.NotEqual(a, heap.Add("b"));
        Assert.Equal("b", heap[heap.Add("b")]);
        heap.Clear();
        Assert.Equal(0, heap.Count);
        Assert.Throws<ArgumentOutOfRangeException>(() => heap[a]);
    }

    /// <summary>x + "!" (concat), x == "red" (equal), and x looked up in {red: 1, blue: 2} (default -1), run twice with fresh inputs.</summary>
    private static (string[] Concat, float[] Equal, int[] Ids) Run(IComputeDevice device, string[] input)
    {
        var b = new GraphBuilder("strings", _ => null);
        var x = b.Input("x", InputRole.Other, DataType.Str, new Shape(3));
        var bang = b.Emit1("StringConst", [], Attributes.From(("strings", new[] { "!" }), ("shape", new long[] { 1 })));
        var red = b.Emit1("StringConst", [], Attributes.From(("strings", new[] { "red" }), ("shape", new long[] { 1 })));
        b.Output("concat", b.Emit1("StringBinary", [x, bang], Attributes.From(("mode", "concat"))));
        b.Output("equal", b.Emit1("StringBinary", [x, red], Attributes.From(("mode", "equal"))));
        b.Output("ids", b.Emit1("LabelEncode", [x], Attributes.From(
            ("keys", new[] { "red", "blue" }), ("values", new[] { 1.0, 2.0 }), ("default", -1.0), ("type", "i32"))));
        var graph = b.Build();
        using var compiled = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Fallback = device.Info.Kind == DeviceKind.Cpu ? null : new DeviceSlot { Device = Providers.Cpu },
            Compile = new CompileOptions(),
        });
        var run = new RunParameters();
        compiled.SetInput(x, ["stale", "stale", "stale"]);
        compiled.Run(run);
        compiled.SetInput(x, input);
        compiled.Run(run);
        var concat = new string[3];
        compiled.GetOutput("concat", concat.AsSpan());
        var mask = new float[3];
        compiled.GetOutput<float>("equal", mask);
        var ids = new int[3];
        compiled.GetOutput<int>("ids", ids);
        return (concat, mask, ids);
    }
}
