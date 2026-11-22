using System.Runtime.InteropServices;
using Sharplight.Cpu;
using Sharplight.Cuda.Interop;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// A placement of many device segments under a device memory limit (<see cref="DeviceMemoryLimits.DeviceMemory"/>): every layer's routed
/// experts that do not fit go to the CPU, which splits the layers into a device segment each. Each segment costs device memory beyond its
/// buffers (its stream, the instantiated graphs of its recorded steps), which the placement keeps free, so the device holds no more than the
/// limit after decode steps that capture graphs and a prompt pass (Kimi-K3's ~185 segments once failed with <c>cuStreamCreate: out of memory</c>
/// at the first capture).
/// </summary>
public sealed unsafe class ManySegmentPlacementTests
{
    private const int Hidden = 128;
    private const int Experts = 16;
    private const int Slots = 2;
    private const int Inner = 1024;
    private const int Layers = 96;

    private static ComputeGraph BuildGraph(out Value x)
    {
        var tg = new TG();
        x = tg.InF("x", TG.T, Hidden);
        var h = x;
        for (int layer = 0; layer < Layers; layer++)
        {
            using var _ = tg.B.Group($"layers.{layer}");
            h = tg.B.MatMul(h, tg.WF($"dense{layer}", Rnd.F(Hidden * Hidden, layer, 0.05f), Hidden, Hidden));
            var logits = tg.B.MatMul(h, tg.WF($"router{layer}", Rnd.F(Experts * Hidden, 300 + layer, 1f), Experts, Hidden));
            var top = tg.B.Emit("TopK", [logits], Attributes.From(("k", (long)Slots)));
            var slotWeightRows = tg.B.Reshape(tg.B.Emit1("Softmax", [top[0]]), 0, Slots, 1);
            var up = tg.B.Emit1("MatMulId", [h, tg.WF($"up{layer}", Rnd.F(Experts * Inner * Hidden, 1000 + layer, 0.05f), Experts, Inner, Hidden), top[1]]);
            var act = tg.B.Emit1("Silu", [up]);
            var down = tg.B.Emit1("MatMulId", [act, tg.WF($"down{layer}", Rnd.F(Experts * Hidden * Inner, 2000 + layer, 0.05f), Experts, Hidden, Inner), top[1]]);
            var summed = tg.B.Emit1("ReduceSum", [tg.B.Permute(tg.B.Mul(down, slotWeightRows), 0, 2, 1)]);
            h = tg.B.Add(h, tg.B.Reshape(summed, 0, Hidden));
        }

        tg.B.Output("y", h);
        return tg.B.Build();
    }

    /// <summary>Device memory in use on the GPU (every process; the test holds the GPU alone).</summary>
    private static long DeviceUsage(CudaDevice device)
    {
        device.Enter();
        device.Synchronize();
        nuint free;
        nuint total;
        CudaApi.Check(CudaApi.cuMemGetInfo(&free, &total), "cuMemGetInfo");
        return (long)total - (long)free;
    }

    /// <summary>
    /// Compiles <see cref="BuildGraph"/> on a CUDA device limited to <paramref name="limit"/> bytes with the CPU as the fallback, runs decode
    /// steps (the second run of a shape captures its graph) and prompt passes, and returns the device segments and the device memory the run
    /// holds beyond what the GPU held before (every process: the test holds the GPU alone).
    /// </summary>
    private static (int DeviceSegments, long Held, long Buffers) Run(long limit, long? reservePerSegment)
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        var options = new DeviceOptions { UseCompilationCache = false };
        options.MemoryLimits.DeviceMemory = limit;
        using var device = (CudaDevice)provider.OpenDevice(provider.GetDevices()[0], options);
        var cpuProvider = new CpuExecutionProvider();
        using var cpu = cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]);
        long before = DeviceUsage(device);
        var graph = BuildGraph(out var x);
        using var model = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Fallback = new DeviceSlot { Device = cpu },
            Compile = new CompileOptions { MaxTokens = 16, ContextLength = 16 },
            ReservePerDevice = 16L << 20,
            ReservePerSegment = reservePerSegment,
            StreamExpertsMinRows = 0,
        });
        foreach (int rows in new[] { 1, 1, 1, 1, 8, 8, 8, 1, 1 })
        {
            var input = Rnd.F(rows * Hidden, 3 + rows, 1f);
            model.SetInput(x, MemoryMarshal.AsBytes(input.AsSpan()));
            var parameters = new RunParameters();
            parameters.Symbols["T"] = rows;
            model.Run(parameters);
            var y = new float[rows * Hidden];
            model.GetOutput<float>("y", y);
            Assert.All(y, v => Assert.True(float.IsFinite(v)));
        }

        return (model.Plan.Segments.Count(s => s.DeviceSlot == 0), DeviceUsage(device) - before, device.GetMemoryInfo().UsedByUs);
    }

    /// <summary>
    /// With the device's own estimate of a segment's overhead (<see cref="IComputeDevice.SegmentOverheadBytes"/>, the default of
    /// <see cref="PlacementOptions.ReservePerSegment"/>) the experts go to the CPU, the layers stay on the GPU, and the run stays within the limit.
    /// </summary>
    [Fact]
    public void ManySegmentsStayWithinTheDeviceLimit()
    {
        const long limit = 512L << 20;
        var (segments, held, buffers) = Run(limit, reservePerSegment: null);
        Assert.True(segments >= 40, $"{segments} device segments");
        Assert.True(held <= limit, $"{segments} device segments hold {held >> 20} MiB after the steps, beyond the {limit >> 20} MiB limit ({buffers >> 20} MiB counted)");
    }

    /// <summary>
    /// A placement that keeps nothing free for its segments fills the limit with experts; the segments' streams and step graphs are counted
    /// against the limit (a compilation whose streams do not fit is placed again with more kept free), and the steps whose graphs no longer
    /// fit run launch by launch: the device still holds no more than the limit.
    /// </summary>
    [Fact]
    public void SegmentsThatDoNotFitTheirGraphsRunWithout()
    {
        const long limit = 160L << 20;
        var (segments, held, buffers) = Run(limit, reservePerSegment: 0);
        Assert.True(held <= limit, $"{segments} device segments hold {held >> 20} MiB after the steps, beyond the {limit >> 20} MiB limit ({buffers >> 20} MiB counted)");
    }
}
