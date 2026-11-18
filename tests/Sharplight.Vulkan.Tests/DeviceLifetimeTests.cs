using System.Runtime.InteropServices;
using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Opening and disposing Vulkan devices and models repeatedly, the way applications that load and unload models do: every
/// disposed device must release its driver context and memory, so the loop runs as long as the caller wants. A device that is
/// not disposed keeps its driver context until the process exits, and the driver allows only a limited number of them per
/// process (63 live devices on the NVIDIA 595 driver); <see cref="LeakedDevicesHitTheDriverLimit"/> measures that limit.
/// </summary>
public sealed class DeviceLifetimeTests
{
    private const int Iterations = 128;

    private static DeviceInfo DeviceUnderTest(VulkanProvider provider)
    {
        var devices = provider.GetDevices();
        return TestSwitches.DeviceIndex is { } index && index >= 0 && index < devices.Count ? devices[index]
            : devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0];
    }

    private static (Sharplight.Graph.ComputeGraph Graph, Sharplight.Graph.Value Input) SmallGraph()
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, 64);
        var w = tg.WF("w", Rnd.F(64 * 32, 7, 1f), 32, 64);
        tg.B.Output("y", tg.B.Unary("Relu", tg.B.MatMul(x, w)));
        return (tg.B.Build(), x);
    }

    private static CompiledModel Compile(Sharplight.Graph.ComputeGraph graph, IComputeDevice device)
        => CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = 4, ContextLength = 4 },
            ReservePerDevice = 0,
        });

    private static float[] Run(CompiledModel model, Sharplight.Graph.Value x, float[] input)
    {
        model.SetInput(x, MemoryMarshal.AsBytes(input.AsSpan()));
        var parameters = new RunParameters();
        parameters.Symbols["T"] = 4;
        model.Run(parameters);
        var y = new float[4 * 32];
        model.GetOutput<float>("y", y);
        return y;
    }

    [Fact]
    public void OpenRunAndDisposeManyDevices()
    {
        Gpu.RequireVulkan();
        var provider = new VulkanProvider();
        var info = DeviceUnderTest(provider);
        var (graph, x) = SmallGraph();
        var input = Rnd.F(4 * 64, 3, 1f);
        float[]? first = null;
        long usedBefore = Gpu.Device!.GetMemoryInfo().Usage;
        for (int i = 0; i < Iterations; i++)
        {
            using var device = provider.OpenDevice(info);
            using var model = Compile(graph, device);
            var y = Run(model, x, input);
            first ??= y;
            Assert.Equal(first, y);
        }

        // The driver's view of the process's device-local memory (VK_EXT_memory_budget) is back where it started, give or take
        // allocator slack: nothing of the opened devices stayed alive.
        long growth = Gpu.Device!.GetMemoryInfo().Usage - usedBefore;
        Assert.True(growth < 64L << 20, $"device-local usage grew by {growth >> 20} MiB over {Iterations} open/dispose cycles");
    }

    /// <summary>
    /// A device disposed while a model compiled on it is alive (out of order): using the model throws instead of writing to the
    /// unmapped staging memory, disposing it afterwards is harmless, and the disposed device refuses new compilations.
    /// </summary>
    [Fact]
    public void ModelOfADisposedDeviceThrowsInsteadOfCrashing()
    {
        Gpu.RequireVulkan();
        var provider = new VulkanProvider();
        var (graph, x) = SmallGraph();
        var input = Rnd.F(4 * 64, 3, 1f);
        var device = provider.OpenDevice(DeviceUnderTest(provider));
        var model = Compile(graph, device);
        Run(model, x, input);
        device.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.SetInput(x, MemoryMarshal.AsBytes(input.AsSpan())));
        var parameters = new RunParameters();
        parameters.Symbols["T"] = 4;
        Assert.Throws<ObjectDisposedException>(() => model.Run(parameters));
        Assert.Throws<ObjectDisposedException>(() => model.GetOutput<float>("y", new float[4 * 32]));
        model.Dispose();
        Assert.ThrowsAny<ObjectDisposedException>(() => Compile(graph, device));
    }

    /// <summary>
    /// Diagnostic (<c>SHARPLIGHT_TEST_VK_DEVICE_LIMIT=1</c>): opens devices without disposing them until the driver refuses, and reports
    /// how many it allowed. It leaves the process with no free driver contexts, so it does not run with the rest of the suite.
    /// </summary>
    [Fact]
    public void LeakedDevicesHitTheDriverLimit()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE_LIMIT") == "1", "diagnostic; set SHARPLIGHT_TEST_VK_DEVICE_LIMIT=1");
        Gpu.RequireVulkan();
        var provider = new VulkanProvider();
        var info = DeviceUnderTest(provider);
        var open = new List<IComputeDevice>();
        string failure = "none";
        try
        {
            for (int i = 0; i < 1000; i++)
            {
                open.Add(provider.OpenDevice(info));
            }
        }
        catch (Exception e)
        {
            failure = e.GetType().Name + ": " + e.Message;
        }
        finally
        {
            foreach (var device in open)
            {
                device.Dispose();
            }
        }

        TestContext.Current.TestOutputHelper?.WriteLine($"opened {open.Count} devices (plus the shared one) before: {failure}");
        Console.WriteLine($"[device-limit] opened {open.Count} devices (plus the shared one) before: {failure}");
        using var again = provider.OpenDevice(info);
    }
}
