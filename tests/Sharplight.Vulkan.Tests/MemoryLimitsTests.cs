using System.Runtime.InteropServices;
using Sharplight.Runtime;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Hardware with less memory than the test GPU, simulated with <see cref="DeviceMemoryLimits"/>: no or a tiny BAR window (staging
/// falls back to host memory and shrinks), a small largest allocation (big buffers are bound to several allocations), a small device
/// (a clear <see cref="DeviceOutOfMemoryException"/>). Every case must compute what the unconstrained device computes. Runs on any
/// Vulkan device; on llvmpipe (<c>SHARPLIGHT_TEST_DEVICE</c>) the memory is unified, the case without a separate BAR window.
/// </summary>
public sealed unsafe class MemoryLimitsTests
{
    private const int K = 512;
    private const int N = 2048;   // a 4 MiB f32 weight
    private const int Rows = 4;

    private static DeviceInfo DeviceUnderTest(VulkanProvider provider)
    {
        var devices = provider.GetDevices();
        return TestSwitches.DeviceIndex is { } index && index >= 0 && index < devices.Count ? devices[index]
            : devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0];
    }

    private static VulkanDevice Open(Action<DeviceMemoryLimits> limit)
    {
        var provider = new VulkanProvider();
        var options = new DeviceOptions { UseCompilationCache = false };
        limit(options.MemoryLimits);
        return (VulkanDevice)provider.OpenDevice(DeviceUnderTest(provider), options);
    }

    private static (Sharplight.Graph.ComputeGraph Graph, Sharplight.Graph.Value Input) Graph()
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var w = tg.WF("w", Rnd.F(N * K, 11, 0.05f), N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        return (tg.B.Build(), x);
    }

    private static float[] Run(IComputeDevice device)
    {
        var (graph, x) = Graph();
        using var model = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = Rows, ContextLength = Rows },
            ReservePerDevice = 0,
        });
        var input = Rnd.F(Rows * K, 5, 1f);
        model.SetInput(x, MemoryMarshal.AsBytes(input.AsSpan()));
        var parameters = new RunParameters();
        parameters.Symbols["T"] = Rows;
        model.Run(parameters);
        var y = new float[Rows * N];
        model.GetOutput<float>("y", y);
        return y;
    }

    private static float[] Reference()
    {
        Gpu.RequireVulkan();
        return Run(Gpu.Device!);
    }

    [Fact]
    public void WithoutBarWindowStagingUsesHostMemory()
    {
        var expected = Reference();
        using var device = Open(limits => limits.HostVisibleDeviceMemory = 0);
        Assert.Equal(expected, Run(device));

        var staging = device.Memory.Allocate(1 << 20, MemoryUsage.Upload, "test staging");
        try
        {
            // On a discrete GPU the staging must come from host memory; unified devices have only device-local memory.
            bool discrete = device.Info.Kind == DeviceKind.DiscreteGpu;
            Assert.True(!discrete || staging.Block!.Tracked == 0, "upload staging was placed in device-local memory");
        }
        finally
        {
            device.Memory.Free(staging);
        }
    }

    [Fact]
    public void TinyBarWindowShrinksStaging()
    {
        var expected = Reference();
        using var device = Open(limits => limits.HostVisibleDeviceMemory = 3L << 20);
        Assert.Equal(expected, Run(device));
    }

    [Fact]
    public void StagingShrinksToTheLargestAllocation()
    {
        var expected = Reference();
        using var device = Open(limits => limits.MaxAllocation = 1L << 20);
        Assert.Equal(1UL << 20, device.Memory.MaxAllocation);
        using (var uploader = new WeightUploader(device, device.CommandPool, 64L << 20, 1))
        {
            var target = device.Memory.Allocate(4 << 20, MemoryUsage.Device, "test target");
            try
            {
                var buffer = new Sharplight.Shaders.Lowering.BufferDecl
                {
                    Id = 0,
                    Kind = Sharplight.Shaders.Lowering.BufferKind.Weight,
                    Name = "w",
                    Bytes = 4 << 20,
                    InitialData = new byte[4 << 20],
                };
                uploader.Upload(buffer, target);
                uploader.Complete();
                if (target.Mapped == null)
                {
                    Assert.True(uploader.StagingSize <= 1UL << 20, $"staging of {uploader.StagingSize} bytes exceeds the 1 MiB allocation cap");
                }
            }
            finally
            {
                device.Memory.Free(target);
            }
        }

        // The 4 MiB weight is larger than one allocation: bound to several (sparse binding), or a clear error without it.
        if (device.SparseBinding)
        {
            Assert.Equal(expected, Run(device));
        }
        else
        {
            var error = Assert.ThrowsAny<OutOfMemoryException>(() => Run(device));
            Assert.Contains("largest allocation", error.Message);
        }
    }

    [Fact]
    public void SparseBufferSpansSeveralAllocations()
    {
        Gpu.RequireVulkan();
        using var device = Open(limits => limits.MaxAllocation = 1L << 20);
        Assert.SkipUnless(device.SparseBinding, "the device has no sparse binding");
        var buffer = device.Memory.Allocate(5UL << 20, MemoryUsage.Device, "test sparse");
        try
        {
            Assert.Equal(5, buffer.Block!.Memory.Count);
            Assert.Equal(5UL << 20, buffer.Block.Memory.Aggregate(0UL, (sum, chunk) => sum + chunk.Size));
        }
        finally
        {
            device.Memory.Free(buffer);
        }
    }

    [Fact]
    public void SmallDeviceFailsWithAClearError()
    {
        Gpu.RequireVulkan();
        using var device = Open(limits => limits.DeviceMemory = 2L << 20);
        Assert.Equal(2L << 20, device.GetMemoryInfo().Total);
        var error = Assert.Throws<DeviceOutOfMemoryException>(() => device.Memory.Allocate(4UL << 20, MemoryUsage.Device, "weight buffer w"));
        Assert.Contains("weight buffer w", error.Message);
        Assert.Contains("Heaps:", error.Message);
        Assert.Contains("device memory 2 MiB", error.Message);
        Assert.Equal(4L << 20, error.RequestedBytes);
    }

    [Fact]
    public void SmallDeviceModelDoesNotFit()
    {
        Gpu.RequireVulkan();
        using var device = Open(limits => limits.DeviceMemory = 2L << 20);
        // Placement sees the small device and refuses up front, or the allocation fails clearly; never a crash or a device loss.
        Assert.ThrowsAny<Exception>(() => Run(device));
        Assert.True(device.GetMemoryInfo().UsedByUs == 0, "a failed compilation left memory allocated");
    }
}
