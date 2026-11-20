using System.Runtime.InteropServices;
using Sharplight.Runtime;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// The CUDA provider under <see cref="DeviceMemoryLimits"/>: without page-locked memory (pageable staging) or with little of it
/// (smaller staging buffers) a model computes what it computes unconstrained; a small device fails with a clear
/// <see cref="DeviceOutOfMemoryException"/> and reports the capped total.
/// </summary>
public sealed class CudaMemoryLimitsTests
{
    private const int K = 512;
    private const int N = 2048;   // a 4 MiB f32 weight
    private const int Rows = 4;

    private static CudaDevice? Open(Action<DeviceMemoryLimits> limit)
    {
        var provider = new CudaProvider();
        if (!provider.CheckAvailability().IsAvailable)
        {
            return null;
        }

        var options = new DeviceOptions { UseCompilationCache = false };
        limit(options.MemoryLimits);
        return (CudaDevice)provider.OpenDevice(provider.GetDevices()[0], options);
    }

    private static float[] Run(IComputeDevice device)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var w = tg.WF("w", Rnd.F(N * K, 11, 0.05f), N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        using var model = CompiledModel.Compile(tg.B.Build(), new PlacementOptions
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

    [Fact]
    public void PageableAndSmallPinnedStagingGiveTheSameResult()
    {
        using var reference = Open(_ => { });
        Assert.SkipUnless(reference is not null, "No CUDA device.");
        var expected = Run(reference);

        using var pageable = Open(limits => limits.PinnedHostMemory = 0)!;
        using (var uploader = new CudaWeightUploader(pageable, 64L << 20, 1))
        {
            Assert.False(uploader.Pinned);
        }

        Assert.Equal(expected, Run(pageable));

        using var small = Open(limits => limits.PinnedHostMemory = 3L << 20)!;
        using (var uploader = new CudaWeightUploader(small, 64L << 20, 1))
        {
            Assert.True(uploader.Pinned);
            Assert.True(uploader.StagingSize <= 2L << 20, $"staging of {uploader.StagingSize} bytes exceeds the pinned cap");
        }

        Assert.Equal(expected, Run(small));
    }

    [Fact]
    public void SmallDeviceFailsWithAClearError()
    {
        using var device = Open(limits => limits.DeviceMemory = 2L << 20);
        Assert.SkipUnless(device is not null, "No CUDA device.");
        Assert.Equal(2L << 20, device.GetMemoryInfo().Total);
        Assert.True(device.GetMemoryInfo().Available <= 2L << 20);
        var error = Assert.Throws<DeviceOutOfMemoryException>(() => device.Allocate(4L << 20, "weight buffer w"));
        Assert.Contains("weight buffer w", error.Message);
        Assert.Contains("device memory 2 MiB", error.Message);
        Assert.ThrowsAny<Exception>(() => Run(device));
    }
}
