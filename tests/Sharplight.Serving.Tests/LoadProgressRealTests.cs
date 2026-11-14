using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Serving.Tests;

/// <summary>Load progress of a real small model (skipped when the file is absent).</summary>
[Collection("RealModel")]
public class LoadProgressRealTests
{
    private static readonly string ModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");

    private static IComputeDevice OpenCpu()
    {
        var cpu = new CpuExecutionProvider();
        return cpu.OpenDevice(cpu.GetDevices()[0]);
    }

    private static IComputeDevice? OpenGpu()
    {
        var vk = new VulkanProvider();
        if (!vk.CheckAvailability().IsAvailable || vk.GetDevices() is not [var first, ..])
        {
            return null;
        }

        return vk.OpenDevice(first);
    }

    private static void LoadAndCheck(IComputeDevice device, bool expectUpload)
    {
        GgufImporter.Register();
        var log = new ProgressLog();
        using var model = LlmModel.Load(ModelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            ContextLength = 256,
            MaxBatchTokens = 32,
            Progress = log,
        });
        Assert.Equal([LoadStage.Reading, LoadStage.Building], log.Stages);
        using var session = model.CreateSession();

        var stages = log.Stages;
        LoadStage[] expected = expectUpload
            ? [LoadStage.Reading, LoadStage.Building, LoadStage.Placing, LoadStage.Compiling, LoadStage.Uploading, LoadStage.Ready]
            : [LoadStage.Reading, LoadStage.Building, LoadStage.Placing, LoadStage.Compiling, LoadStage.Ready];
        Assert.Equal(expected, stages);
        Assert.Equal(LoadStage.Ready, log.Reports[^1].Stage);
        foreach (var stage in stages)
        {
            var last = log.Last(stage);
            Assert.True(last.IsComplete, $"{stage}: {last.Completed} / {last.Total}");
            Assert.True(last.Total > 0, $"{stage} has no work");
        }

        var compiling = log.Last(LoadStage.Compiling);
        Assert.InRange(compiling.Cached, 0, compiling.Total);
        if (expectUpload)
        {
            Assert.True(log.Last(LoadStage.Uploading).Total > 100_000_000, "the weights are several hundred MB");
        }
    }

    [Fact]
    public void Qwen3_Cpu_ReportsStagesInOrder()
    {
        Assert.SkipUnless(File.Exists(ModelPath), "model not found: " + ModelPath);
        using var device = OpenCpu();
        LoadAndCheck(device, expectUpload: false);
    }

    [Fact]
    public void Qwen3_Vulkan_ReportsCompileAndUpload()
    {
        Assert.SkipUnless(File.Exists(ModelPath), "model not found: " + ModelPath);
        using var device = OpenGpu();
        Assert.SkipUnless(device is not null, "no Vulkan device");
        LoadAndCheck(device, expectUpload: true);
    }

    [Fact]
    public void CancelledLoadThrows()
    {
        Assert.SkipUnless(File.Exists(ModelPath), "model not found: " + ModelPath);
        GgufImporter.Register();
        using var device = OpenCpu();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => LlmModel.Load(ModelPath, new LlmLoadOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            CancellationToken = cts.Token,
        }));
    }
}
