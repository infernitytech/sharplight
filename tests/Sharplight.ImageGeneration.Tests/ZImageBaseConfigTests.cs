using Sharplight.Import.Diffusers;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Z-Image (undistilled, 2026-11-18 hot-model sweep) is told apart from Z-Image-Turbo by its file name or its scheduler's shift and
/// then runs with classifier-free guidance, 50 steps, guidance 4 and shift 6. Before, every Z-Image checkpoint ran as the Turbo.
/// </summary>
public sealed class ZImageBaseConfigTests
{
    private static ZImageConfig Config() => new()
    {
        HiddenSize = 3840,
        HeadCount = 30,
        Layers = 30,
        RefinerLayers = 2,
        CaptionDim = 2560,
        InChannels = 64,
        ModulationDim = 256,
    };

    private static Dictionary<string, object> Write(ZImageConfig config)
    {
        var metadata = new Dictionary<string, object>();
        config.WriteTo(metadata);
        return metadata;
    }

    [Theory]
    [InlineData("z-image-Q8_0.gguf", true)]
    [InlineData("z_image_bf16.safetensors", true)]
    [InlineData("zimage-Q4_K_M.gguf", false)]
    [InlineData("zimage.safetensors", false)]
    [InlineData("z-image-turbo-Q8_0.gguf", false)]
    [InlineData("z_image_turbo_bf16.safetensors", false)]
    [InlineData("zImageTurbo_v1.safetensors", false)]
    [InlineData("my-finetune.safetensors", false)]
    public void FileNamesTellBaseFromTurbo(string name, bool isBase) => Assert.Equal(isBase, ZImageConfig.IsBaseName(name));

    [Fact]
    public void BaseFilesGuideAndTurboFilesDoNot()
    {
        var baseModel = Write(Config().WithFileName("z-image-Q8_0.gguf"));
        Assert.Equal(true, baseModel["zimage.classifier_free_guidance"]);
        Assert.Equal(50, baseModel["zimage.default_steps"]);
        Assert.Equal(5.0, baseModel["zimage.default_guidance"]);
        Assert.Equal(6f, baseModel["zimage.scheduler.shift"]);
        Assert.Equal(true, baseModel["zimage.transformer.f32_activations"]);

        var turbo = Write(Config().WithFileName("z-image-turbo-Q8_0.gguf"));
        Assert.Equal(false, turbo["zimage.classifier_free_guidance"]);
        Assert.Equal(9, turbo["zimage.default_steps"]);
        Assert.Equal(3f, turbo["zimage.scheduler.shift"]);
        Assert.Equal(false, turbo["zimage.transformer.f32_activations"]);
    }

    [Theory]
    [InlineData(6.0, false)]
    [InlineData(3.0, true)]
    public void FolderSchedulerShiftTellsBaseFromTurbo(double shift, bool distilled)
    {
        string scheduler = $$"""{ "_class_name": "FlowMatchEulerDiscreteScheduler", "shift": {{shift.ToString("F1", System.Globalization.CultureInfo.InvariantCulture)}} }""";
        var config = Config().WithDiffusers(scheduler, null);
        Assert.Equal(distilled, config.Distilled);
        Assert.Equal(!distilled, Write(config)["zimage.classifier_free_guidance"]);
    }
}
