using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// ControlNets loaded as the <c>controlnet</c> auxiliary source of a UNet model: lllyasviel/control_v11p_sd15_canny with SD 1.5
/// (<c>~/models/sd/controlnet/sd15-canny</c>) and diffusers/controlnet-canny-sdxl-1.0 (fp16 files) with SDXL base
/// (<c>~/models/sd/controlnet/sdxl-canny</c>), against diffusers' StableDiffusionControlNetPipeline / StableDiffusionXLControlNetPipeline
/// (tools/make-controlnet-references.py: float32 on CUDA, Euler, shared initial noise; references in
/// <c>~/models/sd/controlnet-reference/{sd15-canny,sdxl-canny}</c>): one ControlNet + UNet evaluation (two rows with different scales) on the
/// CPU, Vulkan and CUDA (with <c>SHARPLIGHT_TEST_CUDA=1</c>), the whole image on the GPUs, and the <c>.slm</c> round trip. Skips without the files.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class ControlNetTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    public static TheoryData<string, string> GraphCases()
    {
        var data = new TheoryData<string, string>();
        foreach (string family in new[] { "sd15-canny", "sdxl-canny" })
        {
            foreach (string device in new[] { "cpu", "vulkan", "cuda" })
            {
                data.Add(family, device);
            }
        }

        return data;
    }

    public static TheoryData<string, string> ImageCases()
    {
        var data = new TheoryData<string, string>();
        foreach (string family in new[] { "sd15-canny", "sdxl-canny" })
        {
            foreach (string device in new[] { "vulkan", "cuda" })
            {
                data.Add(family, device);
            }
        }

        return data;
    }

    private static string Pipeline(string family) => family.StartsWith("sdxl", StringComparison.Ordinal) ? Env.SdxlDiffusers : Env.Sd15Diffusers;

    private static string ControlNet(string family) => Path.Combine(Home, "models/sd/controlnet", family);

    private static string Reference(string family) => Path.Combine(Home, "models/sd/controlnet-reference", family);

    private static bool Present(string family)
        => Directory.Exists(Pipeline(family)) && Directory.Exists(ControlNet(family)) && File.Exists(Path.Combine(Reference(family), "index.json"));

    private static JsonElement Index(string family)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference(family), "index.json")));
        return document.RootElement.Clone();
    }

    private static float[] Tensor(string family, string name) => Env.ReadFloats(Path.Combine(Reference(family), name + ".f32"));

    private static TaskModel LoadWith(string family, DeviceSlot slot, string? controlNet = null)
        => Env.Load(Pipeline(family), slot, new Dictionary<string, string> { [AuxiliaryRoles.ControlNet] = controlNet ?? ControlNet(family) });

    [Theory]
    [MemberData(nameof(GraphCases))]
    public void UNetWithControlNetMatchesDiffusers(string family, string device)
    {
        var slot = Present(family) ? Env.Device(device) : null;
        Assert.SkipWhen(slot is null, $"{family}, its pipeline, its references or the {device} device missing (CUDA runs with SHARPLIGHT_TEST_CUDA=1)");
        var index = Index(family);
        using var model = LoadWith(family, slot!);
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = index.GetProperty("height").GetInt32(), [DiffusionGraphs.Width] = index.GetProperty("width").GetInt32() };
        var inputs = new List<(string, Array)>
        {
            (DiffusionGraphs.Latent, Tensor(family, "unet_latent")), (DiffusionGraphs.Timestep, Tensor(family, "unet_timestep")),
            (DiffusionGraphs.Context, Tensor(family, "unet_context")),
        };
        if (family.StartsWith("sdxl", StringComparison.Ordinal))
        {
            inputs.Add((DiffusionGraphs.Pooled, Tensor(family, "unet_pooled")));
            inputs.Add((DiffusionGraphs.SizeConditions, Tensor(family, "unet_time_ids")));
        }

        var controlled = GraphRunner.Run(model, DiffusionGraphs.ControlNetRole, DiffusionGraphs.DenoiseControl, DiffusionGraphs.Prediction, 2, dims,
            [.. inputs, (DiffusionGraphs.ControlImage, Tensor(family, "unet_control")), (DiffusionGraphs.ControlScale, Tensor(family, "unet_scale"))]);
        var expected = Tensor(family, "unet_out");
        var (max, rms) = Env.Compare(controlled, expected);
        var (_, fromPlain) = Env.Compare(Tensor(family, "unet_plain_out"), expected);
        TestContext.Current.SendDiagnosticMessage($"{family} {device}: ControlNet + UNet max {max:E2}, relative RMS {rms:E2} (the UNet alone is {fromPlain:E2} away)");
        Assert.True(rms < (device == "cpu" ? 1e-4 : 5e-3), $"{family} {device}: max {max}, relative RMS {rms}");
        Assert.True(fromPlain > 10 * rms, $"{family}: the ControlNet barely changes the prediction ({fromPlain:E2})");
    }

    [Theory]
    [MemberData(nameof(ImageCases))]
    public async Task ImageMatchesDiffusers(string family, string device)
    {
        var slot = Present(family) ? Env.Device(device) : null;
        Assert.SkipWhen(slot is null, $"{family}, its pipeline, its references or the {device} device missing (CUDA runs with SHARPLIGHT_TEST_CUDA=1)");
        var index = Index(family);
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
        using var model = LoadWith(family, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        Assert.True(generator.Defaults.ControlNet);
        var result = await generator.GenerateAsync(new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative_prompt").GetString(),
            ControlImage = PngCodec.Read(Path.Combine(Reference(family), "control.png")),
        }, new ImageGenerationOptions
        {
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            ControlScale = index.GetProperty("scale").GetSingle(),
            Scheduler = Samplers.Euler,
            InitialNoise = Tensor(family, "noise"),
        }, TestContext.Current.CancellationToken);
        Assert.Equal(width, result.Images[0].Width);
        Assert.Equal(height, result.Images[0].Height);
        var expected = PlanarImage.FromPlanar(Tensor(family, "image"), width, height, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"{family}-{device}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"{family}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{family} on {device}: PSNR {psnr:F1} dB");
        Assert.True(psnr > 30, $"{family} on {device}: PSNR {psnr:F1} dB");
    }

    /// <summary>
    /// lllyasviel's original release file (<c>~/models/sd/controlnet/original/control_v11p_sd15_canny.pth</c>, LDM names under
    /// <c>control_model.</c>, read with the restricted unpickler) maps to exactly the diffusers folder's canonical names, shapes, configuration
    /// and weights.
    /// </summary>
    [Fact]
    public void OriginalFileMatchesTheDiffusersFolder()
    {
        string original = Path.Combine(Home, "models/sd/controlnet/original/control_v11p_sd15_canny.pth");
        Assert.SkipUnless(File.Exists(original) && Directory.Exists(ControlNet("sd15-canny")), "the original ControlNet file or the diffusers folder missing");
        var importer = new Sharplight.Import.Diffusers.DiffusersImporter();
        Assert.True(importer.CanOpen(original));
        var fromOriginal = importer.Open(original);
        var fromFolder = importer.Open(ControlNet("sd15-canny"));
        try
        {
            Assert.Equal(fromFolder.Weights.Keys.Order(StringComparer.Ordinal), fromOriginal.Weights.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, weight) in fromFolder.Weights)
            {
                Assert.Equal(weight.Shape, fromOriginal.Weights[key].Shape);
                Assert.Equal(weight.ReadAll(), fromOriginal.Weights[key].ReadAll());
            }

            foreach (var (key, value) in fromFolder.Metadata.Where(pair => pair.Key.StartsWith("controlnet.", StringComparison.Ordinal)
                && !pair.Key.EndsWith(".sample_size", StringComparison.Ordinal)))
            {
                Assert.True(fromOriginal.Metadata.TryGetValue(key, out var other), key);
                Assert.Equal(JsonSerializer.Serialize(value), JsonSerializer.Serialize(other));
            }
        }
        finally
        {
            foreach (var resource in fromOriginal.OwnedResources.Concat(fromFolder.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }

    [Fact]
    public void ControlNetOfAnotherFamilyFailsNamingIt()
    {
        Assert.SkipUnless(Present("sd15-canny") && Directory.Exists(ControlNet("sdxl-canny")), "SD 1.5 or the ControlNets missing");
        var error = Assert.Throws<NotSupportedException>(() => LoadWith("sd15-canny", Env.Cpu(), ControlNet("sdxl-canny")));
        Assert.Contains("SDXL ControlNets", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PackageKeepsTheControlNet()
    {
        Assert.SkipUnless(Present("sd15-canny"), "SD 1.5 or its ControlNet missing");
        _ = Env.Services;
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "imagegen-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"sd15-canny-{Guid.NewGuid():N}.slm");
        try
        {
            TestHost.Converter.Convert(Env.Sd15Diffusers, path, auxiliary: new Dictionary<string, string> { [AuxiliaryRoles.ControlNet] = ControlNet("sd15-canny") });
            using var model = Env.Load(path, Env.Cpu());
            Assert.Equal([DiffusionGraphs.DenoiseControl], model.Description.Component(DiffusionGraphs.ControlNetRole).Graphs);
            using var generator = Env.Clients.Create<IImageGenerator>(model);
            // Another size rebuilds the ControlNet's graph from the package's metadata and weights; the control image sets the size.
            var control = new ImageInput(64, 128, new byte[64 * 128 * 3]);
            var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a bird", ControlImage = control },
                new ImageGenerationOptions { Steps = 1, Seed = 3, ControlScale = 0.5f }, TestContext.Current.CancellationToken);
            Assert.Equal(64, result.Images[0].Width);
            Assert.Equal(128, result.Images[0].Height);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
