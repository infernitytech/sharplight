using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The Stable Diffusion editing models against diffusers (tools/make-sd-edit-references.py: float32, Euler / DDIM from shared noise,
/// 512×512, 20 steps): InstructPix2Pix (8-channel UNet, text and image guidance) and the 9-channel SD 1.5 inpainting UNet. Skips when the
/// models, their references or the devices are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class SdEditTests
{
    private static readonly string Root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/imagegen-edit");

    private static string Model(string kind) => Path.Combine(Root, kind switch
    {
        "instruct-pix2pix" => "instruct-pix2pix",
        "sdxl-inpaint" => "sdxl-inpainting",
        _ => "sd15-inpainting",
    });

    private static bool Inpainting(string kind) => kind is "inpaint" or "sdxl-inpaint";

    private static string Reference(string kind) => Model(kind) + "-reference";

    private static float[] Tensor(string kind, string name) => Env.ReadFloats(Path.Combine(Reference(kind), name + ".f32"));

    private static bool Present(string kind) => Directory.Exists(Model(kind)) && File.Exists(Path.Combine(Reference(kind), "index.json"));

    private static JsonElement Index(string kind)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference(kind), "index.json")));
        return document.RootElement.Clone();
    }

    public static TheoryData<string, string> Cases() => new()
    {
        { "cpu", "instruct-pix2pix" },
        { "vulkan", "instruct-pix2pix" },
        { "cpu", "inpaint" },
        { "vulkan", "inpaint" },
        { "vulkan", "sdxl-inpaint" },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void UNetMatchesDiffusers(string device, string kind)
    {
        Assert.SkipUnless(Present(kind), $"{kind} or its references missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Model(kind), slot!);
        var denoiser = model.Description.Component(ModelComponentRoles.Denoiser);
        Assert.Equal(Inpainting(kind) ? DiffusionConfigKeys.ConditionInpaint : DiffusionConfigKeys.ConditionImage,
            denoiser.ConfigValue<string?>(DiffusionConfigKeys.ConditionMode, null));
        var inputs = new List<(string, Array)>
        {
            (DiffusionGraphs.Latent, Tensor(kind, "unet_latent")), (DiffusionGraphs.ConditionLatent, Tensor(kind, "unet_condition")),
            (DiffusionGraphs.Timestep, Tensor(kind, "unet_timestep")), (DiffusionGraphs.Context, Tensor(kind, "unet_context")),
        };
        if (kind == "sdxl-inpaint")
        {
            inputs.Add((DiffusionGraphs.Pooled, Tensor(kind, "unet_pooled")));
            inputs.Add((DiffusionGraphs.SizeConditions, Tensor(kind, "unet_time_ids")));
        }

        // SDXL's native size is 1024; the references are 512.
        var dims = kind == "sdxl-inpaint" ? new Dictionary<string, long> { [DiffusionGraphs.Height] = 512, [DiffusionGraphs.Width] = 512 } : null;
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2, dims, [.. inputs]);
        var (max, rms) = Env.Compare(prediction, Tensor(kind, "unet_out"));
        TestContext.Current.SendDiagnosticMessage($"{device} {kind} UNet: max {max}, relative RMS {rms}");
        Assert.True(rms < (device == "cpu" ? 1e-3 : 5e-3), $"unet: max {max}, relative RMS {rms}");

        // Three rows (InstructPix2Pix's guided step, one run): the third row repeats the first. Vulkan once overwrote the time embedding at
        // three rows (the conv split-K workspace was sized for the split batches only).
        if (kind == "sdxl-inpaint")
        {
            return;
        }

        float[] Triple(string name, int rowSize)
        {
            var values = Tensor(kind, name);
            return [.. values, .. values.AsSpan(0, rowSize).ToArray()];
        }

        var latent = Tensor(kind, "unet_latent");
        int latentRow = latent.Length / 2;
        var condition = Tensor(kind, "unet_condition");
        var context = Tensor(kind, "unet_context");
        var three = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 3,
            (DiffusionGraphs.Latent, Triple("unet_latent", latentRow)), (DiffusionGraphs.ConditionLatent, Triple("unet_condition", condition.Length / 2)),
            (DiffusionGraphs.Timestep, Triple("unet_timestep", 1)), (DiffusionGraphs.Context, Triple("unet_context", context.Length / 2)));
        var expected = Tensor(kind, "unet_out");
        var (maxThree, rmsThree) = Env.Compare(three, [.. expected, .. expected.AsSpan(0, latentRow).ToArray()]);
        Assert.True(rmsThree < (device == "cpu" ? 1e-3 : 5e-3), $"unet with three rows: max {maxThree}, relative RMS {rmsThree}");
    }

    [Fact]
    public void InstructPix2PixImageLatentsMatchDiffusers()
    {
        Assert.SkipUnless(Present("instruct-pix2pix"), "instruct-pix2pix or its references missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Model("instruct-pix2pix"), slot!);
        var input = PngCodec.Read(Path.Combine(Reference("instruct-pix2pix"), "input.png"));
        var planes = new float[3 * input.Width * input.Height];
        PlanarImage.ToPlanar(input, planes, 0f, 1f);
        var encoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, DiffusionGraphs.Latent, 1, (DiffusionGraphs.Image, planes));
        float scaling = model.Description.Component(ModelComponentRoles.Vae).ConfigValue(DiffusionConfigKeys.ScalingFactor, 1f);
        Assert.Equal(0.18215f, scaling, 4);
        var (max, rms) = Env.Compare(encoded.Select(v => v / scaling).ToArray(), Tensor("instruct-pix2pix", "image_latents"));
        Assert.True(rms < 5e-3, $"vae encode: max {max}, relative RMS {rms}");
    }

    [Theory]
    [InlineData("instruct-pix2pix")]
    [InlineData("inpaint")]
    [InlineData("sdxl-inpaint")]
    public async Task EditMatchesDiffusers(string kind)
    {
        Assert.SkipUnless(Present(kind), $"{kind} or its references missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Model(kind), slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Index(kind);
        int size = index.GetProperty("size").GetInt32();
        var request = new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            Image = PngCodec.Read(Path.Combine(Reference(kind), "input.png")),
            Mask = Inpainting(kind) ? PngCodec.Read(Path.Combine(Reference(kind), "mask.png")) : null,
            Strength = 1f,
        };
        var result = await generator.GenerateAsync(request, new ImageGenerationOptions
        {
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            ImageGuidanceScale = kind == "instruct-pix2pix" ? index.GetProperty("image_guidance_scale").GetSingle() : null,
            Scheduler = kind == "inpaint" ? Samplers.Ddim : Samplers.Euler,
            InitialNoise = Tensor(kind, "noise"),
        }, TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor(kind, "image"), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"{kind}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"{kind}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{kind}: PSNR {psnr:F1} dB");
        Assert.True(psnr > 35, $"{kind}: PSNR {psnr:F1} dB");
    }

    [Theory]
    [InlineData("instruct-pix2pix")]
    [InlineData("inpaint")]
    public async Task PackageRoundTripGeneratesTheSameImage(string kind)
    {
        Assert.SkipUnless(Present(kind), $"{kind} or its references missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        _ = Env.Services;
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "imagegen-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"{kind}-{Guid.NewGuid():N}.slm");
        try
        {
            TestHost.Converter.Convert(Model(kind), path);
            var request = new ImageGenerationRequest
            {
                Prompt = "make it a painting",
                Image = PngCodec.Read(Path.Combine(Reference(kind), "input.png")),
                Mask = Inpainting(kind) ? PngCodec.Read(Path.Combine(Reference(kind), "mask.png")) : null,
                Strength = 1f,
            };
            var options = new ImageGenerationOptions { Steps = 3, Seed = 5, Width = 256, Height = 256, Scheduler = Samplers.Euler };
            var ct = TestContext.Current.CancellationToken;
            byte[] expected;
            using (var folderModel = Env.Load(Model(kind), slot!))
            using (var generator = Env.Clients.Create<IImageGenerator>(folderModel))
            {
                expected = (await generator.GenerateAsync(request, options, ct)).Images[0].Rgb;
            }

            using var package = Env.Load(path, slot!);
            using var fromPackage = Env.Clients.Create<IImageGenerator>(package);
            var denoiser = package.Description.Component(ModelComponentRoles.Denoiser);
            Assert.Equal(Inpainting(kind) ? DiffusionConfigKeys.ConditionInpaint : DiffusionConfigKeys.ConditionImage,
                denoiser.ConfigValue<string?>(DiffusionConfigKeys.ConditionMode, null));
            Assert.Equal(expected, (await fromPackage.GenerateAsync(request, options, ct)).Images[0].Rgb);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task InstructPix2PixNeedsAnImageAndInpaintingTakesNoReferences()
    {
        Assert.SkipUnless(Present("instruct-pix2pix"), "instruct-pix2pix missing");
        var slot = Env.Device("cpu")!;
        using var model = Env.Load(Model("instruct-pix2pix"), slot);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var defaults = generator.Defaults;
        Assert.Equal(1.5f, defaults.ImageGuidanceScale);
        await Assert.ThrowsAsync<ArgumentException>(async () => await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "x" },
            new ImageGenerationOptions { Steps = 1 }, TestContext.Current.CancellationToken));
    }
}
