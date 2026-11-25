using System.Text.Json;
using Sharplight.Models;
using Sharplight.Imaging;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The SDXL refiner (stabilityai/stable-diffusion-xl-refiner-1.0: OpenCLIP bigG only, five size conditions ending with the aesthetic score)
/// and the base + refiner ensemble (<see cref="ImageGenerationOptions.Refiner"/>) against diffusers' StableDiffusionXLPipeline with
/// <c>denoising_end</c> and StableDiffusionXLImg2ImgPipeline with <c>denoising_start</c> (tools/make-sdxl-refiner-references.py: f32 on
/// CUDA, Euler, the same initial noise). Skips without the folders: <c>~/models/sd/sdxl-diffusers</c>, <c>~/models/sd/sdxl-refiner-diffusers</c>
/// and the references in <c>~/models/sd/sdxl-refiner-reference</c> (<c>SHARPLIGHT_SDXL_BASE</c>, <c>SHARPLIGHT_SDXL_REFINER</c>,
/// <c>SHARPLIGHT_SDXL_REFINER_REFERENCE</c> override them).
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class SdxlRefinerTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    private static string Base => Environment.GetEnvironmentVariable("SHARPLIGHT_SDXL_BASE") ?? Env.SdxlDiffusers;

    private static string Refiner => Environment.GetEnvironmentVariable("SHARPLIGHT_SDXL_REFINER") ?? Path.Combine(Home, "models/sd/sdxl-refiner-diffusers");

    private static string Reference => Environment.GetEnvironmentVariable("SHARPLIGHT_SDXL_REFINER_REFERENCE") ?? Path.Combine(Home, "models/sd/sdxl-refiner-reference");

    private static bool RefinerPresent => Directory.Exists(Refiner) && File.Exists(Path.Combine(Reference, "index.json"));

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    [Fact]
    public void RefinerIsConditionedOnTheAestheticScore()
    {
        Assert.SkipUnless(RefinerPresent, "SDXL refiner or its references missing");
        using var model = Env.Load(Refiner, Env.Cpu());
        var denoise = model.ComponentGraph(ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise);
        Assert.Equal(5, denoise.Inputs.Single(i => i.Name == DiffusionGraphs.SizeConditions).Shape[1].Fixed);
        Assert.Equal(1280, denoise.Inputs.Single(i => i.Name == DiffusionGraphs.Context).Shape[2].Fixed);
        var text = model.ComponentGraph(ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode);
        Assert.True(text.Outputs.ContainsKey(DiffusionGraphs.Pooled));
        Assert.False(model.Description.Component(ModelComponentRoles.TextEncoder).Graphs.Contains(DiffusionGraphs.TextEncode2));
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var defaults = generator.Defaults;
        Assert.Equal(6f, defaults.AestheticScore);
        Assert.Equal(1024, defaults.Width);
    }

    [Fact]
    public void TextEncoderMatchesTransformers()
    {
        Assert.SkipUnless(RefinerPresent, "SDXL refiner or its references missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Refiner, slot!);
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        int[] tokens = PromptTokenizer.Tokenize(tokenizer, 77, Index().GetProperty("prompt").GetString()!, out _);
        var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Hidden, 0, (DiffusionGraphs.Tokens, tokens));
        var pooled = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Pooled, 0, (DiffusionGraphs.Tokens, tokens));
        var (maxHidden, rmsHidden) = Env.Compare(hidden, Tensor("text_hidden"));
        var (maxPooled, rmsPooled) = Env.Compare(pooled, Tensor("text_pooled"));
        TestContext.Current.SendDiagnosticMessage($"refiner text: hidden max {maxHidden} rms {rmsHidden}; pooled max {maxPooled} rms {rmsPooled}");
        Assert.True(rmsHidden < 1e-3, $"hidden: max {maxHidden}, relative RMS {rmsHidden}");
        Assert.True(rmsPooled < 1e-3, $"pooled: max {maxPooled}, relative RMS {rmsPooled}");
    }

    [Fact]
    public void UNetMatchesDiffusers()
    {
        Assert.SkipUnless(RefinerPresent, "SDXL refiner or its references missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Refiner, slot!);
        int size = Index().GetProperty("size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2, dims,
            (DiffusionGraphs.Latent, Tensor("unet_latent")), (DiffusionGraphs.Timestep, Tensor("unet_timestep")),
            (DiffusionGraphs.Context, Tensor("unet_context")), (DiffusionGraphs.Pooled, Tensor("unet_pooled")),
            (DiffusionGraphs.SizeConditions, Tensor("unet_time_ids")));
        var (max, rms) = Env.Compare(prediction, Tensor("unet_out"));
        TestContext.Current.SendDiagnosticMessage($"refiner UNet: max {max}, relative RMS {rms}");
        Assert.True(rms < 5e-3, $"unet: max {max}, relative RMS {rms}");
    }

    [Fact]
    public async Task BaseWithRefinerMatchesDiffusers()
    {
        Assert.SkipUnless(RefinerPresent && Directory.Exists(Base), "SDXL base, refiner or the references missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        var index = Index();
        int size = index.GetProperty("size").GetInt32();
        using var baseModel = Env.Load(Base, slot!);
        using var refinerModel = Env.Load(Refiner, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(baseModel);
        using var refiner = Env.Clients.Create<IImageGenerator>(refinerModel);
        int updates = 0;
        int total = 0;
        ImageGenerationResult? result = null;
        await foreach (var update in generator.GenerateStreamingAsync(new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative_prompt").GetString(),
        }, new ImageGenerationOptions
        {
            Width = size,
            Height = size,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            Scheduler = Samplers.Euler,
            InitialNoise = Tensor("noise"),
            Refiner = refiner,
            RefinerStart = index.GetProperty("switch").GetSingle(),
        }, TestContext.Current.CancellationToken))
        {
            updates++;
            total = update.TotalSteps;
            result = update.Result ?? result;
        }

        // 20 Euler steps cut at 0.8: 16 on the base, 4 on the refiner, plus the final update.
        Assert.Equal(index.GetProperty("steps").GetInt32(), total);
        Assert.Equal(total + 1, updates);
        var expected = PlanarImage.FromPlanar(Tensor("image"), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, "sdxl-refiner-sharplight.png"), PngCodec.Encode(result!.Images[0]));
            File.WriteAllBytes(Path.Combine(output, "sdxl-refiner-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result!.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"SDXL base + refiner: PSNR {psnr:F1} dB");
        Assert.True(psnr > 30, $"PSNR {psnr:F1} dB");
    }

    [Fact]
    public async Task RefinerRejectsItselfAndOtherLatentSpaces()
    {
        Assert.SkipUnless(RefinerPresent, "SDXL refiner or its references missing");
        using var model = Env.Load(Refiner, Env.Cpu());
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        await Assert.ThrowsAsync<NotSupportedException>(async () => await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "x" },
            new ImageGenerationOptions { Width = 64, Height = 64, Steps = 1, Refiner = generator }, TestContext.Current.CancellationToken));
    }
}
