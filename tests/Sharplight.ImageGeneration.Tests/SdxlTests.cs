using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// SDXL base 1.0 (two text encoders, pooled and size conditioning) on Vulkan against diffusers references
/// (tools/make-diffusion-references.py, 1024×1024, 20 steps, fp32) in ~/models/sd/sdxl-reference. Skips when the model, the
/// references or a GPU are missing.
/// </summary>
public sealed class SdxlTests
{
    private static readonly string Reference = Path.Combine(Path.GetDirectoryName(Env.SdxlDiffusers)!, "sdxl-reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
    {
        if (!Directory.Exists(Env.SdxlDiffusers) || !File.Exists(Path.Combine(Reference, "index.json")))
        {
            return null;
        }

        return Env.Vulkan();
    }

    [Fact]
    public void GraphsMatchDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "SDXL, its references or a Vulkan GPU missing");
        using var model = Env.Load(Env.SdxlDiffusers, slot!);
        var index = Index();
        var second = index.GetProperty("tokens_2").EnumerateObject().ToList();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            var hidden = GraphRunner.Text(model, [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())]);
            var (maxAbs, rms) = Env.Compare(hidden, Tensor($"text_hidden_{i}"));
            Assert.True(rms < 2e-3, $"text {i}: max {maxAbs}, relative RMS {rms}");
            int[] secondTokens = [.. second[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            var secondHidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode2, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (maxAbs, rms) = Env.Compare(secondHidden, Tensor($"text2_hidden_{i}"));
            TestContext.Current.SendDiagnosticMessage($"text 2 {i}: relative RMS {rms:E2}");
            Assert.True(rms < 5e-3, $"text 2 {i}: max {maxAbs}, relative RMS {rms}");
            var pooled = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode2, DiffusionGraphs.Pooled, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (maxAbs, rms) = Env.Compare(pooled, Tensor($"text2_pooled_{i}"));
            Assert.True(rms < 5e-3, $"pooled {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var unet = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2,
            (DiffusionGraphs.Latent, Tensor("unet_latent")), (DiffusionGraphs.Timestep, Tensor("unet_timesteps")), (DiffusionGraphs.Context, Tensor("unet_context")),
            (DiffusionGraphs.Pooled, Tensor("unet_pooled")), (DiffusionGraphs.SizeConditions, Tensor("unet_time_ids")));
        var (unetMax, unetRms) = Env.Compare(unet, Tensor("unet_out"));
        TestContext.Current.SendDiagnosticMessage($"unet: relative RMS {unetRms:E2}");
        Assert.True(unetRms < 5e-3, $"unet: max {unetMax}, relative RMS {unetRms}");

        const float scaling = 0.13025f;
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1,
            (DiffusionGraphs.Latent, Tensor("vae_latent").Select(v => v * scaling).ToArray()));
        var expected = Tensor("vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray();
        var (vaeMax, vaeRms) = Env.Compare(decoded, expected);
        TestContext.Current.SendDiagnosticMessage($"vae: relative RMS {vaeRms:E2}");
        Assert.True(vaeRms < 5e-3, $"vae: max {vaeMax}, relative RMS {vaeRms}");
    }

    public static TheoryData<string> SamplerNames => [Samplers.Euler, Samplers.DpmPlusPlus2MKarras];

    [Theory]
    [MemberData(nameof(SamplerNames))]
    public async Task TextToImageMatchesDiffusers(string sampler)
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "SDXL, its references or a Vulkan GPU missing");
        Assert.SkipUnless(File.Exists(Path.Combine(Reference, sampler + "_image.f32")), $"no {sampler} reference");
        using var model = Env.Load(Env.SdxlDiffusers, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Index();
        var result = await generator.GenerateAsync(
            new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, NegativePrompt = index.GetProperty("negative_prompt").GetString() },
            new ImageGenerationOptions
            {
                Steps = index.GetProperty("steps").GetInt32(),
                GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
                Scheduler = sampler,
                InitialNoise = Tensor("noise"),
            },
            TestContext.Current.CancellationToken);
        Assert.Equal(1024, result.Images[0].Width);   // SDXL's native size
        var expected = PlanarImage.FromPlanar(Tensor(sampler + "_image"), 1024, 1024, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"sdxl-{sampler}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"sdxl-{sampler}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{sampler}: PSNR {psnr:F1} dB");
        // fp16 weights (the published files): the multistep sampler drifts further along the trajectory than Euler.
        double threshold = sampler == Samplers.Euler ? 38 : 28;
        Assert.True(psnr > threshold, $"{sampler}: PSNR {psnr:F1} dB");
    }
}
