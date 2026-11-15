using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Stable Diffusion 1.5 on Vulkan against diffusers references (tools/make-diffusion-references.py, 512×512, 20 steps) in
/// ~/models/sd/sd15-reference. Skips when the model, the references or a GPU are missing.
/// </summary>
public sealed class Sd15Tests
{
    private static readonly string Reference = Path.Combine(Path.GetDirectoryName(Env.Sd15Diffusers)!, "sd15-reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
    {
        if (!Directory.Exists(Env.Sd15Diffusers) || !File.Exists(Path.Combine(Reference, "index.json")))
        {
            return null;
        }

        return Env.Vulkan();
    }

    [Fact]
    public void GraphsMatchDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "SD 1.5, its references or a Vulkan GPU missing");
        using var model = Env.Load(Env.Sd15Diffusers, slot!);
        var index = Index();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            var hidden = GraphRunner.Text(model, [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())]);
            var (maxAbs, rms) = Env.Compare(hidden, Tensor($"text_hidden_{i}"));
            Assert.True(rms < 2e-3, $"text {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var unet = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2,
            (DiffusionGraphs.Latent, Tensor("unet_latent")), (DiffusionGraphs.Timestep, Tensor("unet_timesteps")), (DiffusionGraphs.Context, Tensor("unet_context")));
        var (unetMax, unetRms) = Env.Compare(unet, Tensor("unet_out"));
        Assert.True(unetRms < 5e-3, $"unet: max {unetMax}, relative RMS {unetRms}");

        const float scaling = 0.18215f;
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1,
            (DiffusionGraphs.Latent, Tensor("vae_latent").Select(v => v * scaling).ToArray()));
        var expected = Tensor("vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray();
        var (vaeMax, vaeRms) = Env.Compare(decoded, expected);
        Assert.True(vaeRms < 5e-3, $"vae: max {vaeMax}, relative RMS {vaeRms}");
    }

    public static TheoryData<string> SamplerNames => [Samplers.Euler, Samplers.EulerKarras, Samplers.Ddim, Samplers.Pndm, Samplers.DpmPlusPlus2M, Samplers.DpmPlusPlus2MKarras];

    [Theory]
    [MemberData(nameof(SamplerNames))]
    public async Task TextToImageMatchesDiffusers(string sampler)
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "SD 1.5, its references or a Vulkan GPU missing");
        Assert.SkipUnless(File.Exists(Path.Combine(Reference, sampler + "_image.f32")), $"no {sampler} reference");
        using var model = Env.Load(Env.Sd15Diffusers, slot!);
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
        var expected = PlanarImage.FromPlanar(Tensor(sampler + "_image"), 512, 512, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"sd15-{sampler}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"sd15-{sampler}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{sampler}: PSNR {psnr:F1} dB");
        Assert.True(psnr > 35, $"{sampler}: PSNR {psnr:F1} dB");
    }
}
