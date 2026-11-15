using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random SDXL pipeline (tools/make-tiny-sdxl.py): both text encoders with the pooled embedding, the UNet with the added
/// size / pooled conditioning, the VAE, and whole text-to-image runs against diffusers, on the CPU and on Vulkan.
/// </summary>
public sealed class TinyXlTests
{
    private static readonly string Root = Env.TinyXl;

    private static string Diffusers => Path.Combine(Root, "diffusers");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Root, "reference", name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root, "reference", "index.json")));
        return document.RootElement.Clone();
    }

    public static TheoryData<string> Devices => ["cpu", "vulkan"];

    [Theory]
    [MemberData(nameof(Devices))]
    public void GraphsMatchDiffusers(string device)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Root, "reference", "index.json")), "tiny-random-sdxl missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Diffusers, slot!);
        var index = Index();
        var second = index.GetProperty("tokens_2").EnumerateObject().ToList();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int[] secondTokens = [.. second[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            var (maxAbs, rms) = Env.Compare(GraphRunner.Text(model, tokens), Tensor($"text_hidden_{i}"));
            Assert.True(rms < 1e-4, $"text {i}: max {maxAbs}, relative RMS {rms}");
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode2, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (maxAbs, rms) = Env.Compare(hidden, Tensor($"text2_hidden_{i}"));
            Assert.True(rms < 1e-4, $"text 2 {i}: max {maxAbs}, relative RMS {rms}");
            var pooled = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode2, DiffusionGraphs.Pooled, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (maxAbs, rms) = Env.Compare(pooled, Tensor($"text2_pooled_{i}"));
            Assert.True(rms < 1e-4, $"pooled {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var unet = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2,
            (DiffusionGraphs.Latent, Tensor("unet_latent")), (DiffusionGraphs.Timestep, Tensor("unet_timesteps")),
            (DiffusionGraphs.Context, Tensor("unet_context")), (DiffusionGraphs.Pooled, Tensor("unet_pooled")),
            (DiffusionGraphs.SizeConditions, Tensor("unet_time_ids")));
        var (unetMax, unetRms) = Env.Compare(unet, Tensor("unet_out"));
        Assert.True(unetRms < 1e-4, $"unet: max {unetMax}, relative RMS {unetRms}");

        const float scaling = 0.13025f;
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1,
            (DiffusionGraphs.Latent, Tensor("vae_latent").Select(v => v * scaling).ToArray()));
        var (vaeMax, vaeRms) = Env.Compare(decoded, Tensor("vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray());
        Assert.True(vaeRms < 1e-4, $"vae: max {vaeMax}, relative RMS {vaeRms}");
    }

    public static TheoryData<string, string, bool> Runs()
    {
        var data = new TheoryData<string, string, bool>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string sampler in new[] { Samplers.Euler, Samplers.Ddim, Samplers.DpmPlusPlus2MKarras })
            {
                data.Add(device, sampler, true);
            }

            data.Add(device, Samplers.Euler, false);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Runs))]
    public async Task TextToImageMatchesDiffusers(string device, string sampler, bool negative)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Root, "reference", "index.json")), "tiny-random-sdxl missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Diffusers, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Index();
        // Without a negative prompt the model's force_zeros_for_empty_prompt makes the unconditional row all zeros.
        var request = new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = negative ? index.GetProperty("negative_prompt").GetString() : null,
        };
        var options = new ImageGenerationOptions
        {
            Width = 64,
            Height = 64,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            Scheduler = sampler,
            InitialNoise = Tensor("noise"),
        };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor(sampler + (negative ? "" : "_noneg") + "_image"), 64, 64, 0f, 1f);
        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        Assert.True(psnr > 40, $"{sampler} on {device} (negative prompt: {negative}): PSNR {psnr:F1} dB");
    }
}
