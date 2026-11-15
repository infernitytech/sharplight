using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// SDXL as the stable-diffusion.cpp GGUF files people download (see <see cref="Env.SdxlGguf"/>) on Vulkan against the diffusers references
/// of the same networks (tools/make-diffusion-references.py): SDXL base 1.0 (~/models/sd/sdxl-reference, 1024×1024, 20 steps, guidance
/// 7.5) and SDXL-Turbo (~/models/sd/sdxl-turbo-reference from stabilityai/sdxl-turbo: 512×512, 4 steps, no guidance, Euler on trailing
/// timesteps). The quantized weights bound the agreement; skips when a file, its reference or a GPU is missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class SdxlGgufTests
{
    /// <summary>The GGUF files: gpustack Q8_0 (sd.cpp names), kostakoff Q4_K (SGM names) and the SDXL-Turbo Q8_0.</summary>
    public static TheoryData<string> Variants => ["Q8_0", "Q4_K", "turbo"];

    private static string ReferenceOf(string variant) => Path.Combine(Path.GetDirectoryName(Env.SdxlDiffusers)!, variant == "turbo" ? "sdxl-turbo-reference" : "sdxl-reference");

    private static (double Graph, double Psnr) Tolerances(string variant) => variant switch
    {
        "Q4_K" => (0.15, 14),
        _ => (0.05, 20),
    };

    private static float[] Tensor(string reference, string name) => Env.ReadFloats(Path.Combine(reference, name + ".f32"));

    private static Runtime.DeviceSlot? Gpu(string variant)
    {
        string reference = ReferenceOf(variant);
        return File.Exists(Env.SdxlGguf(variant)) && File.Exists(Path.Combine(reference, "index.json")) ? Env.Vulkan() : null;
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void GraphsMatchDiffusers(string variant)
    {
        var slot = Gpu(variant);
        Assert.SkipWhen(slot is null, $"SDXL {variant} GGUF, its references or a Vulkan GPU missing");
        string reference = ReferenceOf(variant);
        double tolerance = Tolerances(variant).Graph;
        using var model = Env.Load(Env.SdxlGguf(variant), slot!);
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, "index.json")));
        var index = document.RootElement;
        var second = index.GetProperty("tokens_2").EnumerateObject().ToList();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            var hidden = GraphRunner.Text(model, [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())]);
            var (_, rms) = Env.Compare(hidden, Tensor(reference, $"text_hidden_{i}"));
            TestContext.Current.SendDiagnosticMessage($"{variant} text {i}: relative RMS {rms:E2}");
            Assert.True(rms < tolerance, $"text {i}: relative RMS {rms}");
            int[] secondTokens = [.. second[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            var secondHidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode2, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (_, rms) = Env.Compare(secondHidden, Tensor(reference, $"text2_hidden_{i}"));
            TestContext.Current.SendDiagnosticMessage($"{variant} text 2 {i}: relative RMS {rms:E2}");
            Assert.True(rms < tolerance, $"text 2 {i}: relative RMS {rms}");
            var pooled = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode2, DiffusionGraphs.Pooled, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (_, rms) = Env.Compare(pooled, Tensor(reference, $"text2_pooled_{i}"));
            TestContext.Current.SendDiagnosticMessage($"{variant} pooled {i}: relative RMS {rms:E2}");
            Assert.True(rms < tolerance, $"pooled {i}: relative RMS {rms}");
            i++;
        }

        var unet = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2,
            (DiffusionGraphs.Latent, Tensor(reference, "unet_latent")), (DiffusionGraphs.Timestep, Tensor(reference, "unet_timesteps")),
            (DiffusionGraphs.Context, Tensor(reference, "unet_context")), (DiffusionGraphs.Pooled, Tensor(reference, "unet_pooled")),
            (DiffusionGraphs.SizeConditions, Tensor(reference, "unet_time_ids")));
        var (_, unetRms) = Env.Compare(unet, Tensor(reference, "unet_out"));
        TestContext.Current.SendDiagnosticMessage($"{variant} unet: relative RMS {unetRms:E2}");
        Assert.True(unetRms < tolerance, $"unet: relative RMS {unetRms}");
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public async Task TextToImageMatchesDiffusers(string variant)
    {
        var slot = Gpu(variant);
        string reference = ReferenceOf(variant);
        Assert.SkipWhen(slot is null || !File.Exists(Path.Combine(reference, "euler_image.f32")), $"SDXL {variant} GGUF, its Euler reference or a Vulkan GPU missing");
        using var model = Env.Load(Env.SdxlGguf(variant), slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, "index.json")));
        var index = document.RootElement;
        var result = await generator.GenerateAsync(
            new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, NegativePrompt = index.GetProperty("negative_prompt").GetString() },
            new ImageGenerationOptions
            {
                Steps = index.GetProperty("steps").GetInt32(),
                GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
                Scheduler = Samplers.Euler,
                InitialNoise = Tensor(reference, "noise"),
            },
            TestContext.Current.CancellationToken);
        int size = variant == "turbo" ? 512 : 1024;
        Assert.Equal(size, result.Images[0].Width);   // the native size the importer chose
        var expected = PlanarImage.FromPlanar(Tensor(reference, "euler_image"), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"sdxl-gguf-{variant}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"sdxl-gguf-{variant}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{variant}: PSNR {psnr:F1} dB");
        Assert.True(psnr > Tolerances(variant).Psnr, $"{variant}: PSNR {psnr:F1} dB");
    }
}
