using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// FLUX.1-dev (guidance embedding, resolution-dependent shift) from city96's Q8_0 GGUF on Vulkan, with the companions of the schnell
/// GGUF folder (the text encoders and VAE are the same), against diffusers running the same GGUF transformer (dequantized, f32 on the
/// CPU; tools/make-flux-references.py --transformer-gguf --dev) in ~/models/flux/dev-q8-reference. Skips when the files, the references
/// or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class FluxDevTests
{
    private static readonly string Reference = Path.Combine(Env.FluxRoot, "dev-q8-reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
        => File.Exists(Env.FluxDevGguf) && File.Exists(Env.FluxSchnellGguf) && File.Exists(Path.Combine(Reference, "index.json")) ? Env.Vulkan() : null;

    private static Tasks.TaskModel Load(Runtime.DeviceSlot slot)
    {
        string companions = Path.GetDirectoryName(Env.FluxSchnellGguf)!;
        return Env.Load(Env.FluxDevGguf, slot, new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = Path.Combine(companions, "clip_l.safetensors"),
            [AuxiliaryRoles.SecondTextEncoder] = Path.Combine(companions, "t5-v1_1-xxl-encoder-Q8_0.gguf"),
            [AuxiliaryRoles.Vae] = Path.Combine(companions, "ae.safetensors"),
        });
    }

    [Fact]
    public void TransformerMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "FLUX.1-dev GGUF, its references or a Vulkan GPU missing");
        using var model = Load(slot!);
        Assert.Equal(28, model.Description.ConfigValue(DiffusionConfigKeys.Steps, 0));
        Assert.Equal(3.5, model.Description.ConfigValue(DiffusionConfigKeys.GuidanceScale, 0.0));
        int size = Index().GetProperty("transformer_size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")), (DiffusionGraphs.Pooled, Tensor("transformer_pooled")),
            (DiffusionGraphs.Guidance, Tensor("transformer_guidance")));
        var (max, rms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"dev transformer: relative RMS {rms:E2}");
        Assert.True(rms < 1e-2, $"transformer: max {max}, relative RMS {rms}");
    }

    [Fact]
    public async Task TextToImageMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "FLUX.1-dev GGUF, its references or a Vulkan GPU missing");
        var index = Index();
        int size = index.GetProperty("sizes")[0].GetInt32();
        using var model = Load(slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var options = new ImageGenerationOptions
        {
            Width = size,
            Height = size,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            InitialNoise = Tensor($"noise_{size}"),
        };
        var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, options,
            TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor($"image_{size}"), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"flux-dev-gguf-{size}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"flux-dev-gguf-{size}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"dev {size}: PSNR {psnr:F1} dB");
        Assert.True(psnr > 30, $"dev {size}: PSNR {psnr:F1} dB");
    }
}
