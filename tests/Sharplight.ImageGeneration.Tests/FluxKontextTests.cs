using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// FLUX.1 Kontext dev (QuantStack's Q8_0 GGUF, recognized by its name, with the FLUX.1 companions of the schnell GGUF folder: CLIP-L, the T5
/// Q8_0 GGUF and the VAE) editing an image on Vulkan, against diffusers' FluxKontextPipeline running the same files
/// (tools/make-flux-kontext-references.py: the dequantized GGUF transformer on CUDA, f32 text encoders, embedded guidance 2.5, same initial
/// latents) in ~/models/flux/kontext-q8-reference (the root can be moved with <c>SHARPLIGHT_KONTEXT_ROOT</c>). Skips when the files, the
/// references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class FluxKontextTests
{
    private static readonly string Root = Environment.GetEnvironmentVariable("SHARPLIGHT_KONTEXT_ROOT") is { Length: > 0 } root ? root : Env.FluxRoot;
    private static readonly string Reference = Path.Combine(Root, "kontext-q8-reference");
    private static readonly string Gguf = Path.Combine(Root, "gguf-kontext", "flux1-kontext-dev-Q8_0.gguf");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
        => File.Exists(Gguf) && File.Exists(Env.FluxSchnellGguf) && File.Exists(Path.Combine(Reference, "index.json")) ? Env.Vulkan() : null;

    private static Tasks.TaskModel Load(Runtime.DeviceSlot slot)
    {
        string companions = Path.GetDirectoryName(Env.FluxSchnellGguf)!;
        return Env.Load(Gguf, slot, new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = Path.Combine(companions, "clip_l.safetensors"),
            [AuxiliaryRoles.SecondTextEncoder] = Path.Combine(companions, "t5-v1_1-xxl-encoder-Q8_0.gguf"),
            [AuxiliaryRoles.Vae] = Path.Combine(companions, "ae.safetensors"),
        });
    }

    [Fact]
    public void TransformerWithReferenceMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "FLUX.1 Kontext GGUF, its references or a Vulkan GPU missing");
        using var model = Load(slot!);
        Assert.Equal(1, model.Description.Component(ModelComponentRoles.Denoiser).ConfigValue(DiffusionConfigKeys.ReferenceImages, 0));
        Assert.Equal(2.5, model.Description.ConfigValue(DiffusionConfigKeys.GuidanceScale, 0.0));
        int size = Index().GetProperty("size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size, [DiffusionGraphs.References] = 1 };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.ReferenceLatent, Tensor("transformer_reference")),
            (DiffusionGraphs.Timestep, Tensor("transformer_timestep")), (DiffusionGraphs.Context, Tensor("transformer_context")),
            (DiffusionGraphs.Pooled, Tensor("transformer_pooled")), (DiffusionGraphs.Guidance, Tensor("transformer_guidance")));
        var (max, rms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"kontext transformer: relative RMS {rms:E2}");
        Assert.True(rms < 2e-2, $"transformer: max {max}, relative RMS {rms}");
    }

    [Fact]
    public async Task EditMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "FLUX.1 Kontext GGUF, its references or a Vulkan GPU missing");
        var index = Index();
        int size = index.GetProperty("size").GetInt32();
        var stopwatch = Stopwatch.StartNew();
        using var model = Load(slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var options = new ImageGenerationOptions
        {
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            InitialNoise = Tensor($"noise_{size}"),
        };
        var input = PlanarImage.FromPlanar(Tensor("edit_input"), size, size, 0f, 1f);
        var request = new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, ReferenceImages = [input] };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        double first = stopwatch.Elapsed.TotalSeconds;
        stopwatch.Restart();
        result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        double second = stopwatch.Elapsed.TotalSeconds;
        var expected = PlanarImage.FromPlanar(Tensor($"image_{size}"), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"kontext-{size}-input.png"), PngCodec.Encode(input));
            File.WriteAllBytes(Path.Combine(output, $"kontext-{size}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"kontext-{size}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        double unchanged = Env.Psnr(result.Images[0].Rgb, input.Rgb);
        TestContext.Current.SendDiagnosticMessage($"kontext edit {size}: PSNR {psnr:F1} dB (vs the input {unchanged:F1} dB); first run {first:F1} s, second run {second:F1} s");
        Assert.True(psnr > 25, $"{size}: PSNR {psnr:F1} dB");
    }
}
