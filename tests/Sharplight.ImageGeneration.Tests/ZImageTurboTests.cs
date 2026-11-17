using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Z-Image-Turbo (unsloth's transformer-only Q8_0 GGUF, with the Qwen3-4B text encoder folder and the VAE of the Z-Image-Turbo
/// diffusers folder as companions) on Vulkan against diffusers running the same files (tools/make-zimage-references.py: the dequantized
/// GGUF transformer on CUDA, the f32 text encoder, 9 steps, no guidance, same initial latents) in ~/models/zimage/zimage-turbo-reference.
/// Skips when the files, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class ZImageTurboTests
{
    private static readonly string Reference = Path.Combine(Env.ZImageRoot, "zimage-turbo-reference");
    private static readonly string Folder = Path.Combine(Env.ZImageRoot, "Z-Image-Turbo");
    private static readonly string Gguf = Path.Combine(Env.ZImageRoot, "gguf", "z-image-turbo-Q8_0.gguf");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Dictionary<string, string> Companions() => new()
    {
        [AuxiliaryRoles.TextEncoder] = Path.Combine(Folder, "text_encoder"),
        [AuxiliaryRoles.Vae] = Path.Combine(Folder, "vae", "diffusion_pytorch_model.safetensors"),
    };

    private static Runtime.DeviceSlot? Gpu()
        => File.Exists(Gguf) && Directory.Exists(Path.Combine(Folder, "text_encoder")) && File.Exists(Path.Combine(Reference, "index.json")) ? Env.Vulkan() : null;

    [Fact]
    public void GraphsMatchDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "Z-Image-Turbo, its references or a Vulkan GPU missing");
        using var model = Env.Load(Gguf, slot!, Companions());
        var index = Index();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int length = index.GetProperty("lengths").GetProperty(prompt.Name).GetInt32();
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, tokens));
            int d = hidden.Length / tokens.Length;
            var (maxAbs, rms) = Env.Compare(hidden.AsSpan(0, length * d), Tensor($"text_hidden_{i}").AsSpan(0, length * d));
            TestContext.Current.SendDiagnosticMessage($"text encoder {i}: relative RMS {rms:E2}");
            // bf16 weights against the f32 reference.
            Assert.True(rms < 2e-2, $"text encoder {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        int size = index.GetProperty("transformer_size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")), (DiffusionGraphs.ContextMask, Tensor("transformer_context_mask")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"transformer: relative RMS {transformerRms:E2}");
        Assert.True(transformerRms < 2e-2, $"transformer: max {transformerMax}, relative RMS {transformerRms}");
    }

    /// <summary>The transformer on the CPU (diagnostics: runs only with <c>SHARPLIGHT_ZIMAGE_CPU=1</c>; minutes).</summary>
    /// <summary>
    /// unsloth's Qwen3-4B Q8_0 GGUF (Z-Image's text encoder is exactly Qwen/Qwen3-4B) as the companion against the pipeline's encoder (bf16,
    /// run in f32 by diffusers): the Q8_0 weights cost accuracy; the text graphs keep f32 activations, which int8 quantization would lose.
    /// </summary>
    [Fact]
    public void GgufTextEncoderMatchesDiffusers()
    {
        string textEncoder = Env.Qwen3FourBGguf;
        var slot = Gpu();
        Assert.SkipWhen(slot is null || !File.Exists(textEncoder), "Z-Image-Turbo, the Qwen3-4B GGUF, its references or a Vulkan GPU missing");
        var companions = Companions();
        companions[AuxiliaryRoles.TextEncoder] = textEncoder;
        using var model = Env.Load(Gguf, slot!, companions);
        var index = Index();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int length = index.GetProperty("lengths").GetProperty(prompt.Name).GetInt32();
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, tokens));
            int d = hidden.Length / tokens.Length;
            var (maxAbs, rms) = Env.Compare(hidden.AsSpan(0, length * d), Tensor($"text_hidden_{i}").AsSpan(0, length * d));
            TestContext.Current.SendDiagnosticMessage($"GGUF text encoder {i}: relative RMS {rms:E2}");
            Assert.True(rms < 3e-2, $"GGUF text encoder {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }
    }

    [Fact]
    public void TransformerOnTheCpuMatchesDiffusers()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_ZIMAGE_CPU") == "1", "CPU diagnostics not requested");
        Assert.SkipUnless(File.Exists(Gguf) && File.Exists(Path.Combine(Reference, "index.json")), "Z-Image-Turbo or its references missing");
        using var model = Env.Load(Gguf, Env.Cpu(), Companions());
        int size = Index().GetProperty("transformer_size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")), (DiffusionGraphs.ContextMask, Tensor("transformer_context_mask")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"CPU transformer: relative RMS {transformerRms:E2}, NaN {prediction.Count(float.IsNaN)}");
        Assert.True(transformerRms < 3e-2, $"transformer: max {transformerMax}, relative RMS {transformerRms}");
    }

    public static TheoryData<int> Sizes => [512, 1024];

    [Theory]
    [MemberData(nameof(Sizes))]
    public async Task TextToImageMatchesDiffusers(int size)
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "Z-Image-Turbo, its references or a Vulkan GPU missing");
        Assert.SkipUnless(File.Exists(Path.Combine(Reference, $"image_{size}.f32")), $"no {size} reference");
        var stopwatch = Stopwatch.StartNew();
        using var model = Env.Load(Gguf, slot!, Companions());
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        double load = stopwatch.Elapsed.TotalSeconds;
        var index = Index();
        var options = new ImageGenerationOptions
        {
            Width = size,
            Height = size,
            Steps = index.GetProperty("steps").GetInt32(),
            InitialNoise = Tensor($"noise_{size}"),
        };
        var request = new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! };
        stopwatch.Restart();
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
            File.WriteAllBytes(Path.Combine(output, $"zimage-turbo-{size}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"zimage-turbo-{size}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{size}: PSNR {psnr:F1} dB; load {load:F1} s, first run (compiling) {first:F1} s, second run {second:F1} s");
        Assert.True(psnr > 25, $"{size}: PSNR {psnr:F1} dB");
    }
}
