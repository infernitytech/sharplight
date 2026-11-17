using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// FLUX.2 Klein 4B (unsloth's transformer-only Q8_0 GGUF, with the Qwen3-4B text encoder folder and the VAE of the FLUX.2 Klein diffusers
/// folder as companions) on Vulkan against diffusers running the same files (tools/make-flux2-references.py: the dequantized GGUF
/// transformer on CUDA, the f32 text encoder, 4 distilled steps, same initial latents) in ~/models/flux2/flux2-klein-4b-q8-reference (the
/// root can be moved with <c>SHARPLIGHT_FLUX2_ROOT</c>). Skips when the files, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class Flux2KleinTests
{
    private static readonly string Root = Environment.GetEnvironmentVariable("SHARPLIGHT_FLUX2_ROOT") is { Length: > 0 } root ? root : Env.Flux2Root;
    private static readonly string Reference = Path.Combine(Root, "flux2-klein-4b-q8-reference");
    private static readonly string Folder = Path.Combine(Root, "FLUX.2-klein-4B");
    private static readonly string Gguf = Path.Combine(Root, "gguf", "flux-2-klein-4b-Q8_0.gguf");

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
        Assert.SkipWhen(slot is null, "FLUX.2 Klein, its references or a Vulkan GPU missing");
        using var model = Env.Load(Gguf, slot!, Companions());
        var index = Index();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int length = index.GetProperty("lengths").GetProperty(prompt.Name).GetInt32();
            float[] mask = [.. Enumerable.Range(0, tokens.Length).Select(t => t < length ? 1f : 0f)];
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, tokens), (DiffusionGraphs.TokenMask, mask));
            var (maxAbs, rms) = Env.Compare(hidden, Tensor($"text_hidden_{i}"));
            TestContext.Current.SendDiagnosticMessage($"text encoder {i}: relative RMS {rms:E2}");
            // bf16 weights against the f32 reference.
            Assert.True(rms < 2e-2, $"text encoder {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        int size = index.GetProperty("transformer_size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"transformer: relative RMS {transformerRms:E2}");
        Assert.True(transformerRms < 2e-2, $"transformer: max {transformerMax}, relative RMS {transformerRms}");
    }

    /// <summary>The pipeline folder's tokenizer gives transformers' chat-templated ids (thinking disabled).</summary>
    [Fact]
    public void PromptTokensMatchTransformers()
    {
        Assert.SkipUnless(Directory.Exists(Path.Combine(Folder, "tokenizer")) && File.Exists(Path.Combine(Reference, "index.json")), "FLUX.2 Klein or its references missing");
        using var model = Env.Load(Gguf, Env.Cpu(), Companions());
        var index = Index();
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        string template = model.Description.Component(ModelComponentRoles.Tokenizer).ConfigValue<string?>(DiffusionConfigKeys.PromptTemplate, null)!;
        int at = template.IndexOf("{prompt}", StringComparison.Ordinal);
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int length = index.GetProperty("lengths").GetProperty(prompt.Name).GetInt32();
            int[] expected = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32()).Take(length)];
            int[] actual = [.. tokenizer.Encode(template[..at], addSpecial: false, parseSpecial: true),
                .. tokenizer.Encode(prompt.Name, addSpecial: false, parseSpecial: false),
                .. tokenizer.Encode(template[(at + "{prompt}".Length)..], addSpecial: false, parseSpecial: true)];
            Assert.Equal(expected, actual);
        }
    }

    public static TheoryData<int> Sizes => [512, 1024];

    [Theory]
    [MemberData(nameof(Sizes))]
    public async Task TextToImageMatchesDiffusers(int size)
    {
        const string variant = "flux2-klein-4b";
        string gguf = Gguf;
        string reference = Reference;
        var slot = Gpu();
        Assert.SkipWhen(slot is null || !File.Exists(gguf), "FLUX.2 Klein, its references or a Vulkan GPU missing");
        Assert.SkipUnless(File.Exists(Path.Combine(reference, $"image_{size}.f32")), $"no {variant} {size} reference");
        float[] Tensor(string name) => Env.ReadFloats(Path.Combine(reference, name + ".f32"));
        var stopwatch = Stopwatch.StartNew();
        using var model = Env.Load(gguf, slot!, Companions());
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        double load = stopwatch.Elapsed.TotalSeconds;
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, "index.json")));
        var index = document.RootElement;
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
            File.WriteAllBytes(Path.Combine(output, $"{variant}-{size}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"{variant}-{size}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{variant} {size}: PSNR {psnr:F1} dB; load {load:F1} s, first run (compiling) {first:F1} s, second run {second:F1} s");
        Assert.True(psnr > 25, $"{size}: PSNR {psnr:F1} dB");
    }
}
