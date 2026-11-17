using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// FLUX.2 Dev (unsloth's transformer-only Q4_K_M GGUF, with unsloth's Mistral-Small-3.2-24B-Instruct-2506 Q8_0 GGUF, FLUX.2 Dev's text
/// encoder, and the VAE of the FLUX.2 Dev diffusers folder as companions) on Vulkan against diffusers running the same files
/// (tools/make-flux2-dev-references.py: the dequantized GGUF transformer on CUDA, the dequantized text encoder in f32 on the CPU, embedded
/// guidance 4, same initial latents) in ~/models/flux2-dev/dev-q4km-reference (the root can be moved with <c>SHARPLIGHT_FLUX2_DEV_ROOT</c>).
/// Skips when the files, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class Flux2DevTests
{
    private static readonly string Root = Environment.GetEnvironmentVariable("SHARPLIGHT_FLUX2_DEV_ROOT") is { Length: > 0 } root ? root : Env.Flux2DevRoot;
    private static readonly string Reference = Path.Combine(Root, "dev-q4km-reference");
    private static readonly string Folder = Path.Combine(Root, "FLUX.2-dev");
    private static readonly string Gguf = Path.Combine(Root, "gguf", "flux2-dev-Q4_K_M.gguf");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static readonly string TextEncoder = Path.Combine(Root, "gguf", "Mistral-Small-3.2-24B-Instruct-2506-Q8_0.gguf");

    private static Dictionary<string, string> Companions() => new()
    {
        [AuxiliaryRoles.TextEncoder] = TextEncoder,
        [AuxiliaryRoles.Vae] = Path.Combine(Folder, "vae", "diffusion_pytorch_model.safetensors"),
    };

    private static Runtime.DeviceSlot? Gpu()
        => File.Exists(Gguf) && File.Exists(TextEncoder) && File.Exists(Path.Combine(Reference, "index.json")) ? Env.Vulkan() : null;

    [Fact]
    public void GraphsMatchDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "FLUX.2 Dev, its references or a Vulkan GPU missing");
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
            int d = hidden.Length / tokens.Length;
            var (_, promptRms) = Env.Compare(hidden.AsSpan(0, length * d), Tensor($"text_hidden_{i}").AsSpan(0, length * d));
            var (_, firstRms) = Env.Compare(hidden.AsSpan(0, d), Tensor($"text_hidden_{i}").AsSpan(0, d));
            var (_, layerRms) = Env.Compare(hidden.AsSpan(d + 0, d / 3), Tensor($"text_hidden_{i}").AsSpan(d, d / 3));
            TestContext.Current.SendDiagnosticMessage($"text encoder {i}: relative RMS {rms:E2} (prompt rows {promptRms:E2}, first row {firstRms:E2}, row 1 layer 10 {layerRms:E2})");
            Assert.True(rms < 2e-2, $"text encoder {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        int size = index.GetProperty("transformer_size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")), (DiffusionGraphs.Guidance, Tensor("transformer_guidance")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"transformer: relative RMS {transformerRms:E2}");
        Assert.True(transformerRms < 2e-2, $"transformer: max {transformerMax}, relative RMS {transformerRms}");
    }

    /// <summary>The text encoder GGUF's tokenizer gives the ids of the pipeline's processor (Mistral's system-prompt template).</summary>
    [Fact]
    public void PromptTokensMatchTransformers()
    {
        Assert.SkipUnless(File.Exists(TextEncoder) && File.Exists(Path.Combine(Reference, "index.json")), "FLUX.2 Dev or its references missing");
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
        const string variant = "flux2-dev";
        string gguf = Gguf;
        string reference = Reference;
        var slot = Gpu();
        Assert.SkipWhen(slot is null || !File.Exists(gguf), "FLUX.2 Dev, its references or a Vulkan GPU missing");
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
            GuidanceScale = index.GetProperty("guidance").GetSingle(),
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
