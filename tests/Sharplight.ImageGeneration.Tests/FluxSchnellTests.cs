using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// FLUX.1-schnell on Vulkan against diffusers references (tools/make-flux-references.py: f32 on the CPU, 4 steps, 512×512 and
/// 1024×1024, same initial latents) in ~/models/flux/schnell-reference: from the diffusers folder (bf16 files) and from city96's
/// transformer-only Q8_0 GGUF with the T5 Q8_0 GGUF, clip_l.safetensors and ae.safetensors (found beside it). Skips when the files,
/// the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class FluxSchnellTests
{
    private static readonly string Reference = Path.Combine(Env.FluxRoot, "schnell-reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static string ModelPath(string source) => source == "gguf" ? Env.FluxSchnellGguf : Env.FluxSchnellDiffusers;

    private static Runtime.DeviceSlot? Gpu(string source)
    {
        string path = ModelPath(source);
        if (!(Directory.Exists(path) || File.Exists(path)) || !File.Exists(Path.Combine(Reference, "index.json")))
        {
            return null;
        }

        return Env.Vulkan();
    }

    public static TheoryData<string> Sources => ["diffusers", "gguf"];

    [Theory]
    [MemberData(nameof(Sources))]
    public void GraphsMatchDiffusers(string source)
    {
        var slot = Gpu(source);
        Assert.SkipWhen(slot is null, "FLUX.1-schnell, its references or a Vulkan GPU missing");
        using var model = Env.Load(ModelPath(source), slot!);
        var index = Index();
        var second = index.GetProperty("tokens_2").EnumerateObject().ToList();
        // Q8_0 weights (GGUF) against the bf16 reference: the quantization error, not the implementation's.
        double tolerance = source == "gguf" ? 3e-2 : 5e-3;
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int[] secondTokens = [.. second[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            var pooled = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Pooled, 0, (DiffusionGraphs.Tokens, tokens));
            var (maxAbs, rms) = Env.Compare(pooled, Tensor($"text_pooled_{i}"));
            TestContext.Current.SendDiagnosticMessage($"{source} CLIP pooled {i}: relative RMS {rms:E2}");
            // clip_l.safetensors (beside the GGUF) holds f16 weights, the diffusers folder bf16 ones.
            Assert.True(rms < (source == "gguf" ? 2e-2 : 5e-3), $"pooled {i}: max {maxAbs}, relative RMS {rms}");
            var hidden = GraphRunner.Run(model, DiffusionGraphs.SecondTextEncoderRole, DiffusionGraphs.TextEncode2, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (maxAbs, rms) = Env.Compare(hidden, Tensor($"text2_hidden_{i}"));
            TestContext.Current.SendDiagnosticMessage($"{source} T5 {i}: relative RMS {rms:E2}");
            Assert.True(rms < tolerance, $"T5 {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        int size = index.GetProperty("transformer_size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")), (DiffusionGraphs.Pooled, Tensor("transformer_pooled")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"{source} transformer: relative RMS {transformerRms:E2}");
        Assert.True(transformerRms < tolerance, $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        const float scaling = 0.3611f;
        const float shift = 0.1159f;
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1, dims,
            (DiffusionGraphs.Latent, Tensor("vae_latent").Select(v => (v - shift) * scaling).ToArray()));
        var (vaeMax, vaeRms) = Env.Compare(decoded, Tensor("vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray());
        TestContext.Current.SendDiagnosticMessage($"{source} vae: relative RMS {vaeRms:E2}");
        Assert.True(vaeRms < 5e-3, $"vae: max {vaeMax}, relative RMS {vaeRms}");
    }

    public static TheoryData<string, int> Runs()
    {
        var data = new TheoryData<string, int>();
        foreach (string source in new[] { "diffusers", "gguf" })
        {
            data.Add(source, 512);
            data.Add(source, 1024);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Runs))]
    public async Task TextToImageMatchesDiffusers(string source, int size)
    {
        var slot = Gpu(source);
        Assert.SkipWhen(slot is null, "FLUX.1-schnell, its references or a Vulkan GPU missing");
        Assert.SkipUnless(File.Exists(Path.Combine(Reference, $"image_{size}.f32")), $"no {size} reference");
        var stopwatch = Stopwatch.StartNew();
        using var model = Env.Load(ModelPath(source), slot!);
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
            File.WriteAllBytes(Path.Combine(output, $"flux-schnell-{source}-{size}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"flux-schnell-{size}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{source} {size}: PSNR {psnr:F1} dB; load {load:F1} s, first run (compiling) {first:F1} s, second run {second:F1} s");
        Assert.True(psnr > (source == "gguf" ? 25 : 35), $"{source} {size}: PSNR {psnr:F1} dB");
    }

    [Fact]
    public void CompanionsAreFoundBesideTheTransformer()
    {
        Assert.SkipUnless(File.Exists(Env.FluxSchnellGguf), "FLUX.1-schnell GGUF missing");
        var companions = TestHost.Loader.Importers.FindCompanions(Env.FluxSchnellGguf);
        Assert.Equal(3, companions.Count);
        Assert.EndsWith(".gguf", companions[AuxiliaryRoles.SecondTextEncoder], StringComparison.Ordinal);
        Assert.Equal("clip_l.safetensors", Path.GetFileName(companions[AuxiliaryRoles.TextEncoder]));
        Assert.Equal("ae.safetensors", Path.GetFileName(companions[AuxiliaryRoles.Vae]));
    }
}
