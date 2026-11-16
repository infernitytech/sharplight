using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Stable Diffusion 3.5 Medium on Vulkan against diffusers references (tools/make-sd3-references.py: f32 weights on CUDA, 28 steps,
/// guidance 7, 512×512 and 1024×1024, same initial latents) in ~/models/sd3/sd35-medium-reference: from the diffusers folder (fp16
/// files) and from city96's transformer-only Q8_0 GGUF with the folder's CLIP and VAE files and the T5 encoder GGUF as companions. Skips when the files, the
/// references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class Sd35MediumTests
{
    private static readonly string Reference = Path.Combine(Env.Sd3Root, "sd35-medium-reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static string ModelPath(string source) => source == "gguf" ? Env.Sd35MediumGguf : Env.Sd35MediumDiffusers;

    /// <summary>The companions of the transformer-only GGUF: the folder's CLIP and VAE files, city96's T5-XXL encoder Q8_0 GGUF (FLUX's).</summary>
    private static Dictionary<string, string> Companions() => new()
    {
        [AuxiliaryRoles.TextEncoder] = Path.Combine(Env.Sd35MediumDiffusers, "text_encoder", "model.safetensors"),
        [AuxiliaryRoles.SecondTextEncoder] = Path.Combine(Env.Sd35MediumDiffusers, "text_encoder_2", "model.safetensors"),
        [AuxiliaryRoles.ThirdTextEncoder] = Env.T5XxlGguf,
        [AuxiliaryRoles.Vae] = Path.Combine(Env.Sd35MediumDiffusers, "vae", "diffusion_pytorch_model.safetensors"),
    };

    private static Tasks.TaskModel Load(string source, Runtime.DeviceSlot slot)
        => source == "gguf" ? Env.Load(ModelPath(source), slot, Companions()) : Env.Load(ModelPath(source), slot);

    private static Runtime.DeviceSlot? Gpu(string source)
    {
        string path = ModelPath(source);
        if (!(Directory.Exists(path) || File.Exists(path)) || !Directory.Exists(Env.Sd35MediumDiffusers) || !File.Exists(Path.Combine(Reference, "index.json")))
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
        Assert.SkipWhen(slot is null, "SD3.5 Medium, its references or a Vulkan GPU missing");
        using var model = Load(source, slot!);
        var index = Index();
        var second = index.GetProperty("tokens_2").EnumerateObject().ToList();
        var third = index.GetProperty("tokens_3").EnumerateObject().ToList();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int[] secondTokens = [.. second[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            int[] thirdTokens = [.. third[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            foreach (var (graph, ids, name) in new[] { (DiffusionGraphs.TextEncode, tokens, "text"), (DiffusionGraphs.TextEncode2, secondTokens, "text2") })
            {
                foreach (string output in new[] { DiffusionGraphs.Pooled, DiffusionGraphs.Hidden })
                {
                    var actual = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, graph, output, 0, (DiffusionGraphs.Tokens, ids));
                    var (maxAbs, rms) = Env.Compare(actual, Tensor($"{name}_{output}_{i}"));
                    TestContext.Current.SendDiagnosticMessage($"{source} {name} {output} {i}: relative RMS {rms:E2}");
                    Assert.True(rms < 5e-3, $"{name} {output} {i}: max {maxAbs}, relative RMS {rms}");
                }
            }

            var t5 = GraphRunner.Run(model, DiffusionGraphs.ThirdTextEncoderRole, DiffusionGraphs.TextEncode3, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, thirdTokens));
            var (t5Max, t5Rms) = Env.Compare(t5, Tensor($"text3_hidden_{i}"));
            TestContext.Current.SendDiagnosticMessage($"{source} T5 {i}: relative RMS {t5Rms:E2}");
            // The T5 encoder GGUF (Q8_0, every matrix) is ~20 % from the f32 encoder's output (see FluxSchnellTests).
            Assert.True(t5Rms < (source == "gguf" ? 0.3 : 5e-3), $"T5 {i}: max {t5Max}, relative RMS {t5Rms}");
            i++;
        }

        int size = index.GetProperty("transformer_size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")), (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")), (DiffusionGraphs.Pooled, Tensor("transformer_pooled")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"{source} transformer: relative RMS {transformerRms:E2}");
        Assert.True(transformerRms < (source == "gguf" ? 5e-2 : 1e-2), $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        const float scaling = 1.5305f;
        const float shift = 0.0609f;
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
        Assert.SkipWhen(slot is null, "SD3.5 Medium, its references or a Vulkan GPU missing");
        Assert.SkipUnless(File.Exists(Path.Combine(Reference, $"image_{size}.f32")), $"no {size} reference");
        var stopwatch = Stopwatch.StartNew();
        using var model = Load(source, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        double load = stopwatch.Elapsed.TotalSeconds;
        var index = Index();
        var options = new ImageGenerationOptions
        {
            Width = size,
            Height = size,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
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
            File.WriteAllBytes(Path.Combine(output, $"sd35-medium-{source}-{size}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"sd35-medium-{size}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{source} {size}: PSNR {psnr:F1} dB; load {load:F1} s, first run (compiling) {first:F1} s, second run {second:F1} s");
        Assert.True(psnr > (source == "gguf" ? 18 : 25), $"{source} {size}: PSNR {psnr:F1} dB");
    }
}
