using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.VideoGeneration;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// LTX-Video 0.9.0 and 0.9.5 (diffusers folders, the T5-XXL encoder shared with FLUX.1) on Vulkan against diffusers references
/// (tools/make-ltx-references.py: f32 on CUDA, 25 frames at 704×480, 20 Euler steps, guidance 3, the same initial latents) in
/// ~/models/video/ltx-reference and ltx095-reference: the T5 conditioning, one transformer evaluation and the whole text-to-video run.
/// Skips when the model, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class LtxVideoTests
{
    /// <summary>"0.9.0" and "0.9.5".</summary>
    public static TheoryData<string> Variants => ["0.9.0", "0.9.5"];

    private static string ModelPath(string variant) => Path.Combine(Env.VideoRoot, variant == "0.9.0" ? "LTX-Video" : "LTX-Video-0.9.5");

    private static string Reference(string variant) => Path.Combine(Env.VideoRoot, variant == "0.9.0" ? "ltx-reference" : "ltx095-reference", "reference");

    private static float[] Tensor(string variant, string name) => Env.ReadFloats(Path.Combine(Reference(variant), name + ".f32"));

    private static JsonElement Index(string variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference(variant), "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu(string variant)
        => Directory.Exists(ModelPath(variant)) && File.Exists(Path.Combine(Reference(variant), "index.json")) ? Env.Vulkan() : null;

    private static Dictionary<string, long> Dimensions(JsonElement index) => new()
    {
        [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 8 + 1,
        [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
        [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
    };

    [Theory]
    [MemberData(nameof(Variants))]
    public void GraphsMatchDiffusers(string variant)
    {
        var slot = Gpu(variant);
        Assert.SkipWhen(slot is null, "LTX-Video, its references or a Vulkan GPU missing");
        using var model = Env.Load(ModelPath(variant), slot!);
        var index = Index(variant);
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        foreach (var (prompt, name) in new[] { (index.GetProperty("prompt").GetString()!, "text_hidden"), (index.GetProperty("negative_prompt").GetString()!, "negative_hidden") })
        {
            var ids = tokenizer.Encode(prompt, addSpecial: true, parseSpecial: false);
            var tokens = new int[128];
            ids.CopyTo(tokens);
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.Hidden, 0,
                (VideoDiffusionGraphs.Tokens, tokens));
            var (maxAbs, rms) = Env.Compare(hidden, Tensor(variant, name));
            TestContext.Current.SendDiagnosticMessage($"LTX {variant} T5 {name}: relative RMS {rms:E2}");
            Assert.True(rms < 5e-3, $"T5 {name}: max {maxAbs}, relative RMS {rms}");
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, Dimensions(index),
            (VideoDiffusionGraphs.Latent, Tensor(variant, "noise")),
            (VideoDiffusionGraphs.Timestep, new[] { index.GetProperty("transformer_timestep").GetSingle() }),
            (VideoDiffusionGraphs.Context, Tensor(variant, "text_hidden")),
            (VideoDiffusionGraphs.ContextMask, Tensor(variant, "text_mask")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(variant, "transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"LTX {variant} transformer: relative RMS {transformerRms:E2}");
        Assert.True(transformerRms < 5e-3, $"transformer: max {transformerMax}, relative RMS {transformerRms}");
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public async Task TextToVideoMatchesDiffusers(string variant)
    {
        var slot = Gpu(variant);
        Assert.SkipWhen(slot is null, "LTX-Video, its references or a Vulkan GPU missing");
        using var model = Env.Load(ModelPath(variant), slot!);
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var index = Index(variant);
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
        var request = new VideoGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, NegativePrompt = index.GetProperty("negative_prompt").GetString() };
        var options = new VideoGenerationOptions
        {
            Width = width,
            Height = height,
            Frames = index.GetProperty("frames").GetInt32(),
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            InitialNoise = Tensor(variant, "noise"),
        };
        var stopwatch = Stopwatch.StartNew();
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        double seconds = stopwatch.Elapsed.TotalSeconds;
        var expected = Tensor(variant, "video");
        int frameValues = 3 * width * height;
        double worst = double.MaxValue;
        for (int f = 0; f < result.Frames.Count; f++)
        {
            var reference = PlanarImage.FromPlanar(expected.AsSpan(f * frameValues, frameValues), width, height, 0f, 1f);
            worst = Math.Min(worst, Env.Psnr(result.Frames[f].Rgb, reference.Rgb));
        }

        TestContext.Current.SendDiagnosticMessage($"LTX {variant} {width}x{height}x{result.Frames.Count}: {seconds:F1} s (incl. compilation), worst frame {worst:F1} dB");
        Assert.True(worst > 25, $"LTX: worst frame PSNR {worst:F1} dB");
    }
}
