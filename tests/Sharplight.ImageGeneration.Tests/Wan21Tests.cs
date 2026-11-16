using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.VideoGeneration;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Wan 2.1 T2V-1.3B (diffusers folder) on Vulkan against diffusers references (tools/make-wan-references.py: f32 on CUDA, 17 frames
/// at 832×480, 20 UniPC steps, guidance 5, the same initial latents) in ~/models/video/wan21-reference: the umT5 conditioning, one
/// transformer evaluation and the whole text-to-video run. Skips when the model, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class Wan21Tests
{
    private static readonly string ModelPath = Path.Combine(Env.VideoRoot, "Wan2.1-T2V-1.3B-Diffusers");

    private static readonly string Reference = Path.Combine(Env.VideoRoot, "wan21-reference", "reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
        => Directory.Exists(ModelPath) && File.Exists(Path.Combine(Reference, "index.json")) ? Env.Vulkan() : null;

    private static Dictionary<string, long> Dimensions(JsonElement index) => new()
    {
        [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 4 + 1,
        [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
        [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
    };

    [Fact]
    public void GraphsMatchDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "Wan 2.1 T2V-1.3B, its references or a Vulkan GPU missing");
        using var model = Env.Load(ModelPath, slot!);
        var index = Index();
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        foreach (var (prompt, name) in new[] { (index.GetProperty("prompt").GetString()!, "text_hidden"), (index.GetProperty("negative_prompt").GetString()!, "negative_hidden") })
        {
            var ids = tokenizer.Encode(prompt, addSpecial: true, parseSpecial: false);
            var tokens = new int[512];
            var mask = new float[512];
            ids.CopyTo(tokens);
            mask.AsSpan(0, ids.Length).Fill(1f);
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.Hidden, 0,
                (VideoDiffusionGraphs.Tokens, tokens), (VideoDiffusionGraphs.TokenMask, mask));
            hidden.AsSpan(ids.Length * (hidden.Length / 512)).Clear();
            var (maxAbs, rms) = Env.Compare(hidden, Tensor(name));
            Assert.True(rms < 5e-3, $"umT5 {name}: max {maxAbs}, relative RMS {rms}");
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, Dimensions(index),
            (VideoDiffusionGraphs.Latent, Tensor("noise")),
            (VideoDiffusionGraphs.Timestep, new[] { index.GetProperty("transformer_timestep").GetSingle() }),
            (VideoDiffusionGraphs.Context, Tensor("text_hidden")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor("transformer_out"));
        Assert.True(transformerRms < 5e-3, $"transformer: max {transformerMax}, relative RMS {transformerRms}");
    }

    [Fact]
    public async Task TextToVideoMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "Wan 2.1 T2V-1.3B, its references or a Vulkan GPU missing");
        using var model = Env.Load(ModelPath, slot!);
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var index = Index();
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
            InitialNoise = Tensor("noise"),
        };
        var stopwatch = Stopwatch.StartNew();
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        TestContext.Current.SendDiagnosticMessage($"Wan 2.1 {width}x{height}x{result.Frames.Count}: {stopwatch.Elapsed.TotalSeconds:F1} s");
        var expected = Tensor("video");
        int frameValues = 3 * width * height;
        double worst = double.MaxValue;
        for (int f = 0; f < result.Frames.Count; f++)
        {
            var reference = PlanarImage.FromPlanar(expected.AsSpan(f * frameValues, frameValues), width, height, 0f, 1f);
            worst = Math.Min(worst, Env.Psnr(result.Frames[f].Rgb, reference.Rgb));
        }

        Assert.True(worst > 25, $"worst frame PSNR {worst:F1} dB");
    }
}
