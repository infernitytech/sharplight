using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Real Wan models (diffusers folders) on Vulkan against diffusers references (tools/make-wan-references.py: f32 on CUDA, 17 frames at
/// 832×480, 20 UniPC steps, guidance 5, the same initial latents): Wan 2.1 T2V-1.3B text-to-video (~/models/video/wan21-reference)
/// and Wan 2.2 TI2V-5B image-to-video (~/models/video/wan22-i2v-reference with its start.png): the umT5 conditioning, one transformer
/// evaluation and the whole run. Skips when a model, its references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class WanTests
{
    /// <summary>"2.1", "2.2" (diffusers folders) and "2.2-gguf" (QuantStack's Q8_0 transformer with the 2.2 folder as companion).</summary>
    public static TheoryData<string> Variants => ["2.1", "2.2", "2.2-gguf"];

    private static string FolderOf(string variant)
        => Path.Combine(Env.VideoRoot, variant == "2.1" ? "Wan2.1-T2V-1.3B-Diffusers" : "Wan2.2-TI2V-5B-Diffusers");

    private static string ModelPath(string variant)
        => variant == "2.2-gguf" ? Path.Combine(Env.VideoRoot, "wan22-gguf", "Wan2.2-TI2V-5B-Q8_0.gguf") : FolderOf(variant);

    private static TaskModel Load(string variant, Runtime.DeviceSlot slot)
        => variant == "2.2-gguf"
            ? Env.Load(ModelPath(variant), slot, new Dictionary<string, string> { [AuxiliaryRoles.TextEncoder] = FolderOf(variant), [AuxiliaryRoles.Vae] = FolderOf(variant) })
            : Env.Load(ModelPath(variant), slot);

    /// <summary>Q8_0 weights against the f32 reference: the file's quantization error, not the implementation's.</summary>
    private static double Tolerance(string variant) => variant == "2.2-gguf" ? 5e-2 : 5e-3;

    private static string Reference(string variant) => Path.Combine(Env.VideoRoot, variant == "2.1" ? "wan21-reference" : "wan22-i2v-reference", "reference");

    private static float[] Tensor(string variant, string name) => Env.ReadFloats(Path.Combine(Reference(variant), name + ".f32"));

    private static JsonElement Index(string variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference(variant), "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu(string variant)
        => (Directory.Exists(ModelPath(variant)) || File.Exists(ModelPath(variant))) && Directory.Exists(FolderOf(variant)) && File.Exists(Path.Combine(Reference(variant), "index.json")) ? Env.Vulkan() : null;

    private static Dictionary<string, long> Dimensions(JsonElement index) => new()
    {
        [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 4 + 1,
        [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
        [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
    };

    [Theory]
    [MemberData(nameof(Variants))]
    public void GraphsMatchDiffusers(string variant)
    {
        var slot = Gpu(variant);
        Assert.SkipWhen(slot is null, $"Wan {variant}, its references or a Vulkan GPU missing");
        using var model = Load(variant, slot!);
        var index = Index(variant);
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
            var (maxAbs, rms) = Env.Compare(hidden, Tensor(variant, name));
            TestContext.Current.SendDiagnosticMessage($"Wan {variant} umT5 {name}: relative RMS {rms:E2}");
            Assert.True(rms < 5e-3, $"umT5 {name}: max {maxAbs}, relative RMS {rms}");
        }

        var dims = Dimensions(index);
        int scale = model.Description.Component(ModelComponentRoles.Vae).ConfigValue(DiffusionConfigKeys.Downscale, 8) * 2;
        bool perToken = model.Description.Component(ModelComponentRoles.Transformer).ConfigValue(VideoConfigKeys.TokenTimesteps, false);
        long tokenCount = dims[VideoDiffusionGraphs.FramesDimension] * (dims[VideoDiffusionGraphs.Height] / scale) * (dims[VideoDiffusionGraphs.Width] / scale);
        var timesteps = new float[perToken ? tokenCount : 1];
        timesteps.AsSpan().Fill(index.GetProperty("transformer_timestep").GetSingle());
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor(variant, "noise")),
            (VideoDiffusionGraphs.Timestep, timesteps),
            (VideoDiffusionGraphs.Context, Tensor(variant, "text_hidden")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(variant, "transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"Wan {variant} transformer: relative RMS {transformerRms:E2}");
        Assert.True(transformerRms < Tolerance(variant), $"transformer: max {transformerMax}, relative RMS {transformerRms}");
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public async Task VideoMatchesDiffusers(string variant)
    {
        var slot = Gpu(variant);
        Assert.SkipWhen(slot is null, $"Wan {variant}, its references or a Vulkan GPU missing");
        using var model = Load(variant, slot!);
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var index = Index(variant);
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
        var request = new VideoGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative_prompt").GetString(),
            Image = index.GetProperty("image").GetString() is { } image ? PngCodec.Read(Path.Combine(Path.GetDirectoryName(Reference(variant))!, image)) : null,
        };
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

        TestContext.Current.SendDiagnosticMessage($"Wan {variant} {width}x{height}x{result.Frames.Count}: {seconds:F1} s (incl. compilation), worst frame {worst:F1} dB");
        Assert.True(worst > (variant == "2.2-gguf" ? 20 : 25), $"Wan {variant}: worst frame PSNR {worst:F1} dB");
    }
}
