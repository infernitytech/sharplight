using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// HunyuanVideo text-to-video as city96's transformer GGUF with the HunyuanVideo diffusers folder as companion, on Vulkan, against
/// diffusers running the same GGUF (tools/make-hunyuan-video-references.py: f32, the Llama encoder on the CPU, 832×480, 33 frames,
/// 20 Euler steps, embedded guidance 6, the same initial latents): the Llama states, CLIP's pooled embedding, the first transformer
/// evaluation and the whole run. Skips when the files, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class HunyuanVideoTests
{
    private static string ModelPath => Path.Combine(Env.VideoRoot, "hunyuan-video-gguf", "hunyuan-video-t2v-720p-Q4_K_M.gguf");

    private static string Folder => Path.Combine(Env.VideoRoot, "HunyuanVideo");

    private static string Reference => Path.Combine(Env.VideoRoot, "hunyuan-video-reference", "reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
        => File.Exists(ModelPath) && Directory.Exists(Folder) && File.Exists(Path.Combine(Reference, "index.json")) ? Env.Vulkan() : null;

    private static TaskModel Load(Runtime.DeviceSlot slot)
        => Env.Load(ModelPath, slot, new Dictionary<string, string> { [AuxiliaryRoles.TextEncoder] = Folder, [AuxiliaryRoles.Vae] = Folder });

    [Fact]
    public void GraphsMatchDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "HunyuanVideo, its references or a Vulkan GPU missing");
        using var model = Load(slot!);
        var index = Index();
        var encoder = model.Description.Component(ModelComponentRoles.TextEncoder);
        int prefix = encoder.ConfigValue(VideoConfigKeys.PromptPrefixTokens, 0);
        int length = encoder.ConfigValue(DiffusionConfigKeys.ContextLength, 256);
        var tokens = PromptTokenizer.Tokenize(model.CreateTokenizer(ModelComponentRoles.Tokenizer), prefix + length, index.GetProperty("prompt").GetString()!,
            out int count, encoder.Config[VideoConfigKeys.PromptTemplate].GetString());
        var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.Hidden, 0, (VideoDiffusionGraphs.Tokens, tokens));
        int width = hidden.Length / tokens.Length;
        int valid = count - prefix;
        var (textMax, textRms) = Env.Compare(hidden.AsSpan(prefix * width, valid * width).ToArray(), Tensor("text_hidden").AsSpan(0, valid * width).ToArray());
        TestContext.Current.SendDiagnosticMessage($"HunyuanVideo Llama states ({valid} tokens): relative RMS {textRms:E2}");
        Assert.True(textRms < 5e-3, $"Llama states: max {textMax}, relative RMS {textRms}");

        var clipTokens = PromptTokenizer.Tokenize(model.CreateTokenizer(DiffusionGraphs.SecondTokenizerRole), 77, index.GetProperty("prompt").GetString()!, out _);
        var pooled = GraphRunner.Run(model, DiffusionGraphs.SecondTextEncoderRole, DiffusionGraphs.TextEncode2, DiffusionGraphs.Pooled, 0, (DiffusionGraphs.Tokens, clipTokens));
        var (pooledMax, pooledRms) = Env.Compare(pooled, Tensor("pooled"));
        TestContext.Current.SendDiagnosticMessage($"HunyuanVideo CLIP pooled: relative RMS {pooledRms:E2}");
        Assert.True(pooledRms < 5e-3, $"CLIP pooled: max {pooledMax}, relative RMS {pooledRms}");

        var dims = new Dictionary<string, long>
        {
            [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 4 + 1,
            [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
            [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
        };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor("transformer_latent")),
            (VideoDiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (VideoDiffusionGraphs.Context, Tensor("text_hidden")),
            (VideoDiffusionGraphs.ContextMask, Tensor("text_mask")),
            (DiffusionGraphs.Pooled, Tensor("pooled")),
            (DiffusionGraphs.Guidance, new[] { index.GetProperty("guidance_scale").GetSingle() }));
        var (maxAbs, rms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"HunyuanVideo transformer: relative RMS {rms:E2}");
        Assert.True(rms < 3e-2, $"transformer: max {maxAbs}, relative RMS {rms}");
    }

    [Fact]
    public async Task VideoMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "HunyuanVideo, its references or a Vulkan GPU missing");
        using var model = Load(slot!);
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var index = Index();
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
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
        var result = await generator.GenerateAsync(new VideoGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, options,
            TestContext.Current.CancellationToken);
        double seconds = stopwatch.Elapsed.TotalSeconds;
        var expected = Tensor("video");
        int frameValues = 3 * width * height;
        double worst = double.MaxValue;
        var perFrame = new List<string>();
        for (int f = 0; f < result.Frames.Count; f++)
        {
            var reference = PlanarImage.FromPlanar(expected.AsSpan(f * frameValues, frameValues), width, height, 0f, 1f);
            double psnr = Env.Psnr(result.Frames[f].Rgb, reference.Rgb);
            perFrame.Add($"{psnr:F1}");
            worst = Math.Min(worst, psnr);
        }

        TestContext.Current.SendDiagnosticMessage($"HunyuanVideo {width}x{height}x{result.Frames.Count}: {seconds:F1} s (incl. compilation), worst frame {worst:F1} dB "
            + $"(frames: {string.Join(" ", perFrame)})");
        Assert.True(worst > 20, $"HunyuanVideo: worst frame PSNR {worst:F1} dB");
    }
}
