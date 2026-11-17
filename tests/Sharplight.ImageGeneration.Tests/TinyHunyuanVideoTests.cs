using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.VideoGeneration;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random HunyuanVideo pipeline (tools/make-tiny-hunyuan-video.py): the templated Llama states, CLIP's pooled embedding, the
/// transformer and a whole 25-frame text-to-video run (temporal decode tiles) against diffusers, on the CPU and on Vulkan.
/// </summary>
public sealed class TinyHunyuanVideoTests
{
    private static readonly string Root = Path.Combine(Env.VideoRoot, "tiny-random-hunyuan-video");

    public static TheoryData<string> Devices => ["cpu", "vulkan"];

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Root, "reference", name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root, "reference", "index.json")));
        return document.RootElement.Clone();
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public async Task MatchesDiffusers(string device)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Root, "reference", "index.json")), "tiny-random-hunyuan-video missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Path.Combine(Root, "diffusers"), slot!);
        var index = Index();
        var dims = new Dictionary<string, long>
        {
            [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 4 + 1,
            [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
            [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
        };

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor("transformer_latent")),
            (VideoDiffusionGraphs.Timestep, new[] { 700f }),
            (VideoDiffusionGraphs.Context, Tensor("text_hidden")),
            (VideoDiffusionGraphs.ContextMask, Tensor("text_mask")),
            (DiffusionGraphs.Pooled, Tensor("pooled")),
            (DiffusionGraphs.Guidance, new[] { index.GetProperty("guidance_scale").GetSingle() }));
        var (maxAbs, rms) = Env.Compare(prediction, Tensor("transformer_out"));
        Assert.True(rms < 1e-4, $"transformer: max {maxAbs}, relative RMS {rms}");

        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var request = new VideoGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! };
        var options = new VideoGenerationOptions
        {
            Width = index.GetProperty("width").GetInt32(),
            Height = index.GetProperty("height").GetInt32(),
            Frames = index.GetProperty("frames").GetInt32(),
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            InitialNoise = Tensor("noise"),
        };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        int width = options.Width.Value;
        int height = options.Height.Value;
        Assert.Equal(options.Frames, result.Frames.Count);
        var expected = Tensor("video");
        int frameValues = 3 * width * height;
        for (int f = 0; f < result.Frames.Count; f++)
        {
            var reference = PlanarImage.FromPlanar(expected.AsSpan(f * frameValues, frameValues), width, height, 0f, 1f);
            double psnr = Env.Psnr(result.Frames[f].Rgb, reference.Rgb);
            Assert.True(psnr > 40, $"frame {f}: PSNR {psnr:F1} dB on {device}");
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void TextEncodersMatchTransformers(string device)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Root, "reference", "index.json")), "tiny-random-hunyuan-video missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Path.Combine(Root, "diffusers"), slot!);
        var index = Index();
        var encoder = model.Description.Component(ModelComponentRoles.TextEncoder);
        string template = encoder.Config[VideoConfigKeys.PromptTemplate].GetString()!;
        int prefix = encoder.ConfigValue(VideoConfigKeys.PromptPrefixTokens, 0);
        int length = encoder.ConfigValue(DiffusionConfigKeys.ContextLength, 256);
        var tokens = PromptTokenizer.Tokenize(model.CreateTokenizer(ModelComponentRoles.Tokenizer), prefix + length, index.GetProperty("prompt").GetString()!,
            out int count, template);
        Assert.Equal(prefix + index.GetProperty("text_tokens").GetInt32(), count);
        var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.Hidden, 0,
            (VideoDiffusionGraphs.Tokens, tokens));
        int width = hidden.Length / tokens.Length;
        int valid = count - prefix;
        var (maxAbs, rms) = Env.Compare(hidden.AsSpan(prefix * width, valid * width).ToArray(), Tensor("text_hidden").AsSpan(0, valid * width).ToArray());
        Assert.True(rms < 1e-4, $"Llama states: max {maxAbs}, relative RMS {rms}");

        var clipTokens = PromptTokenizer.Tokenize(model.CreateTokenizer(DiffusionGraphs.SecondTokenizerRole), 77, index.GetProperty("prompt").GetString()!, out _);
        var pooled = GraphRunner.Run(model, DiffusionGraphs.SecondTextEncoderRole, DiffusionGraphs.TextEncode2, DiffusionGraphs.Pooled, 0, (DiffusionGraphs.Tokens, clipTokens));
        var (pooledMax, pooledRms) = Env.Compare(pooled, Tensor("pooled"));
        Assert.True(pooledRms < 1e-4, $"CLIP pooled: max {pooledMax}, relative RMS {pooledRms}");
    }

    /// <summary>The transformer as an original-name single file (as GGUF / ComfyUI store it) with the diffusers folder as companion.</summary>
    [Theory]
    [MemberData(nameof(Devices))]
    public void TransformerOnlyFileWithCompanions(string device)
    {
        string single = Path.Combine(Root, "single", "hunyuan-video.safetensors");
        Assert.SkipUnless(File.Exists(single), "tiny-random-hunyuan-video single file missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        var error = Assert.Throws<NotSupportedException>(() => Env.Load(single, slot!));
        Assert.Contains("text-encoder", error.Message, StringComparison.Ordinal);
        string folder = Path.Combine(Root, "diffusers");
        using var model = Env.Load(single, slot!, new Dictionary<string, string> { [Vision.AuxiliaryRoles.TextEncoder] = folder, [Vision.AuxiliaryRoles.Vae] = folder });
        var index = Index();
        var dims = new Dictionary<string, long>
        {
            [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 4 + 1,
            [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
            [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
        };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor("transformer_latent")),
            (VideoDiffusionGraphs.Timestep, new[] { 700f }),
            (VideoDiffusionGraphs.Context, Tensor("text_hidden")),
            (VideoDiffusionGraphs.ContextMask, Tensor("text_mask")),
            (DiffusionGraphs.Pooled, Tensor("pooled")),
            (DiffusionGraphs.Guidance, new[] { index.GetProperty("guidance_scale").GetSingle() }));
        var (maxAbs, rms) = Env.Compare(prediction, Tensor("transformer_out"));
        Assert.True(rms < 1e-4, $"transformer: max {maxAbs}, relative RMS {rms}");
    }
}
