using System.Text.Json;
using Sharplight.Models;
using Sharplight.Vision;
using Sharplight.Imaging;
using Sharplight.Tasks;
using Sharplight.VideoGeneration;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random Wan pipelines (tools/make-tiny-wan.py; "t2v" in the Wan 2.1 layout, "ti2v" in the Wan 2.2 TI2V layout with the
/// residual VAE and per-token timesteps): the umT5 conditioning, the transformer, the streamed causal VAE decoder, the image encoder and
/// whole text-to-video (UniPC and Euler) and image-to-video runs against diffusers, on the CPU and on Vulkan.
/// </summary>
public sealed class TinyWanTests
{
    private static string Root(string variant) => Path.Combine(Env.TinyWan, variant);

    private static float[] Tensor(string variant, string name) => Env.ReadFloats(Path.Combine(Root(variant), "reference", name + ".f32"));

    private static JsonElement Index(string variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root(variant), "reference", "index.json")));
        return document.RootElement.Clone();
    }

    private static bool Present(string variant) => File.Exists(Path.Combine(Root(variant), "reference", "index.json"));

    public static TheoryData<string, string> Cases()
    {
        var data = new TheoryData<string, string>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string variant in new[] { "t2v", "ti2v" })
            {
                data.Add(device, variant);
            }
        }

        return data;
    }

    public static TheoryData<string, string, string> PipelineCases()
    {
        var data = new TheoryData<string, string, string>();
        foreach (var row in Cases())
        {
            foreach (string sampler in new[] { Samplers.UniPc, Samplers.Euler })
            {
                data.Add(row.Data.Item1, row.Data.Item2, sampler);
            }
        }

        return data;
    }

    private static Dictionary<string, long> Dimensions(JsonElement index) => new()
    {
        [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 4 + 1,
        [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
        [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void GraphsMatchDiffusers(string device, string variant)
    {
        Assert.SkipUnless(Present(variant), "tiny-random-wan missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Path.Combine(Root(variant), "diffusers"), slot!);
        var index = Index(variant);
        var dims = Dimensions(index);

        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        int length = index.GetProperty("max_sequence_length").GetInt32();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] expected = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            Assert.Equal(expected, tokenizer.Encode(prompt.Name, addSpecial: true, parseSpecial: false));
            var tokens = new int[length];
            var mask = new float[length];
            expected.CopyTo(tokens, 0);
            mask.AsSpan(0, expected.Length).Fill(1f);
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.Hidden, 0,
                (VideoDiffusionGraphs.Tokens, tokens), (VideoDiffusionGraphs.TokenMask, mask));
            hidden.AsSpan(expected.Length * (hidden.Length / length)).Clear();
            var (maxAbs, rms) = Env.Compare(hidden, Tensor(variant, $"text_hidden_{i}"));
            Assert.True(rms < 1e-4, $"umT5 {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor(variant, "transformer_latent")),
            (VideoDiffusionGraphs.Timestep, Tensor(variant, "transformer_timestep")),
            (VideoDiffusionGraphs.Context, Tensor(variant, "transformer_context")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(variant, "transformer_out"));
        Assert.True(transformerRms < 1e-4, $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        var decoded = DecodeVideo(model, dims, Tensor(variant, "vae_latent"), index);
        var expectedVideo = Tensor(variant, "vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray();
        var (vaeMax, vaeRms) = Env.Compare(decoded, FramesFirst(expectedVideo, 3, index.GetProperty("frames").GetInt32()));
        Assert.True(vaeRms < 1e-4, $"vae decode: max {vaeMax}, relative RMS {vaeRms}");

        var encoded = GraphRunner.Run(model, ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeEncode, VideoDiffusionGraphs.Latent, 0, dims,
            (VideoDiffusionGraphs.Image, Tensor(variant, "vae_image").Select(v => v / 2 + 0.5f).ToArray()));
        var (encodeMax, encodeRms) = Env.Compare(encoded, Tensor(variant, "vae_encoded_mean"));
        Assert.True(encodeRms < 1e-4, $"vae encode: max {encodeMax}, relative RMS {encodeRms}");
    }

    [Theory]
    [MemberData(nameof(PipelineCases))]
    public async Task TextToVideoMatchesDiffusers(string device, string variant, string sampler)
    {
        Assert.SkipUnless(Present(variant), "tiny-random-wan missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Path.Combine(Root(variant), "diffusers"), slot!);
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var index = Index(variant);
        var request = new VideoGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, NegativePrompt = index.GetProperty("negative_prompt").GetString() };
        var result = await generator.GenerateAsync(request, Options(index, variant, sampler), TestContext.Current.CancellationToken);
        AssertVideo(result, Tensor(variant, $"video_{sampler}"), index, $"{variant} {sampler} on {device}");
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    public async Task ImageToVideoMatchesDiffusers(string device)
    {
        Assert.SkipUnless(Present("ti2v"), "tiny-random-wan missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Path.Combine(Root("ti2v"), "diffusers"), slot!);
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var defaults = (VideoGenerationDefaults)generator.GetService(typeof(VideoGenerationDefaults))!;
        Assert.True(defaults.SupportsImage);
        var index = Index("ti2v");
        var image = PngCodec.Decode(File.ReadAllBytes(Path.Combine(Root("ti2v"), "reference", "input.png")));
        var request = new VideoGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative_prompt").GetString(),
            Image = image,
        };
        var result = await generator.GenerateAsync(request, Options(index, "ti2v", Samplers.UniPc), TestContext.Current.CancellationToken);
        AssertVideo(result, Tensor("ti2v", "video_i2v"), index, $"ti2v image-to-video on {device}");
    }

    public static TheoryData<string, string, string> SingleFileCases()
    {
        var data = new TheoryData<string, string, string>();
        foreach (var row in Cases())
        {
            foreach (string file in new[] { "wan.safetensors", "wan-F16.gguf" })
            {
                data.Add(row.Data.Item1, row.Data.Item2, file);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(SingleFileCases))]
    public async Task TransformerOnlyFilesWithCompanions(string device, string variant, string file)
    {
        Assert.SkipUnless(Present(variant) && File.Exists(Path.Combine(Root(variant), "single", file)), "tiny-random-wan single files missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        string diffusers = Path.Combine(Root(variant), "diffusers");
        using var model = Env.Load(Path.Combine(Root(variant), "single", file), slot!, new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = diffusers,
            [AuxiliaryRoles.Vae] = diffusers,
        });
        var index = Index(variant);
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, Dimensions(index),
            (VideoDiffusionGraphs.Latent, Tensor(variant, "transformer_latent")),
            (VideoDiffusionGraphs.Timestep, Tensor(variant, "transformer_timestep")),
            (VideoDiffusionGraphs.Context, Tensor(variant, "transformer_context")));
        var (maxAbs, rms) = Env.Compare(prediction, Tensor(variant, "transformer_out"));
        // The GGUF stores the weights in f16.
        Assert.True(rms < (file.EndsWith(".gguf", StringComparison.Ordinal) ? 3e-3 : 1e-4), $"transformer from {file}: max {maxAbs}, relative RMS {rms}");
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var request = new VideoGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, NegativePrompt = index.GetProperty("negative_prompt").GetString() };
        var result = await generator.GenerateAsync(request, Options(index, variant, Samplers.UniPc), TestContext.Current.CancellationToken);
        if (file.EndsWith(".safetensors", StringComparison.Ordinal))
        {
            AssertVideo(result, Tensor(variant, "video_unipc"), index, $"{variant} from {file} on {device}");
        }
        else
        {
            Assert.Equal(index.GetProperty("frames").GetInt32(), result.Frames.Count);
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public async Task TwoExpertsMatchDiffusers(string device)
    {
        Assert.SkipUnless(Present("a14b"), "tiny-random-wan a14b missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Path.Combine(Root("a14b"), "diffusers"), slot!);
        var index = Index("a14b");
        var dims = Dimensions(index);
        foreach (var (role, graph, reference) in new[]
        {
            (ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, "transformer_out"),
            (VideoDiffusionGraphs.SecondTransformerRole, VideoDiffusionGraphs.Denoise2, "transformer_2_out"),
        })
        {
            var prediction = GraphRunner.Run(model, role, graph, VideoDiffusionGraphs.Prediction, 0, dims,
                (VideoDiffusionGraphs.Latent, Tensor("a14b", "transformer_latent")),
                (VideoDiffusionGraphs.Timestep, Tensor("a14b", "transformer_timestep")),
                (VideoDiffusionGraphs.Context, Tensor("a14b", "transformer_context")));
            var (maxAbs, rms) = Env.Compare(prediction, Tensor("a14b", reference));
            Assert.True(rms < 1e-4, $"{graph}: max {maxAbs}, relative RMS {rms}");
        }

        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var request = new VideoGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, NegativePrompt = index.GetProperty("negative_prompt").GetString() };
        var options = Options(index, "a14b", Samplers.UniPc) with { GuidanceScale2 = index.GetProperty("guidance_scale_2").GetSingle() };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        AssertVideo(result, Tensor("a14b", "video_unipc"), index, $"two experts on {device}");
    }

    public static TheoryData<string> Devices => ["cpu", "vulkan"];

    [Fact]
    public void TransformerOnlyFileWithoutCompanionsNamesWhatIsMissing()
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Root("t2v"), "single", "wan.safetensors")), "tiny-random-wan single files missing");
        var error = Assert.Throws<NotSupportedException>(() => Env.Load(Path.Combine(Root("t2v"), "single", "wan.safetensors"), Env.Cpu()));
        Assert.Contains("text-encoder", error.Message, StringComparison.Ordinal);
        Assert.Contains("vae", error.Message, StringComparison.Ordinal);
    }

    private static VideoGenerationOptions Options(JsonElement index, string variant, string sampler) => new()
    {
        Width = index.GetProperty("width").GetInt32(),
        Height = index.GetProperty("height").GetInt32(),
        Frames = index.GetProperty("frames").GetInt32(),
        Steps = index.GetProperty("steps").GetInt32(),
        GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
        Scheduler = sampler,
        InitialNoise = Tensor(variant, "noise"),
    };

    private static void AssertVideo(VideoGenerationResult result, float[] expected, JsonElement index, string what)
    {
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
        int frames = index.GetProperty("frames").GetInt32();
        Assert.Equal(frames, result.Frames.Count);
        int frameValues = 3 * width * height;
        for (int f = 0; f < frames; f++)
        {
            var reference = PlanarImage.FromPlanar(expected.AsSpan(f * frameValues, frameValues), width, height, 0f, 1f);
            double psnr = Env.Psnr(result.Frames[f].Rgb, reference.Rgb);
            Assert.True(psnr > 40, $"{what}: frame {f} PSNR {psnr:F1} dB");
        }
    }

    /// <summary>Decodes a latent video [C, T, h, w] latent frame by latent frame; returns frames [F, 3, H, W].</summary>
    private static float[] DecodeVideo(TaskModel model, IReadOnlyDictionary<string, long> dims, float[] latents, JsonElement index)
    {
        int latentFrames = (int)dims[VideoDiffusionGraphs.FramesDimension];
        int channels = latents.Length / latentFrames / PlaneOf(model, index);
        using var first = model.CompileComponent(ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeDecodeFirst, null, dims);
        using var next = model.CompileComponent(ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeDecodeNext, null, dims);
        var stream = new CausalDecoderStream();
        var all = new List<float>();
        int plane = PlaneOf(model, index);
        for (int t = 0; t < latentFrames; t++)
        {
            var frame = new float[channels * plane];
            for (int c = 0; c < channels; c++)
            {
                latents.AsSpan((c * latentFrames + t) * plane, plane).CopyTo(frame.AsSpan(c * plane));
            }

            all.AddRange(stream.Decode(t == 0 ? first : next, frame).Pixels);
        }

        return [.. all];
    }

    private static int PlaneOf(TaskModel model, JsonElement index)
    {
        int scale = model.Description.Component(ModelComponentRoles.Vae).ConfigValue(DiffusionConfigKeys.Downscale, 8);
        return index.GetProperty("height").GetInt32() / scale * (index.GetProperty("width").GetInt32() / scale);
    }

    /// <summary>[C, F, H, W] → [F, C, H, W].</summary>
    private static float[] FramesFirst(float[] video, int channels, int frames)
    {
        int plane = video.Length / channels / frames;
        var result = new float[video.Length];
        for (int c = 0; c < channels; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                video.AsSpan((c * frames + f) * plane, plane).CopyTo(result.AsSpan((f * channels + c) * plane));
            }
        }

        return result;
    }
}
