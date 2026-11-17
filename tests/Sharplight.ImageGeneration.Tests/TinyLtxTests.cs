using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.VideoGeneration;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random LTX-Video pipeline (tools/make-tiny-ltx.py): the unmasked T5 conditioning, the transformer with its key-masked
/// cross-attention and fractional RoPE, the whole-video non-causal VAE decoder and a whole text-to-video run (flow-matching Euler with
/// the resolution-dependent shift and the terminal stretch) against diffusers, on the CPU and on Vulkan; variant "095" is the
/// LTX-Video 0.9.5 layout (timestep-conditioned decoder, residual upsamplers that halve the channels, static shift).
/// </summary>
public sealed class TinyLtxTests
{
    private static string Root(string variant) => Path.Combine(Env.VideoRoot, "tiny-random-ltx", variant);

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
            data.Add(device, "");
            data.Add(device, "095");
        }

        return data;
    }

    private static Dictionary<string, long> Dimensions(JsonElement index) => new()
    {
        [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 8 + 1,
        [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
        [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void GraphsMatchDiffusers(string device, string variant)
    {
        Assert.SkipUnless(Present(variant), $"tiny-random-ltx {variant} missing");
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
            expected.CopyTo(tokens, 0);
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, VideoDiffusionGraphs.TextEncode, VideoDiffusionGraphs.Hidden, 0,
                (VideoDiffusionGraphs.Tokens, tokens));
            var (maxAbs, rms) = Env.Compare(hidden, Tensor(variant, $"text_hidden_{i}"));
            Assert.True(rms < 1e-4, $"T5 {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor(variant, "transformer_latent")),
            (VideoDiffusionGraphs.Timestep, Tensor(variant, "transformer_timestep")),
            (VideoDiffusionGraphs.Context, Tensor(variant, "transformer_context")),
            (VideoDiffusionGraphs.ContextMask, Tensor(variant, "transformer_context_mask")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(variant, "transformer_out"));
        Assert.True(transformerRms < 1e-4, $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, VideoDiffusionGraphs.VaeDecode, VideoDiffusionGraphs.Frames, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor(variant, "vae_latent")));
        int frames = index.GetProperty("frames").GetInt32();
        var expectedVideo = FramesFirst(Tensor(variant, "vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray(), 3, frames);
        var (vaeMax, vaeRms) = Env.Compare(decoded, expectedVideo);
        Assert.True(vaeRms < 1e-4, $"vae decode: max {vaeMax}, relative RMS {vaeRms}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task TextToVideoMatchesDiffusers(string device, string variant)
    {
        Assert.SkipUnless(Present(variant), $"tiny-random-ltx {variant} missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Path.Combine(Root(variant), "diffusers"), slot!);
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var index = Index(variant);
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
        int frames = index.GetProperty("frames").GetInt32();
        var request = new VideoGenerationRequest { Prompt = index.GetProperty("prompt").GetString()!, NegativePrompt = index.GetProperty("negative_prompt").GetString() };
        var options = new VideoGenerationOptions
        {
            Width = width,
            Height = height,
            Frames = frames,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            InitialNoise = Tensor(variant, "noise"),
        };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        Assert.Equal(frames, result.Frames.Count);
        var expected = Tensor(variant, "video");
        int frameValues = 3 * width * height;
        for (int f = 0; f < frames; f++)
        {
            var reference = PlanarImage.FromPlanar(expected.AsSpan(f * frameValues, frameValues), width, height, 0f, 1f);
            double psnr = Env.Psnr(result.Frames[f].Rgb, reference.Rgb);
            Assert.True(psnr > 40, $"LTX on {device}: frame {f} PSNR {psnr:F1} dB");
        }
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
