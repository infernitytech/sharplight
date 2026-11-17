using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// LTX-Video 13B 0.9.8 distilled as single files (QuantStack's transformer Q4_K_M GGUF and the stand-alone VAE file, the T5 encoder of an
/// LTX-Video diffusers folder as companion) on Vulkan against diffusers' condition pipeline on the same files
/// (tools/make-ltx-single-file-references.py: f32 on CUDA, 704×480, 25 frames, the 8 distilled timesteps without guidance, the same
/// initial noise): one transformer evaluation and the whole text-to-video run. Skips when the files, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class LtxSingleFileTests
{
    private static string Transformer => Path.Combine(Env.VideoRoot, "ltxv-13b-098-distilled-gguf", "LTXV-13B-0.9.8-distilled-Q4_K_M.gguf");

    private static string VaeFile => Path.Combine(Env.VideoRoot, "ltxv-13b-098-distilled-gguf", "vae", "LTXV-13B-0.9.8-distilled-VAE.safetensors");

    private static string T5Folder => Path.Combine(Env.VideoRoot, "LTX-Video-0.9.5");

    private static string Reference => Path.Combine(Env.VideoRoot, "ltx13b-reference", "reference");

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
        => File.Exists(Transformer) && File.Exists(VaeFile) && Directory.Exists(T5Folder) && File.Exists(Path.Combine(Reference, "index.json")) ? Env.Vulkan() : null;

    private static Tasks.TaskModel Load(Runtime.DeviceSlot slot)
        => Env.Load(Transformer, slot, new Dictionary<string, string> { [AuxiliaryRoles.TextEncoder] = T5Folder, [AuxiliaryRoles.Vae] = VaeFile });

    [Fact]
    public void TransformerMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "LTX-Video 13B files, the references or a Vulkan GPU missing");
        using var model = Load(slot!);
        var index = Index();
        var dims = new Dictionary<string, long>
        {
            [VideoDiffusionGraphs.FramesDimension] = (index.GetProperty("frames").GetInt32() - 1) / 8 + 1,
            [VideoDiffusionGraphs.Height] = index.GetProperty("height").GetInt32(),
            [VideoDiffusionGraphs.Width] = index.GetProperty("width").GetInt32(),
        };
        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, VideoDiffusionGraphs.Prediction, 0, dims,
            (VideoDiffusionGraphs.Latent, Tensor("noise")),
            (VideoDiffusionGraphs.Timestep, new[] { index.GetProperty("transformer_timestep").GetSingle() }),
            (VideoDiffusionGraphs.Context, Tensor("text_hidden")),
            (VideoDiffusionGraphs.ContextMask, Tensor("text_mask")));
        var (maxAbs, rms) = Env.Compare(prediction, Tensor("transformer_out"));
        TestContext.Current.SendDiagnosticMessage($"LTX 13B transformer: relative RMS {rms:E2}");
        Assert.True(rms < 3e-2, $"transformer: max {maxAbs}, relative RMS {rms}");
    }

    [Fact]
    public async Task VideoMatchesDiffusers()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "LTX-Video 13B files, the references or a Vulkan GPU missing");
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

        TestContext.Current.SendDiagnosticMessage($"LTX 13B {width}x{height}x{result.Frames.Count}: {seconds:F1} s (incl. compilation), worst frame {worst:F1} dB "
            + $"(frames: {string.Join(" ", perFrame)})");
        Assert.True(worst > 20, $"LTX 13B: worst frame PSNR {worst:F1} dB");
    }
}
