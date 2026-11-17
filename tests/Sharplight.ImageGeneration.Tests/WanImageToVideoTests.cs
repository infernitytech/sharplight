using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The image-conditioned Wan models as transformer-only GGUF files on Vulkan against diffusers running the same files
/// (tools/make-wan-i2v-references.py: f32 on CUDA, 832×480, 17 frames, 20 UniPC steps, the same initial latents): Wan 2.1 I2V-14B
/// (city96 Q4_K_M, the CLIP image encoder from the Wan 2.1 I2V configuration folder), Wan 2.1 FLF2V-14B (first and last frame) and
/// Wan 2.2 I2V-A14B (QuantStack's two Q4_K_M experts); umT5, the VAE and the tokenizer come from the Wan 2.1 T2V folder. Checks the image
/// encoder's states, the first conditional transformer evaluation and the whole run. Skips when a file, its references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class WanImageToVideoTests
{
    /// <summary>"2.1-i2v", "2.1-flf2v" and "2.2-i2v-a14b".</summary>
    public static TheoryData<string> Variants => ["2.1-i2v", "2.1-flf2v", "2.2-i2v-a14b"];

    private static string Companion => Path.Combine(Env.VideoRoot, "Wan2.1-T2V-1.3B-Diffusers");

    private static string ModelPath(string variant) => variant switch
    {
        "2.1-i2v" => Path.Combine(Env.VideoRoot, "wan21-i2v-gguf", "wan2.1-i2v-14b-480p-Q4_K_M.gguf"),
        "2.1-flf2v" => Path.Combine(Env.VideoRoot, "wan21-flf2v-gguf", "wan2.1-flf2v-14b-720p-Q4_K_M.gguf"),
        _ => Path.Combine(Env.VideoRoot, "wan22-i2v-a14b-gguf", "HighNoise", "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"),
    };

    private static string Reference(string variant) => Path.Combine(Env.VideoRoot, variant switch
    {
        "2.1-i2v" => "wan21-i2v-reference",
        "2.1-flf2v" => "wan21-flf2v-reference",
        _ => "wan22-i2v-a14b-reference",
    }, "reference");

    private static Dictionary<string, string> Auxiliary(string variant)
    {
        var auxiliary = new Dictionary<string, string> { [AuxiliaryRoles.TextEncoder] = Companion, [AuxiliaryRoles.Vae] = Companion };
        switch (variant)
        {
            case "2.1-i2v":
                auxiliary[AuxiliaryRoles.ImageEncoder] = Path.Combine(Env.VideoRoot, "Wan2.1-I2V-14B-480P-config");
                break;
            case "2.1-flf2v":
                auxiliary[AuxiliaryRoles.ImageEncoder] = Path.Combine(Env.VideoRoot, "Wan2.1-FLF2V-14B-720P-config");
                break;
            default:
                auxiliary[AuxiliaryRoles.SecondTransformer] = Path.Combine(Env.VideoRoot, "wan22-i2v-a14b-gguf", "LowNoise", "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf");
                break;
        }

        return auxiliary;
    }

    private static float[] Tensor(string variant, string name) => Env.ReadFloats(Path.Combine(Reference(variant), name + ".f32"));

    private static JsonElement Index(string variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference(variant), "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu(string variant)
        => File.Exists(ModelPath(variant)) && Directory.Exists(Companion) && File.Exists(Path.Combine(Reference(variant), "index.json")) ? Env.Vulkan() : null;

    private static ImageInput Image(string variant, string name) => PngCodec.Read(Path.Combine(Reference(variant), name));

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
        using var model = Env.Load(ModelPath(variant), slot!, Auxiliary(variant));
        var index = Index(variant);
        float[]? imageStates = null;
        if (model.Description.TryGetComponent(ModelComponentRoles.VisionEncoder, out var vision))
        {
            var processor = new ImageProcessor(ImageProcessorConfig.FromJson(vision.Config[ImageProcessor.ComponentKey]));
            var states = new List<float>();
            var images = index.TryGetProperty("last_image", out _) ? new[] { "start.png", "last.png" } : ["start.png"];
            foreach (string image in images)
            {
                states.AddRange(GraphRunner.Run(model, ModelComponentRoles.VisionEncoder, VideoDiffusionGraphs.ImageEncode, VideoDiffusionGraphs.ImageHidden, 0,
                    (VideoDiffusionGraphs.Image, processor.Process(Image(variant, image)).Patchify(vision.ConfigValue(DiffusionConfigKeys.PatchSize, 14)))));
            }

            imageStates = [.. states];
            // image_hidden_cpu: the states transformers computes on the CPU in f32, when the reference has them (on CUDA, TF32
            // convolutions alone move them by 1e-2).
            string expected = File.Exists(Path.Combine(Reference(variant), "image_hidden_cpu.f32")) ? "image_hidden_cpu" : "image_hidden";
            var (imageMax, imageRms) = Env.Compare(imageStates, Tensor(variant, expected));
            TestContext.Current.SendDiagnosticMessage($"Wan {variant} image encoder: relative RMS {imageRms:E2}");
            Assert.True(imageRms < 5e-3, $"image encoder: max {imageMax}, relative RMS {imageRms}");
            // The transformer comparison runs on the reference's image states, so it measures the transformer alone.
            imageStates = Tensor(variant, "image_hidden");
        }

        var dims = Dimensions(index);
        foreach (var (role, graph, name) in new[]
        {
            (ModelComponentRoles.Transformer, VideoDiffusionGraphs.Denoise, "transformer"),
            (VideoDiffusionGraphs.SecondTransformerRole, VideoDiffusionGraphs.Denoise2, "transformer_2"),
        })
        {
            if (!File.Exists(Path.Combine(Reference(variant), name + "_out.f32")))
            {
                continue;
            }

            var inputs = new List<(string, Array)>
            {
                (VideoDiffusionGraphs.Latent, Tensor(variant, name + "_latent")),
                (VideoDiffusionGraphs.Timestep, Tensor(variant, name + "_timestep")),
                (VideoDiffusionGraphs.Context, Tensor(variant, "text_hidden")),
            };
            if (imageStates is not null)
            {
                inputs.Add((VideoDiffusionGraphs.ImageContext, imageStates));
            }

            var prediction = GraphRunner.Run(model, role, graph, VideoDiffusionGraphs.Prediction, 0, dims, [.. inputs]);
            var (maxAbs, rms) = Env.Compare(prediction, Tensor(variant, name + "_out"));
            TestContext.Current.SendDiagnosticMessage($"Wan {variant} {name}: relative RMS {rms:E2}");
            Assert.True(rms < 3e-2, $"{name}: max {maxAbs}, relative RMS {rms}");
        }
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public async Task VideoMatchesDiffusers(string variant)
    {
        var slot = Gpu(variant);
        Assert.SkipWhen(slot is null, $"Wan {variant}, its references or a Vulkan GPU missing");
        using var model = Env.Load(ModelPath(variant), slot!, Auxiliary(variant));
        using var generator = Env.Clients.Create<IVideoGenerator>(model);
        var index = Index(variant);
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
        var request = new VideoGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative_prompt").GetString(),
            Image = Image(variant, "start.png"),
            LastImage = index.TryGetProperty("last_image", out _) ? Image(variant, "last.png") : null,
        };
        var options = new VideoGenerationOptions
        {
            Width = width,
            Height = height,
            Frames = index.GetProperty("frames").GetInt32(),
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            GuidanceScale2 = index.TryGetProperty("guidance_scale_2", out var second) ? second.GetSingle() : null,
            InitialNoise = Tensor(variant, "noise"),
        };
        var stopwatch = Stopwatch.StartNew();
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        double seconds = stopwatch.Elapsed.TotalSeconds;
        var expected = Tensor(variant, "video");
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

        TestContext.Current.SendDiagnosticMessage($"Wan {variant} {width}x{height}x{result.Frames.Count}: {seconds:F1} s (incl. compilation), worst frame {worst:F1} dB "
            + $"(frames: {string.Join(" ", perFrame)})");
        Assert.True(worst > 20, $"Wan {variant}: worst frame PSNR {worst:F1} dB");
    }
}
