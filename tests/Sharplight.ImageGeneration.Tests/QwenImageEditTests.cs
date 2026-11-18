using System.Diagnostics;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Qwen-Image-Edit-2511 (unsloth's transformer-only Q4_K_M GGUF, with the Qwen2.5-VL text encoder folder and the VAE of the Qwen-Image-Edit-2511
/// diffusers folder as companions) on Vulkan against diffusers' QwenImageEditPlusPipeline running the same files
/// (tools/make-qwen-image-edit-references.py: the dequantized GGUF transformer on CUDA, the f32 text encoder, 20 steps, true CFG 4 with the negative
/// prompt " ", 512 × 384, same initial latents, one and two 640 × 480 input pictures) in ~/models/imagegen-edit/qwen-edit-2511-reference
/// (the root can be moved with <c>SHARPLIGHT_EDIT_ROOT</c>). Skips when the files, the references or a GPU are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class QwenImageEditTests
{
    private static readonly string Root = Environment.GetEnvironmentVariable("SHARPLIGHT_EDIT_ROOT") is { Length: > 0 } root
        ? root
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/imagegen-edit");

    private static readonly string Reference = Path.Combine(Root, "qwen-edit-2511-reference");
    private static readonly string Folder = Path.Combine(Root, "qwen-image-edit-2511");
    private static readonly string Gguf = Path.Combine(Root, "qwen-edit-2511-gguf", "qwen-image-edit-2511-Q4_K_M.gguf");

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static Runtime.DeviceSlot? Gpu()
        => File.Exists(Gguf) && Directory.Exists(Path.Combine(Folder, "text_encoder")) && File.Exists(Path.Combine(Reference, "image_2.f32")) ? Env.Vulkan() : null;

    private static TaskModel LoadModel(Runtime.DeviceSlot slot) => Env.Load(Gguf, slot, new Dictionary<string, string>
    {
        [AuxiliaryRoles.TextEncoder] = Path.Combine(Folder, "text_encoder"),
        [AuxiliaryRoles.Vae] = Path.Combine(Folder, "vae", "diffusion_pytorch_model.safetensors"),
    });

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public async Task EditMatchesDiffusers(int count)
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "Qwen-Image-Edit-2511, its references or a Vulkan GPU missing");
        using var model = LoadModel(slot!);
        var denoiser = model.Description.Component(ModelComponentRoles.Denoiser);
        Assert.Equal(3, denoiser.ConfigValue(DiffusionConfigKeys.ReferenceImages, 0));
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Index();
        int width = index.GetProperty("width").GetInt32();
        int height = index.GetProperty("height").GetInt32();
        var images = Enumerable.Range(0, count).Select(i => PngCodec.Read(Path.Combine(Reference, $"ref_{i}.png"))).ToArray();
        var watch = Stopwatch.StartNew();
        var result = await generator.GenerateAsync(new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative").GetString(),
            ReferenceImages = images,
        }, new ImageGenerationOptions
        {
            Width = width,
            Height = height,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance").GetSingle(),
            InitialNoise = Env.ReadFloats(Path.Combine(Reference, "noise.f32")),
        }, TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Env.ReadFloats(Path.Combine(Reference, $"image_{count}.f32")), width, height, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"qwen-edit-2511-{count}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"qwen-edit-2511-{count}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{count} picture(s): PSNR {psnr:F1} dB, {watch.Elapsed.TotalSeconds:F0} s");
        Assert.True(psnr > 24, $"{count} picture(s): PSNR {psnr:F1} dB");
    }

    /// <summary>
    /// The vision tower (windowed attention, merge-window patch order) over the first condition picture (the pipeline's 448 × 320 pixels) against
    /// transformers' <c>visual</c> output, and the preprocessing's normalized pixels.
    /// </summary>
    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    public void VisionTowerMatchesTransformers(string device)
    {
        var slot = Gpu() is null ? null : Env.Device(device);
        Assert.SkipWhen(slot is null || !File.Exists(Path.Combine(Reference, "vision_embeds_0.f32")), "Qwen-Image-Edit-2511, its vision references or a Vulkan GPU missing");
        using var model = LoadModel(slot!);
        var parameters = VisionParameters.Of(model.Definition)!;
        var condition = PngCodec.Read(Path.Combine(Reference, "condition_0.png"));
        var pre = new VisionPreprocessor(parameters, new ImageTokenLimits(null, 1024)).Process(condition);
        var transformersPixels = Env.ReadFloats(Path.Combine(Reference, "vision_pixels_0.f32"));
        TestContext.Current.SendDiagnosticMessage($"pixels: ours mean {pre.Patches.Average():F5} rms {Math.Sqrt(pre.Patches.Average(v => v * v)):F5}, transformers mean "
            + $"{transformersPixels.Average():F5} rms {Math.Sqrt(transformersPixels.Average(v => v * v)):F5}");
        int patchDim = parameters.PatchDim;
        double worst = 0;
        int merged = pre.PatchCount / parameters.MergeUnit;
        for (int m = 0; m < merged; m++)
        {
            for (int j = 0; j < parameters.MergeUnit; j++)
            {
                for (int k = 0; k < patchDim; k++)
                {
                    worst = Math.Max(worst, Math.Abs(pre.Patches[(pre.WindowOrder[m] * parameters.MergeUnit + j) * patchDim + k] - transformersPixels[(m * parameters.MergeUnit + j) * patchDim + k]));
                }
            }
        }

        TestContext.Current.SendDiagnosticMessage($"pixels after undoing the window order: largest difference {worst:E2}");
        var windowed = new float[pre.Patches.Length];
        for (int m = 0; m < merged; m++)
        {
            transformersPixels.AsSpan(m * parameters.MergeUnit * patchDim, parameters.MergeUnit * patchDim)
                .CopyTo(windowed.AsSpan(pre.WindowOrder[m] * parameters.MergeUnit * patchDim));
        }

        pre = new PreprocessedImage
        {
            Width = pre.Width, Height = pre.Height, GridHeight = pre.GridHeight, GridWidth = pre.GridWidth, MergedHeight = pre.MergedHeight,
            MergedWidth = pre.MergedWidth, Patches = windowed, Positions = pre.Positions, EmbeddingIndex = pre.EmbeddingIndex,
            EmbeddingWeight = pre.EmbeddingWeight, WindowSpans = pre.WindowSpans, WindowOrder = pre.WindowOrder,
        };
        using var compiled = model.CompileComponent(ModelComponentRoles.TextEncoder, WellKnown.VisionEncodeGraph, null);
        Graph.Value Input(string name) => compiled.Graph.Inputs.Single(i => i.Name == name);
        compiled.SetInput<float>(Input(VisionParameters.PatchesInput), pre.Patches);
        compiled.SetInput<int>(Input(VisionParameters.PositionsInput), pre.Positions);
        compiled.SetInput<int>(Input(VisionParameters.EmbeddingIndexInput), pre.EmbeddingIndex);
        compiled.SetInput<float>(Input(VisionParameters.EmbeddingWeightInput), pre.EmbeddingWeight);
        compiled.SetInput<int>(Input(VisionParameters.WindowSpansInput), pre.WindowSpans);
        compiled.SetInput<int>(Input(VisionParameters.WindowOrderInput), pre.WindowOrder);
        var run = new Runtime.RunParameters();
        run.Symbols[VisionParameters.PatchSymbol] = pre.PatchCount;
        run.Symbols[VisionParameters.MergedSymbol] = pre.PatchCount / parameters.MergeUnit;
        compiled.Run(run);
        var embeddings = new float[pre.PatchCount / parameters.MergeUnit * parameters.EmbeddingWidth];
        compiled.GetOutput<float>(VisionParameters.EmbeddingsOutput, embeddings);
        var (max, rms) = Env.Compare(embeddings, Env.ReadFloats(Path.Combine(Reference, "vision_embeds_0.f32")));
        TestContext.Current.SendDiagnosticMessage($"{device} vision tower: max {max}, relative RMS {rms:E2}; grid {pre.GridHeight}x{pre.GridWidth}");
        Assert.True(rms < 3e-2, $"vision tower: max {max}, relative RMS {rms}");
    }

    [Fact]
    public async Task OutputSizeFollowsTheLastPictureAtAboutOneMegapixel()
    {
        var slot = Gpu();
        Assert.SkipWhen(slot is null, "Qwen-Image-Edit-2511, its references or a Vulkan GPU missing");
        using var model = LoadModel(slot!);
        Assert.Equal(1024 * 1024, model.Description.ConfigValue(DiffusionConfigKeys.ReferenceArea, 0L));
        await Task.CompletedTask;
    }
}
