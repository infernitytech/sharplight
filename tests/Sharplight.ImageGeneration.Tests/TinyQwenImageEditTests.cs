using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random Qwen-Image-Edit-Plus pipelines (tools/make-tiny-qwen-image-edit.py: "edit-2509", and "edit-2511" with zero_cond_t) against
/// diffusers' QwenImageEditPlusPipeline: the dual-stream transformer with one and two reference images (their RoPE frames, the reference tokens
/// modulated at timestep 0 in -2511) and whole edits (the Qwen2.5-VL text encoder reading the images through its windowed vision tower with
/// M-RoPE positions, the reference latents, norm-rescaled true guidance), on the CPU and on Vulkan, from the diffusers folder and from the
/// transformer-only Q8_0 GGUF (the -2511 file with its zero-timestep marker tensor) with the folder's text encoder and VAE as companions.
/// </summary>
public sealed class TinyQwenImageEditTests
{
    private static string Root(string variant) => Path.Combine(Env.TinyQwenImageEdit, variant);

    private static float[] Tensor(string variant, string name) => Env.ReadFloats(Path.Combine(Root(variant), "reference", name + ".f32"));

    private static JsonElement Index(string variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root(variant), "reference", "index.json")));
        return document.RootElement.Clone();
    }

    private static bool Present(string variant) => File.Exists(Path.Combine(Root(variant), "reference", "index.json"));

    private static TaskModel Load(string variant, string source, DeviceSlot slot)
    {
        string diffusers = Path.Combine(Root(variant), "diffusers");
        if (source == "diffusers")
        {
            return Env.Load(diffusers, slot);
        }

        return Env.Load(Path.Combine(Root(variant), "single", $"qwen-image-{variant}-Q8_0.gguf"), slot, new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = Path.Combine(diffusers, "text_encoder"),
            [AuxiliaryRoles.Vae] = Path.Combine(diffusers, "vae", "diffusion_pytorch_model.safetensors"),
        });
    }

    public static TheoryData<string, string, string> Cases()
    {
        var data = new TheoryData<string, string, string>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string variant in new[] { "edit-2509", "edit-2511" })
            {
                foreach (string source in new[] { "diffusers", "gguf" })
                {
                    data.Add(device, variant, source);
                }
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void TransformerWithReferencesMatchesDiffusers(string device, string variant, string source)
    {
        Assert.SkipUnless(Present(variant), "tiny-random-qwen-image-edit missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(variant, source, slot!);
        var denoiser = model.Description.Component(ModelComponentRoles.Denoiser);
        Assert.Equal(3, denoiser.ConfigValue(DiffusionConfigKeys.ReferenceImages, 0));
        var index = Index(variant);
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var references = Tensor(variant, "transformer_refs");
        int perReference = references.Length / 2;
        foreach (int n in new[] { 1, 2 })
        {
            var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width, [DiffusionGraphs.References] = n };
            var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
                (DiffusionGraphs.Latent, Tensor(variant, "transformer_latent")),
                (DiffusionGraphs.ReferenceLatent, references.AsSpan(0, n * perReference).ToArray()),
                (DiffusionGraphs.Timestep, Tensor(variant, "transformer_timestep")),
                (DiffusionGraphs.Context, Tensor(variant, "transformer_context")),
                (DiffusionGraphs.ContextMask, Tensor(variant, "transformer_context_mask")));
            var (max, rms) = Env.Compare(prediction, Tensor(variant, $"transformer_out_{n}"));
            double tolerance = source == "gguf" ? 3e-2 : (device == "cpu" ? 1e-4 : 1e-3);
            Assert.True(rms < tolerance, $"{n} reference(s): max {max}, relative RMS {rms}");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task EditMatchesDiffusers(string device, string variant, string source)
    {
        Assert.SkipUnless(Present(variant), "tiny-random-qwen-image-edit missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(variant, source, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Index(variant);
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var defaults = generator.Defaults;
        Assert.Equal(3, defaults.ReferenceImages);
        var images = new[] { PngCodec.Read(Path.Combine(Root(variant), "reference", "ref_0.png")), PngCodec.Read(Path.Combine(Root(variant), "reference", "ref_1.png")) };
        foreach (int n in source == "gguf" ? new[] { 2 } : new[] { 1, 2 })
        {
            var result = await generator.GenerateAsync(new ImageGenerationRequest
            {
                Prompt = index.GetProperty("prompt").GetString()!,
                NegativePrompt = index.GetProperty("negative").GetString(),
                ReferenceImages = images.Take(n).ToArray(),
            }, new ImageGenerationOptions
            {
                Width = width,
                Height = height,
                Steps = index.GetProperty("steps").GetInt32(),
                GuidanceScale = index.GetProperty("guidance").GetSingle(),
                InitialNoise = Tensor(variant, "noise"),
            }, TestContext.Current.CancellationToken);
            var expected = PlanarImage.FromPlanar(Tensor(variant, source == "gguf" ? "q8_image_2" : $"image_{n}"), width, height, 0f, 1f);
            string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
            if (output.Length > 0)
            {
                Directory.CreateDirectory(output);
                File.WriteAllBytes(Path.Combine(output, $"{variant}-{source}-{device}-{n}-sharplight.png"), PngCodec.Encode(result.Images[0]));
                File.WriteAllBytes(Path.Combine(output, $"{variant}-{source}-{device}-{n}-diffusers.png"), PngCodec.Encode(expected));
            }

            double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
            TestContext.Current.SendDiagnosticMessage($"{variant} {source} {device} {n} reference(s): PSNR {psnr:F1} dB");
            Assert.True(psnr > (source == "gguf" ? 30 : 38), $"{n} reference(s): PSNR {psnr:F1} dB");
        }
    }

    [Theory]
    [InlineData("edit-2509")]
    [InlineData("edit-2511")]
    public async Task PackageGeneratesTheSameEdit(string variant)
    {
        Assert.SkipUnless(Present(variant), "tiny-random-qwen-image-edit missing");
        _ = Env.Services;
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "imagegen-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"tiny-{variant}-{Guid.NewGuid():N}.slm");
        try
        {
            TestHost.Converter.Convert(Path.Combine(Root(variant), "diffusers"), path);
            var index = Index(variant);
            var request = new ImageGenerationRequest
            {
                Prompt = index.GetProperty("prompt").GetString()!,
                NegativePrompt = index.GetProperty("negative").GetString(),
                ReferenceImages = [PngCodec.Read(Path.Combine(Root(variant), "reference", "ref_0.png")), PngCodec.Read(Path.Combine(Root(variant), "reference", "ref_1.png"))],
            };
            var options = new ImageGenerationOptions { Width = 96, Height = 64, Steps = 2, Seed = 5 };
            var ct = TestContext.Current.CancellationToken;
            byte[] expected;
            using (var folderModel = Load(variant, "diffusers", Env.Cpu()))
            using (var generator = Env.Clients.Create<IImageGenerator>(folderModel))
            {
                expected = (await generator.GenerateAsync(request, options, ct)).Images[0].Rgb;
            }

            using var package = Env.Load(path, Env.Cpu());
            Assert.Contains(package.Description.Component(ModelComponentRoles.TextEncoder).Graphs, g => g == WellKnown.VisionEncodeGraph);
            Assert.True(VisionParameters.Of(package.Definition) is not null);
            Assert.Equal(3, package.Description.Component(ModelComponentRoles.Denoiser).ConfigValue(DiffusionConfigKeys.ReferenceImages, 0));
            using var fromPackage = Env.Clients.Create<IImageGenerator>(package);
            Assert.Equal(expected, (await fromPackage.GenerateAsync(request, options, ct)).Images[0].Rgb);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
