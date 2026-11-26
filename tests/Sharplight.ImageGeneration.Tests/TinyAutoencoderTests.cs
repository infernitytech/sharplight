using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Tiny autoencoders (TAESD: madebyollin's taesd, taesdxl, taesd3 and taef1 in <c>~/models/sd/tae/{name}</c>) loaded as the <c>tiny-vae</c>
/// auxiliary source of the pipeline whose latents they decode, against diffusers' <c>AutoencoderTiny</c> (tools/make-taesd-references.py,
/// float32 on CUDA; references in <c>~/models/sd/tae-reference</c>): decoder and encoder graphs on the CPU, Vulkan and CUDA (with
/// <c>SHARPLIGHT_TEST_CUDA=1</c>), the original decoder / encoder files, the <c>.slm</c> round trip, and SD 1.5 decoding through TAESD
/// against diffusers' StableDiffusionPipeline with the TAESD as its VAE. Skips without the files.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class TinyAutoencoderTests
{
    public static TheoryData<string, string> Cases()
    {
        var data = new TheoryData<string, string>();
        foreach (string name in new[] { "taesd", "taesdxl", "taesd3", "taef1" })
        {
            foreach (string device in new[] { "cpu", "vulkan", "cuda" })
            {
                data.Add(name, device);
            }
        }

        return data;
    }

    /// <summary>The pipeline whose latent space a tiny autoencoder decodes.</summary>
    private static string Pipeline(string name) => name switch
    {
        "taesd" => Env.Sd15Diffusers,
        "taesdxl" => Env.SdxlDiffusers,
        "taesd3" => Env.Sd35MediumDiffusers,
        _ => Env.FluxSchnellDiffusers,
    };

    private static string Folder(string name) => Path.Combine(Env.TinyAutoencoders, name);

    private static string Reference(string name) => Path.Combine(Env.TinyAutoencoderReference, name);

    private static bool Present(string name)
        => Directory.Exists(Pipeline(name)) && Directory.Exists(Folder(name)) && File.Exists(Path.Combine(Reference(name), "index.json"));

    private static JsonElement Index(string name)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference(name), "index.json")));
        return document.RootElement.Clone();
    }

    private static TaskModel LoadWith(string name, DeviceSlot slot, string? tinyPath = null)
        => Env.Load(Pipeline(name), slot, new Dictionary<string, string> { [AuxiliaryRoles.TinyVae] = tinyPath ?? Folder(name) });

    [Theory]
    [MemberData(nameof(Cases))]
    public void GraphsMatchDiffusers(string name, string device)
    {
        var slot = Present(name) ? Env.Device(device) : null;
        Assert.SkipWhen(slot is null, $"{name}, its pipeline, its references or the {device} device missing (CUDA runs with SHARPLIGHT_TEST_CUDA=1)");
        using var model = LoadWith(name, slot!);
        int size = Index(name).GetProperty("size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        string reference = Reference(name);
        var decoded = GraphRunner.Run(model, DiffusionGraphs.TinyVaeRole, DiffusionGraphs.TinyVaeDecode, DiffusionGraphs.Image, 1, dims,
            (DiffusionGraphs.Latent, Env.ReadFloats(Path.Combine(reference, "latent.f32"))));
        var encoded = GraphRunner.Run(model, DiffusionGraphs.TinyVaeRole, DiffusionGraphs.TinyVaeEncode, DiffusionGraphs.Latent, 1, dims,
            (DiffusionGraphs.Image, Env.ReadFloats(Path.Combine(reference, "image.f32"))));
        var (maxDecoded, rmsDecoded) = Env.Compare(decoded, Env.ReadFloats(Path.Combine(reference, "decoded.f32")));
        var (maxEncoded, rmsEncoded) = Env.Compare(encoded, Env.ReadFloats(Path.Combine(reference, "encoded.f32")));
        TestContext.Current.SendDiagnosticMessage($"{name} {device}: decoder max {maxDecoded:E2} rms {rmsDecoded:E2}; encoder max {maxEncoded:E2} rms {rmsEncoded:E2}");
        // f32 on the CPU; the GPUs multiply with f16 operands.
        double tolerance = device == "cpu" ? 1e-4 : 5e-3;
        Assert.True(rmsDecoded < tolerance, $"decoder: max {maxDecoded}, relative RMS {rmsDecoded}");
        Assert.True(rmsEncoded < tolerance, $"encoder: max {maxEncoded}, relative RMS {rmsEncoded}");
    }

    /// <summary>
    /// The original decoder file (with the encoder file beside it) maps to the diffusers file's canonical names and shapes; taesd's hold the
    /// same weights, taesdxl's original files are a later training run than its diffusers file (same layout, other values).
    /// </summary>
    [Theory]
    [InlineData("taesd", true)]
    [InlineData("taesdxl", false)]
    public void OriginalFilesMatchTheDiffusersFile(string name, bool sameWeights)
    {
        string decoder = Path.Combine(Folder(name), name + "_decoder.safetensors");
        Assert.SkipUnless(File.Exists(decoder), $"{name}'s original files missing");
        var importer = new Sharplight.Import.Diffusers.DiffusersImporter();
        var original = importer.Open(decoder);
        var diffusers = importer.Open(Folder(name));
        try
        {
            Assert.Equal("tiny_autoencoder", original.Architecture);
            Assert.Equal(diffusers.Weights.Keys.Order(StringComparer.Ordinal), original.Weights.Keys.Order(StringComparer.Ordinal));
            foreach (var (key, weight) in diffusers.Weights)
            {
                Assert.Equal(weight.Shape, original.Weights[key].Shape);
                if (sameWeights)
                {
                    Assert.Equal(weight.ReadAll(), original.Weights[key].ReadAll());
                }
            }
        }
        finally
        {
            foreach (var resource in original.OwnedResources.Concat(diffusers.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }

    /// <summary>
    /// A transformer-only checkpoint completed with companion files (FLUX.1-schnell Q8_0 GGUF with CLIP-L, T5 and the VAE, the catalog's
    /// install) takes the tiny autoencoder beside its companions; its decoder matches diffusers as in <see cref="GraphsMatchDiffusers"/>.
    /// </summary>
    [Fact]
    public void CompanionCheckpointTakesTheTinyAutoencoder()
    {
        Assert.SkipUnless(File.Exists(Env.FluxSchnellGguf) && Directory.Exists(Folder("taef1")) && File.Exists(Path.Combine(Reference("taef1"), "index.json")),
            "FLUX.1-schnell GGUF, taef1 or its references missing");
        var auxiliary = new Dictionary<string, string>(TestHost.Loader.Importers.FindCompanions(Env.FluxSchnellGguf)) { [AuxiliaryRoles.TinyVae] = Folder("taef1") };
        using var model = Env.Load(Env.FluxSchnellGguf, Env.Cpu(), auxiliary);
        int size = Index("taef1").GetProperty("size").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        var decoded = GraphRunner.Run(model, DiffusionGraphs.TinyVaeRole, DiffusionGraphs.TinyVaeDecode, DiffusionGraphs.Image, 1, dims,
            (DiffusionGraphs.Latent, Env.ReadFloats(Path.Combine(Reference("taef1"), "latent.f32"))));
        var (max, rms) = Env.Compare(decoded, Env.ReadFloats(Path.Combine(Reference("taef1"), "decoded.f32")));
        Assert.True(rms < 1e-4, $"decoder: max {max}, relative RMS {rms}");
    }

    [Fact]
    public void MismatchedLatentSpaceFailsNamingTheRightAutoencoder()
    {
        Assert.SkipUnless(Present("taesd") && Directory.Exists(Folder("taef1")), "taesd, taef1 or SD 1.5 missing");
        var error = Assert.Throws<NotSupportedException>(() => LoadWith("taesd", Env.Cpu(), Folder("taef1")));
        Assert.Contains("TAEF1", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PackageKeepsTheTinyAutoencoder()
    {
        Assert.SkipUnless(Present("taesd"), "taesd or SD 1.5 missing");
        _ = Env.Services;
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "imagegen-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"sd15-taesd-{Guid.NewGuid():N}.slm");
        try
        {
            TestHost.Converter.Convert(Env.Sd15Diffusers, path, auxiliary: new Dictionary<string, string> { [AuxiliaryRoles.TinyVae] = Folder("taesd") });
            using var model = Env.Load(path, Env.Cpu());
            var component = model.Description.Component(DiffusionGraphs.TinyVaeRole);
            Assert.Equal([DiffusionGraphs.TinyVaeDecode, DiffusionGraphs.TinyVaeEncode], component.Graphs);
            using var generator = Env.Clients.Create<IImageGenerator>(model);
            Assert.True(generator.Defaults.TinyAutoencoder);
            // Another size rebuilds the tiny autoencoder's graphs from the package's metadata and weights.
            var latent = Env.ReadFloats(Path.Combine(Reference("taesd"), "latent.f32")).AsSpan(0, 4 * 8 * 16).ToArray();
            var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = 64, [DiffusionGraphs.Width] = 128 };
            var image = GraphRunner.Run(model, DiffusionGraphs.TinyVaeRole, DiffusionGraphs.TinyVaeDecode, DiffusionGraphs.Image, 1, dims, (DiffusionGraphs.Latent, latent));
            Assert.Equal(3 * 64 * 128, image.Length);
            var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a red cube" },
                new ImageGenerationOptions { Width = 64, Height = 64, Steps = 1, Seed = 3, TinyAutoencoder = true }, TestContext.Current.CancellationToken);
            Assert.Equal(64, result.Images[0].Width);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task Sd15WithTaesdMatchesDiffusers()
    {
        string reference = Path.Combine(Env.TinyAutoencoderReference, "sd15-taesd");
        Assert.SkipUnless(Present("taesd") && File.Exists(Path.Combine(reference, "index.json")), "taesd, SD 1.5 or the references missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(reference, "index.json")));
        var index = document.RootElement;
        int size = index.GetProperty("size").GetInt32();
        using var model = LoadWith("taesd", slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var options = new ImageGenerationOptions
        {
            Width = size,
            Height = size,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            Scheduler = Samplers.Euler,
            InitialNoise = Env.ReadFloats(Path.Combine(reference, "noise.f32")),
            TinyAutoencoder = true,
            PreviewInterval = 10,
        };
        ImageGenerationResult? result = null;
        var previews = new List<ImageInput>();
        await foreach (var update in generator.GenerateStreamingAsync(new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, options,
            TestContext.Current.CancellationToken))
        {
            result = update.Result ?? result;
            if (update.Preview is { } preview)
            {
                previews.Add(preview);
            }
        }

        var expected = PlanarImage.FromPlanar(Env.ReadFloats(Path.Combine(reference, "image.f32")), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, "sd15-taesd-sharplight.png"), PngCodec.Encode(result!.Images[0]));
            File.WriteAllBytes(Path.Combine(output, "sd15-taesd-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result!.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"SD 1.5 + TAESD: PSNR {psnr:F1} dB");
        Assert.True(psnr > 40, $"PSNR {psnr:F1} dB");
        // Previews come from the tiny autoencoder at the full size.
        Assert.Equal(2, previews.Count);
        Assert.All(previews, p => Assert.Equal(size, p.Width));
    }
}
