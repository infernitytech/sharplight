using System.Text.Json;
using Sharplight.Imaging;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The image-generation hot-model sweep (docs/model-support.md section 19.6): every folder of the sweep root (default
/// <c>~/models/hot-imagegen</c>, <c>SHARPLIGHT_HOT_ROOT</c> moves it) holds a <c>manifest.json</c> (model file or folder, auxiliary files by
/// role, the reference folder, size, steps, guidance, negative prompt, sampler) written next to the references of
/// tools/make-hot-imagegen-references.py. Each model is loaded as distributed, generates from the reference's initial latents on Vulkan
/// and is compared with diffusers' image (PSNR); <c>SHARPLIGHT_HOT_DEVICE=cpu</c> runs on the CPU. Entries whose files are missing are skipped.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class HotSweepTests
{
    private static readonly string s_root = Environment.GetEnvironmentVariable("SHARPLIGHT_HOT_ROOT") is { Length: > 0 } root
        ? root
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hot-imagegen");

    /// <summary><c>SHARPLIGHT_HOT_FILTER</c>: only the entries whose folder name contains this text.</summary>
    private static readonly string s_filter = Environment.GetEnvironmentVariable("SHARPLIGHT_HOT_FILTER") ?? "";

    public static TheoryData<string> Entries()
    {
        var data = new TheoryData<string>();
        if (Directory.Exists(s_root))
        {
            foreach (string folder in Directory.EnumerateDirectories(s_root).Order(StringComparer.Ordinal))
            {
                if (File.Exists(Path.Combine(folder, "manifest.json")) && Path.GetFileName(folder).Contains(s_filter, StringComparison.Ordinal))
                {
                    data.Add(Path.GetFileName(folder));
                }
            }
        }

        if (data.Count == 0)
        {
            data.Add("none");
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Entries))]
    public async Task ModelMatchesDiffusers(string id)
    {
        string manifestPath = Path.Combine(s_root, id, "manifest.json");
        Assert.SkipUnless(File.Exists(manifestPath), $"{id}: no manifest");
        using var document = JsonDocument.Parse(File.ReadAllBytes(manifestPath));
        var manifest = document.RootElement;
        string Resolve(string relative) => Path.Combine(s_root, id, relative);
        string model = Resolve(manifest.GetProperty("model").GetString()!);
        var auxiliary = new Dictionary<string, string>();
        if (manifest.TryGetProperty("auxiliary", out var aux))
        {
            foreach (var entry in aux.EnumerateObject())
            {
                auxiliary[entry.Name] = Resolve(entry.Value.GetString()!);
            }
        }

        Assert.SkipUnless(File.Exists(model) || Directory.Exists(model), $"{id}: {model} missing");
        Assert.SkipUnless(auxiliary.Values.All(path => File.Exists(path) || Directory.Exists(path)), $"{id}: companion files missing");
        string referenceFolder = Resolve(manifest.GetProperty("reference").GetString()!);
        var slot = Env.Device(Environment.GetEnvironmentVariable("SHARPLIGHT_HOT_DEVICE") ?? "vulkan");
        Assert.SkipWhen(slot is null, "no device");
        float[] Tensor(string name) => Env.ReadFloats(Path.Combine(referenceFolder, name + ".f32"));
        int size = manifest.GetProperty("size").GetInt32();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        using var loaded = auxiliary.Count == 0 ? Env.Load(model, slot!) : Env.Load(model, slot!, auxiliary);
        using var generator = Env.Clients.Create<IImageGenerator>(loaded);
        double load = stopwatch.Elapsed.TotalSeconds;
        var options = new ImageGenerationOptions
        {
            Width = size,
            Height = manifest.TryGetProperty("height", out var h) ? h.GetInt32() : size,
            Steps = manifest.GetProperty("steps").GetInt32(),
            GuidanceScale = manifest.TryGetProperty("guidance", out var g) ? g.GetSingle() : null,
            Scheduler = manifest.TryGetProperty("sampler", out var s) && s.ValueKind == JsonValueKind.String ? s.GetString() : null,
            InitialNoise = Tensor("noise"),
        };
        var request = new ImageGenerationRequest
        {
            Prompt = manifest.GetProperty("prompt").GetString()!,
            NegativePrompt = manifest.TryGetProperty("negative", out var n) && n.ValueKind == JsonValueKind.String ? n.GetString() : null,
        };
        stopwatch.Restart();
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        double seconds = stopwatch.Elapsed.TotalSeconds;
        var expected = PlanarImage.FromPlanar(Tensor("image"), size, options.Height ?? size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"hot-{id}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"hot-{id}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{id} ({size}): PSNR {psnr:F1} dB; load {load:F1} s, image {seconds:F1} s");
        Assert.True(psnr > manifest.GetProperty("min_psnr").GetDouble(), $"{id}: PSNR {psnr:F1} dB");
    }
}
