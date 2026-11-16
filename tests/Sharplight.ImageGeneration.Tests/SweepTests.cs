using System.Text.Json;
using Sharplight.Imaging;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Popular single-file checkpoints from the Hub (SD 1.5 and SDXL fine-tunes and merges, tools/make-sweep-references.py): each file in
/// ~/models/imagegen-sweep/{id}/ is loaded as distributed and generates from the same latents as diffusers' from_single_file run (Euler,
/// f32, guided with a negative prompt) on Vulkan; skips the entries whose files or references are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class SweepTests
{
    private static readonly string s_root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/imagegen-sweep");

    public static TheoryData<string> Entries()
    {
        var data = new TheoryData<string>();
        if (Directory.Exists(s_root))
        {
            foreach (string folder in Directory.EnumerateDirectories(s_root).Order(StringComparer.Ordinal))
            {
                data.Add(Path.GetFileName(folder));
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
    public async Task CheckpointMatchesDiffusers(string id)
    {
        string folder = Path.Combine(s_root, id);
        string referenceFolder = Path.Combine(folder, "reference");
        Assert.SkipUnless(File.Exists(Path.Combine(referenceFolder, "index.json")), $"{id}: no reference");
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(referenceFolder, "index.json")));
        var index = document.RootElement;
        string file = Path.Combine(folder, index.GetProperty("file").GetString()!);
        Assert.SkipUnless(File.Exists(file), $"{id}: {file} missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        float[] Tensor(string name) => Env.ReadFloats(Path.Combine(referenceFolder, name + ".f32"));
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        using var model = Env.Load(file, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        int size = index.GetProperty("size").GetInt32();
        var options = new ImageGenerationOptions
        {
            Width = size,
            Height = size,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance").GetSingle(),
            Scheduler = index.GetProperty("sampler").GetString(),
            InitialNoise = Tensor("noise"),
        };
        var request = new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative").ValueKind == JsonValueKind.String ? index.GetProperty("negative").GetString() : null,
        };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        double seconds = stopwatch.Elapsed.TotalSeconds;
        var expected = PlanarImage.FromPlanar(Tensor("image"), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"sweep-{id}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"sweep-{id}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{id} ({index.GetProperty("family").GetString()}, {size}): PSNR {psnr:F1} dB; load + first image {seconds:F1} s");
        Assert.True(psnr > 25, $"{id}: PSNR {psnr:F1} dB");
    }
}
