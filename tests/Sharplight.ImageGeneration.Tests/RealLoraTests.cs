using System.Globalization;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Real LoRAs from the Hub on their base checkpoints (tools/make-lora-references.py: ~/models/lora/{id}/ with the LoRA file and the
/// diffusers references from <c>load_lora_weights</c>, same latents, Euler, f32) on Vulkan; skips the entries whose files are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class RealLoraTests
{
    private static readonly string s_models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");

    public static TheoryData<string> Entries()
    {
        var data = new TheoryData<string>();
        string root = Path.Combine(s_models, "lora");
        if (Directory.Exists(root))
        {
            foreach (string folder in Directory.EnumerateDirectories(root).Where(f => !Path.GetFileName(f).StartsWith("tiny-", StringComparison.Ordinal)).Order(StringComparer.Ordinal))
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
    public async Task LoraMatchesDiffusers(string id)
    {
        string folder = Path.Combine(s_models, "lora", id);
        string referenceFolder = Path.Combine(folder, "reference");
        Assert.SkipUnless(File.Exists(Path.Combine(referenceFolder, "index.json")), $"{id}: no reference");
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(referenceFolder, "index.json")));
        var index = document.RootElement;
        string baseFile = Path.Combine(s_models, index.GetProperty("base").GetString()!);
        string lora = Path.Combine(folder, index.GetProperty("lora").GetString()!);
        Assert.SkipUnless(File.Exists(baseFile) && File.Exists(lora), $"{id}: base or LoRA file missing");
        var slot = Env.Vulkan();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        float[] Tensor(string name) => Env.ReadFloats(Path.Combine(referenceFolder, name + ".f32"));
        string role = $"{AuxiliaryRoles.Lora}@{index.GetProperty("scale").GetSingle().ToString(CultureInfo.InvariantCulture)}";
        using var model = Env.Load(baseFile, slot!, new Dictionary<string, string> { [role] = lora });
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
        var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, options,
            TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor("image"), size, size, 0f, 1f);
        var unadapted = PlanarImage.FromPlanar(Tensor("image_base"), size, size, 0f, 1f);
        string output = Environment.GetEnvironmentVariable("SHARPLIGHT_SD_TEST_OUTPUT") ?? "";
        if (output.Length > 0)
        {
            Directory.CreateDirectory(output);
            File.WriteAllBytes(Path.Combine(output, $"lora-{id}-sharplight.png"), PngCodec.Encode(result.Images[0]));
            File.WriteAllBytes(Path.Combine(output, $"lora-{id}-diffusers.png"), PngCodec.Encode(expected));
        }

        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        double fromBase = Env.Psnr(result.Images[0].Rgb, unadapted.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{id}: {psnr:F1} dB from diffusers with the LoRA, {fromBase:F1} dB from the image without it");
        Assert.True(psnr > 25, $"{id}: PSNR {psnr:F1} dB");
        Assert.True(fromBase < psnr - 5, $"{id}: the LoRA barely changes the image ({fromBase:F1} dB from the unadapted one)");
    }
}
