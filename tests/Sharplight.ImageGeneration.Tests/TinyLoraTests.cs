using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// LoRAs merged at load (tools/make-tiny-lora.py): random LoRAs for the tiny SD 1.x, SDXL and FLUX.1-dev pipelines in diffusers' PEFT
/// format and kohya-ss' format (SD 1.x: diffusers module names; SDXL: the SGM UNet's names and both text encoders; FLUX: the original
/// names with fused q/k/v), loaded with strength 0.8 (role <c>lora@0.8</c>), give diffusers' <c>load_lora_weights</c> images from the
/// same initial noise; on the CPU and on Vulkan.
/// </summary>
public sealed class TinyLoraTests
{
    public static TheoryData<string, string, string> Cases()
    {
        var data = new TheoryData<string, string, string>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string family in new[] { "sd1", "sdxl", "flux" })
            {
                foreach (string format in new[] { "peft", "kohya" })
                {
                    data.Add(device, family, format);
                }
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task LoraImageMatchesDiffusers(string device, string family, string format)
    {
        string folder = Path.Combine(Env.TinyLora, family);
        string referenceFolder = Path.Combine(folder, "reference");
        Assert.SkipUnless(File.Exists(Path.Combine(referenceFolder, "index.json")), "tiny-lora references missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(referenceFolder, "index.json")));
        var index = document.RootElement;
        float[] Tensor(string name) => Env.ReadFloats(Path.Combine(referenceFolder, name + ".f32"));
        string pipeline = family == "flux" ? Path.Combine(Env.TinyFlux, "dev", "diffusers") : Path.Combine(Env.TinyRoot(family), "diffusers");
        string role = $"{AuxiliaryRoles.Lora}@{index.GetProperty("scale").GetSingle().ToString(System.Globalization.CultureInfo.InvariantCulture)}";
        using var model = Env.Load(pipeline, slot!, new Dictionary<string, string> { [role] = Path.Combine(folder, format + ".safetensors") });
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        int width = index.TryGetProperty("width", out var w) ? w.GetInt32() : index.GetProperty("size").GetInt32();
        int height = index.TryGetProperty("height", out var h) ? h.GetInt32() : index.GetProperty("size").GetInt32();
        var options = new ImageGenerationOptions
        {
            Width = width,
            Height = height,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance").GetSingle(),
            InitialNoise = Tensor("noise"),
        };
        var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, options,
            TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor("image_" + format), width, height, 0f, 1f);
        var unadapted = PlanarImage.FromPlanar(Tensor("image_base"), width, height, 0f, 1f);
        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        double fromBase = Env.Psnr(result.Images[0].Rgb, unadapted.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{family} {format} on {device}: {psnr:F1} dB from diffusers with the LoRA, {fromBase:F1} dB from the image without it");
        Assert.True(psnr > 40, $"{family} {format} LoRA on {device}: PSNR {psnr:F1} dB");
        Assert.True(fromBase < psnr - 10, $"{family} {format}: the LoRA barely changes the image ({fromBase:F1} dB from the unadapted one)");
    }
}
