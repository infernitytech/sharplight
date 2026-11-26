using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// LyCORIS and DoRA files merged at load (tools/make-tiny-lycoris.py): random LoCon (linears and convolutions), LoHa and LoKr (the
/// convolutions with Tucker cores) and DoRA adapters of the tiny SD 1.x UNet in kohya / LyCORIS names (DoRA also in PEFT's), loaded with
/// strength 0.8 (role <c>lora@0.8</c>), against PEFT's <c>merge_and_unload</c> at that strength: the merged weights (a residual convolution
/// and an attention projection) and the image from the same initial noise, on the CPU, Vulkan and CUDA (with <c>SHARPLIGHT_TEST_CUDA=1</c>).
/// </summary>
public sealed class TinyLycorisTests
{
    private static readonly string Root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/lora/tiny-lycoris");

    private static string Reference => Path.Combine(Root, "reference");

    public static TheoryData<string, string> Cases()
    {
        var data = new TheoryData<string, string>();
        foreach (string device in new[] { "cpu", "vulkan", "cuda" })
        {
            foreach (string variant in new[] { "locon", "loha", "lokr", "dora", "dora-peft" })
            {
                data.Add(device, variant);
            }
        }

        return data;
    }

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Reference, "index.json")));
        return document.RootElement.Clone();
    }

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Reference, name + ".f32"));

    private static string Role(JsonElement index)
        => $"{AuxiliaryRoles.Lora}@{index.GetProperty("scale").GetSingle().ToString(System.Globalization.CultureInfo.InvariantCulture)}";

    [Theory]
    [InlineData("locon")]
    [InlineData("loha")]
    [InlineData("lokr")]
    [InlineData("dora")]
    [InlineData("dora-peft")]
    public void MergedWeightsMatchPeft(string variant)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Reference, "index.json")), "tiny-lycoris references missing");
        var index = Index();
        using var model = Env.Load(Env.TinyDiffusers, Env.Cpu(), new Dictionary<string, string> { [Role(index)] = Path.Combine(Root, variant + ".safetensors") });
        string reference = variant.Split('-')[0];
        foreach (var check in index.GetProperty("checked").EnumerateObject())
        {
            string canonical = check.Value.GetString()!;
            var weight = model.Definition.Weights[canonical];
            Assert.Equal(Tensors.DataType.F32, weight.Type);
            var merged = MemoryMarshal.Cast<byte, float>(weight.ReadAll()).ToArray();
            var (max, rms) = Env.Compare(merged, Tensor($"{reference}_{canonical}"));
            TestContext.Current.SendDiagnosticMessage($"{variant} {canonical}: max {max:E2}, relative RMS {rms:E2}");
            Assert.True(rms < 1e-5, $"{variant} {canonical}: max {max}, relative RMS {rms}");
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task ImageMatchesPeft(string device, string variant)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Reference, "index.json")), "tiny-lycoris references missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, $"no {device} device (CUDA runs with SHARPLIGHT_TEST_CUDA=1)");
        var index = Index();
        using var model = Env.Load(Env.TinyDiffusers, slot!, new Dictionary<string, string> { [Role(index)] = Path.Combine(Root, variant + ".safetensors") });
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        int size = index.GetProperty("size").GetInt32();
        var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, new ImageGenerationOptions
        {
            Width = size,
            Height = size,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance").GetSingle(),
            InitialNoise = Tensor("noise"),
        }, TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor("image_" + variant.Split('-')[0]), size, size, 0f, 1f);
        var unadapted = PlanarImage.FromPlanar(Tensor("image_base"), size, size, 0f, 1f);
        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        double fromBase = Env.Psnr(result.Images[0].Rgb, unadapted.Rgb);
        TestContext.Current.SendDiagnosticMessage($"{variant} on {device}: {psnr:F1} dB from PEFT's merge, {fromBase:F1} dB from the image without it");
        Assert.True(psnr > 40, $"{variant} on {device}: PSNR {psnr:F1} dB");
        Assert.True(fromBase < psnr - 10, $"{variant}: the adapter barely changes the image ({fromBase:F1} dB from the unadapted one)");
    }
}
