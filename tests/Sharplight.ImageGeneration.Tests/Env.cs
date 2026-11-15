using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vulkan;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>Test assets, devices and comparison helpers. Tests skip when their model files are missing.</summary>
internal static class Env
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    /// <summary>The tiny random SD 1.x pipeline (tools/make-tiny-diffusion-models.py).</summary>
    public static readonly string Tiny = Path.Combine(Home, "models/sd/tiny-random-sd");

    /// <summary>The tiny random SD 2.x pipeline (v-prediction, linear projections, GELU text encoder; tools/make-tiny-diffusion-models.py --family sd2).</summary>
    public static readonly string Tiny2 = Path.Combine(Home, "models/sd/tiny-random-sd2");

    /// <summary>Stable Diffusion 2.1 (768-v) as a diffusers folder (fp16 files).</summary>
    public static readonly string Sd21Diffusers = Path.Combine(Home, "models/sd/sd21-diffusers");

    /// <summary>Stable Diffusion 2.1 (768-v) as a single-file checkpoint.</summary>
    public static readonly string Sd21SingleFile = Path.Combine(Home, "models/sd/v2-1_768-ema-pruned.safetensors");

    /// <summary>SDXL base 1.0 as a diffusers folder (fp16 files).</summary>
    public static readonly string SdxlDiffusers = Path.Combine(Home, "models/sd/sdxl-diffusers");

    /// <summary>SDXL base 1.0 as a single-file checkpoint.</summary>
    public static readonly string SdxlSingleFile = Path.Combine(Home, "models/sd/sd_xl_base_1.0.safetensors");

    /// <summary>The tiny random SDXL pipeline (tools/make-tiny-sdxl.py).</summary>
    public static readonly string TinyXl = Path.Combine(Home, "models/sd/tiny-random-sdxl");

    /// <summary>Stable Diffusion 1.5 as a diffusers folder.</summary>
    public static readonly string Sd15Diffusers = Path.Combine(Home, "models/sd/sd15-diffusers");

    /// <summary>Stable Diffusion 1.5 as a single-file checkpoint.</summary>
    public static readonly string Sd15SingleFile = Path.Combine(Home, "models/sd/v1-5-pruned-emaonly.safetensors");

    /// <summary>The original SD 1.5 PyTorch checkpoint (optional).</summary>
    public static readonly string Sd15Ckpt = Path.Combine(Home, "models/sd/v1-5-pruned-emaonly.ckpt");

    /// <summary>Stable Diffusion 1.5 as stable-diffusion.cpp GGUF files.</summary>
    public static string Sd15Gguf(string type) => Path.Combine(Home, $"models/sd/gguf/stable-diffusion-v1-5-pruned-emaonly-{type}.gguf");

    /// <summary>The service provider with every format, adapter, provider and the task layer.</summary>
    public static ServiceProvider Services { get; } = new ServiceCollection().AddSharplight().BuildServiceProvider();

    /// <summary>The task client provider.</summary>
    public static TaskClientProvider Clients => Services.GetRequiredService<TaskClientProvider>();

    /// <summary>The tiny families the parity tests run.</summary>
    public static TheoryData<string> TinyFamilies => ["sd1", "sd2"];

    /// <summary>The tiny pipeline folder.</summary>
    public static string TinyDiffusers => Path.Combine(Tiny, "diffusers");

    /// <summary>The tiny pipeline's reference folder.</summary>
    public static string TinyReference => Path.Combine(Tiny, "reference");

    /// <summary>The root folder of a tiny family ("sd1" or "sd2").</summary>
    public static string TinyRoot(string family) => family switch
    {
        "sd2" => Tiny2,
        "sdxl" => TinyXl,
        _ => Tiny,
    };

    /// <summary>The diffusers folder of a tiny family.</summary>
    public static string TinyDiffusersOf(string family) => Path.Combine(TinyRoot(family), "diffusers");

    /// <summary>A reference tensor of a tiny family.</summary>
    public static float[] TinyTensorOf(string family, string name) => ReadFloats(Path.Combine(TinyRoot(family), "reference", name + ".f32"));

    /// <summary>The reference index of a tiny family.</summary>
    public static JsonElement TinyIndexOf(string family)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(TinyRoot(family), "reference", "index.json")));
        return document.RootElement.Clone();
    }

    /// <summary>CPU device slot.</summary>
    public static DeviceSlot Cpu()
    {
        var provider = new CpuExecutionProvider();
        return new DeviceSlot { Device = provider.OpenDevice(provider.GetDevices()[0]) };
    }

    private static readonly Lazy<DeviceSlot?> s_vulkan = new(OpenVulkan);

    /// <summary>The first discrete Vulkan GPU (SHARPLIGHT_VK_DEVICES pins one), shared by the tests (opened once), or null.</summary>
    public static DeviceSlot? Vulkan() => s_vulkan.Value;

    private static DeviceSlot? OpenVulkan()
    {
        var provider = new VulkanProvider();
        if (!provider.CheckAvailability().IsAvailable)
        {
            return null;
        }

        var devices = provider.GetDevices();
        var device = devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices.FirstOrDefault();
        return device is null ? null : new DeviceSlot { Device = provider.OpenDevice(device) };
    }

    /// <summary>A device slot by name ("cpu" or "vulkan"), or null when unavailable.</summary>
    public static DeviceSlot? Device(string name) => name == "cpu" ? Cpu() : Vulkan();

    /// <summary>Loads a model as a task model on one device.</summary>
    public static TaskModel Load(string path, DeviceSlot slot) => TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [slot] });

    /// <summary>Reads a raw little-endian f32 file.</summary>
    public static float[] ReadFloats(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    /// <summary>A reference tensor of the tiny pipeline.</summary>
    public static float[] TinyTensor(string name) => ReadFloats(Path.Combine(TinyReference, name + ".f32"));

    /// <summary>The tiny pipeline's reference index.</summary>
    public static JsonElement TinyIndex()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(TinyReference, "index.json")));
        return document.RootElement.Clone();
    }

    /// <summary>Largest absolute difference and the error relative to the reference's RMS.</summary>
    public static (float MaxAbs, double RelativeRms) Compare(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0;
        double errors = 0;
        double reference = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            float d = Math.Abs(actual[i] - expected[i]);
            maxAbs = Math.Max(maxAbs, d);
            errors += (double)d * d;
            reference += (double)expected[i] * expected[i];
        }

        return (maxAbs, Math.Sqrt(errors / Math.Max(reference, 1e-30)));
    }

    /// <summary>PSNR in dB of two 8-bit images.</summary>
    public static double Psnr(byte[] a, byte[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }

        double mse = sum / a.Length;
        return mse == 0 ? double.PositiveInfinity : 10 * Math.Log10(255.0 * 255.0 / mse);
    }
}
