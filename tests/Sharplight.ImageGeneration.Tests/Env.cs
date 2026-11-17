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

    /// <summary>The FLUX model folder (diffusers folders, GGUF files with companions, references).</summary>
    public static readonly string FluxRoot = Path.Combine(Home, "models/flux");

    /// <summary>FLUX.1-schnell as a diffusers folder (bf16 files).</summary>
    public static readonly string FluxSchnellDiffusers = Path.Combine(FluxRoot, "FLUX.1-schnell");

    /// <summary>FLUX.1-schnell as city96's transformer-only Q8_0 GGUF, with the T5 GGUF, clip_l.safetensors and ae.safetensors beside it.</summary>
    public static readonly string FluxSchnellGguf = Path.Combine(FluxRoot, "gguf", "flux1-schnell-Q8_0.gguf");

    /// <summary>FLUX.1-dev as city96's transformer-only Q8_0 GGUF (its companions are those of <see cref="FluxSchnellGguf"/>).</summary>
    public static readonly string FluxDevGguf = Path.Combine(FluxRoot, "gguf-dev", "flux1-dev-Q8_0.gguf");

    /// <summary>The tiny random FLUX pipelines (tools/make-tiny-flux.py), one folder per variant ("dev", "schnell").</summary>
    public static readonly string TinyFlux = Path.Combine(Home, "models/flux/tiny-random-flux");

    /// <summary>The tiny random SD3 pipelines (tools/make-tiny-sd3.py), one folder per variant ("sd3", "sd35").</summary>
    public static readonly string TinySd3 = Path.Combine(Home, "models/sd3/tiny-random-sd3");

    /// <summary>The SD3 model folder (diffusers folders, GGUF files with companions, references).</summary>
    public static readonly string Sd3Root = Path.Combine(Home, "models/sd3");

    /// <summary>Stable Diffusion 3.5 Medium as a diffusers folder (fp16 files; an ungated mirror of the official repository).</summary>
    public static readonly string Sd35MediumDiffusers = Path.Combine(Sd3Root, "sd3.5-medium-diffusers");

    /// <summary>Stable Diffusion 3.5 Medium as city96's transformer-only Q8_0 GGUF.</summary>
    public static readonly string Sd35MediumGguf = Path.Combine(Sd3Root, "gguf", "sd3.5_medium-Q8_0.gguf");

    /// <summary>city96's T5-XXL encoder Q8_0 GGUF (the T5 companion of FLUX, SD3 and Chroma).</summary>
    public static readonly string T5XxlGguf = Path.Combine(FluxRoot, "gguf", "t5-v1_1-xxl-encoder-Q8_0.gguf");

    /// <summary>The tiny random Chroma pipeline (tools/make-tiny-chroma.py).</summary>
    public static readonly string TinyChroma = Path.Combine(Home, "models/chroma/tiny-random-chroma");

    /// <summary>The tiny random Z-Image pipeline (tools/make-tiny-zimage.py).</summary>
    public static readonly string TinyZImage = Path.Combine(Home, "models/zimage/tiny-random-zimage");

    /// <summary>The tiny random LoRAs and their references (tools/make-tiny-lora.py).</summary>
    public static readonly string TinyLora = Path.Combine(Home, "models/lora/tiny-lora");

    /// <summary>The tiny random Qwen-Image pipeline (tools/make-tiny-qwen-image.py).</summary>
    public static readonly string TinyQwenImage = Path.Combine(Home, "models/qwen-image/tiny-random-qwen-image");

    /// <summary>The tiny random Qwen-Image 2.1 pipeline (tools/make-tiny-qwen-image21.py).</summary>
    public static readonly string TinyQwenImage21 = Path.Combine(Home, "models/qwen-image21/tiny-random-qwen-image21");

    /// <summary>The FLUX.2 model folder (the FLUX.2 Klein 4B diffusers folder, GGUF files, references).</summary>
    public static readonly string Flux2Root = Path.Combine(Home, "models/flux2");

    /// <summary>The tiny random FLUX.2 Klein pipeline (tools/make-tiny-flux2.py).</summary>
    public static readonly string TinyFlux2 = Path.Combine(Home, "models/flux2/tiny-random-flux2");

    /// <summary>The FLUX.2 Dev model folder (the diffusers folder without its weights, the transformer and Mistral text encoder GGUF files, references).</summary>
    public static readonly string Flux2DevRoot = Path.Combine(Home, "models/flux2-dev");

    /// <summary>The Krea 2 model folder (the Krea 2 Turbo diffusers folder, GGUF files, references).</summary>
    public static readonly string Krea2Root = Path.Combine(Home, "models/krea2");

    /// <summary>The tiny random Krea 2 pipeline (tools/make-tiny-krea2.py).</summary>
    public static readonly string TinyKrea2 = Path.Combine(Home, "models/krea2/tiny-random-krea2");

    /// <summary>The Qwen-Image model folder (the Qwen-Image diffusers folder without the transformer weights, GGUF files, references).</summary>
    public static readonly string QwenImageRoot = Path.Combine(Home, "models/qwen-image");

    /// <summary>The Z-Image model folder (the Z-Image-Turbo diffusers folder, GGUF files, references).</summary>
    public static readonly string ZImageRoot = Path.Combine(Home, "models/zimage");

    /// <summary>The Chroma model folder (GGUF files, configurations, references).</summary>
    public static readonly string ChromaRoot = Path.Combine(Home, "models/chroma");

    /// <summary>The tiny random Wan pipelines (tools/make-tiny-wan.py: t2v, ti2v).</summary>
    public static readonly string TinyWan = Path.Combine(Home, "models/video/tiny-random-wan");

    /// <summary>The video models' folder (SHARPLIGHT_VIDEO_ROOT overrides it, e.g. with a copy on a local disk).</summary>
    public static readonly string VideoRoot = Environment.GetEnvironmentVariable("SHARPLIGHT_VIDEO_ROOT") is { Length: > 0 } videoRoot
        ? videoRoot
        : Path.Combine(Home, "models/video");

    /// <summary>Stable Diffusion 1.5 as a diffusers folder.</summary>
    public static readonly string Sd15Diffusers = Path.Combine(Home, "models/sd/sd15-diffusers");

    /// <summary>Stable Diffusion 1.5 as a single-file checkpoint.</summary>
    public static readonly string Sd15SingleFile = Path.Combine(Home, "models/sd/v1-5-pruned-emaonly.safetensors");

    /// <summary>The original SD 1.5 PyTorch checkpoint (optional).</summary>
    public static readonly string Sd15Ckpt = Path.Combine(Home, "models/sd/v1-5-pruned-emaonly.ckpt");

    /// <summary>Stable Diffusion 1.5 as stable-diffusion.cpp GGUF files.</summary>
    public static string Sd15Gguf(string type) => Path.Combine(Home, $"models/sd/gguf/stable-diffusion-v1-5-pruned-emaonly-{type}.gguf");

    /// <summary>
    /// SDXL as stable-diffusion.cpp GGUF files from the Hub: "Q8_0" / "Q4_0" (gpustack/stable-diffusion-xl-base-1.0-GGUF, sd.cpp's
    /// own tensor names), "Q4_K" (kostakoff/stable-diffusion-xl-base-1.0-GGUF, the original SGM names) and "turbo" (OlegSkutte/sdxl-turbo-GGUF).
    /// </summary>
    public static string SdxlGguf(string variant) => Path.Combine(Home, variant switch
    {
        "Q4_K" => "models/sd/gguf/sd_xl_base_1.0_0_Q4_K.gguf",
        "turbo" => "models/sd/gguf/sd_xl_turbo_1.0.q8_0.gguf",
        _ => $"models/sd/gguf/stable-diffusion-xl-base-1.0-{variant}.gguf",
    });

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
    public static TaskModel Load(string path, DeviceSlot slot) => TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [slot], Log = Log });

    /// <summary>Loads a model with auxiliary files by role (companions of a transformer-only checkpoint).</summary>
    public static TaskModel Load(string path, DeviceSlot slot, IReadOnlyDictionary<string, string> auxiliary)
        => TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [slot], Auxiliary = new Dictionary<string, string>(auxiliary), Log = Log });

    /// <summary>Load and compile messages on stderr when <c>SHARPLIGHT_TEST_LOG</c> is set (diagnosing slow or stuck large-model tests).</summary>
    private static Action<string>? Log => Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_LOG") is { Length: > 0 }
        ? message => Console.Error.WriteLine($"{DateTime.Now:HH:mm:ss} {message}")
        : null;

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
