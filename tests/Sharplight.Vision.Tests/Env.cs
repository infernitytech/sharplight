using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Tests;

namespace Sharplight.Vision.Tests;

internal static class Env
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    public static readonly string Hf08 = Path.Combine(Home, "models/hf/Qwen3.5-0.8B");
    public static readonly string Gguf27 = Path.Combine(Home, "models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf");
    public static readonly string Mmproj27 = Path.Combine(Home, ".lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf");
    public static readonly string Mmproj08 = Path.Combine(AppContext.BaseDirectory, "../../../../../.work/mmproj-0.8B-f32.gguf");
    public static readonly string Gguf08 = Path.Combine(Home, "models/gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf");
    public static readonly string GgufVl2b = Path.Combine(Home, "models/gguf/Qwen3-VL-2B-Instruct/Qwen3VL-2B-Instruct-Q4_K_M.gguf");
    public static readonly string MmprojVl2b = Path.Combine(Home, "models/gguf/Qwen3-VL-2B-Instruct/mmproj-Qwen3VL-2B-Instruct-F16.gguf");
    public static readonly string HfVl2b = Path.Combine(Home, "models/hf/Qwen3-VL-2B-Instruct");
    public static readonly string HfQwen2Vl2b = Path.Combine(Home, "models/hf/Qwen2-VL-2B-Instruct");
    public static readonly string GgufQwen2Vl2b = Path.Combine(Home, "models/gguf/Qwen2-VL-2B-Instruct/Qwen2-VL-2B-Instruct-Q4_K_M.gguf");
    public static readonly string MmprojQwen2Vl2b = Path.Combine(Home, "models/gguf/Qwen2-VL-2B-Instruct/mmproj-Qwen2-VL-2B-Instruct-f16.gguf");
    public static readonly string HfQwen25Vl3b = Path.Combine(Home, "models/hf/Qwen2.5-VL-3B-Instruct");
    public static readonly string GgufQwen25Vl3b = Path.Combine(Home, "models/gguf/Qwen2.5-VL-3B-Instruct/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf");
    public static readonly string MmprojQwen25Vl3b = Path.Combine(Home, "models/gguf/Qwen2.5-VL-3B-Instruct/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf");

    public static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    public static ImageInput Shapes() => new(320, 320, File.ReadAllBytes(Fixture("shapes_320x320.rgb")));

    public static float[] ReadFloats(string path)
    {
        var b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }

    /// <summary>CPU provider device slot (always available).</summary>
    public static DeviceSlot Cpu()
    {
        var p = new CpuExecutionProvider();
        return new DeviceSlot { Device = p.OpenDevice(p.GetDevices()[0]) };
    }

    private static readonly Lazy<DeviceSlot?> s_vulkan = new(OpenGpu);

    /// <summary>
    /// The GPU device slot shared by every test of the process, or null if none is available. Despite the name it is the GPU under
    /// test (<see cref="TestGpu"/>: <c>SHARPLIGHT_TEST_PROVIDER=vulkan|cuda|metal</c>, default Vulkan; the discrete GPU if any,
    /// <c>SHARPLIGHT_TEST_DEVICE</c> selects an index). It is opened once and never disposed by the tests: models do not own their
    /// devices, and a device per test would leave one driver context alive per test until the process exits, which the driver refuses
    /// after a few dozen (<c>VkException: ErrorInitializationFailed</c> from vkCreateDevice).
    /// </summary>
    public static DeviceSlot? Vulkan() => s_vulkan.Value;

    private static DeviceSlot? OpenGpu()
    {
        var (device, reason) = TestGpu.Open();
        if (device is null && reason.StartsWith(TestGpu.OpenFailedPrefix, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(reason);
        }

        return device is null ? null : new DeviceSlot { Device = device };
    }

    public static (float MaxAbs, float MaxRel, double Rms) Compare(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float maxAbs = 0;
        float maxRel = 0;
        double sq = 0;
        double refSq = 0;
        float scale = 0;
        foreach (var v in b)
        {
            scale = Math.Max(scale, Math.Abs(v));
        }

        for (int i = 0; i < a.Length; i++)
        {
            float d = Math.Abs(a[i] - b[i]);
            maxAbs = Math.Max(maxAbs, d);
            maxRel = Math.Max(maxRel, d / Math.Max(Math.Abs(b[i]), 1e-2f * scale));
            sq += (double)d * d;
            refSq += (double)b[i] * b[i];
        }
        return (maxAbs, maxRel, Math.Sqrt(sq / Math.Max(refSq, 1e-30)));
    }

    /// <summary>The task client factories of the test container (<c>services.AddSharplight()</c>).</summary>
    public static IEnumerable<Sharplight.Tasks.ITaskClientFactory> TaskFactories()
        => Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetServices<Sharplight.Tasks.ITaskClientFactory>(TestHost.Services);
}
