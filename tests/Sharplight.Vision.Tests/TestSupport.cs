using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Vision.Tests;

internal static class Env
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    public static readonly string Hf08 = Path.Combine(Home, "models/hf/Qwen3.5-0.8B");
    public static readonly string Gguf27 = Path.Combine(Home, "models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf");
    public static readonly string Mmproj27 = Path.Combine(Home, ".lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf");
    public static readonly string Mmproj08 = Path.Combine(AppContext.BaseDirectory, "../../../../../.work/mmproj-0.8B-f32.gguf");
    public static readonly string Gguf08 = Path.Combine(Home, "models/gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf");

    static Env()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
        ModelPackageImporter.Register();
    }

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

    /// <summary>A Vulkan device slot (the discrete GPU if any, SHARPLIGHT_TEST_VK_DEVICE selects an index), or null if none is available.</summary>
    public static DeviceSlot? Vulkan()
    {
        var p = new VulkanProvider();
        if (!p.CheckAvailability().IsAvailable)
        {
            return null;
        }

        var devs = p.GetDevices();
        var dev = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE") is { } e && int.TryParse(e, out int i) && i < devs.Count ? devs[i]
            : devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
        return dev is null ? null : new DeviceSlot { Device = p.OpenDevice(dev) };
    }

    public static (float MaxAbs, float MaxRel, double Rms) Compare(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float maxAbs = 0, maxRel = 0; double sq = 0, refSq = 0;
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
            sq += (double)d * d; refSq += (double)b[i] * b[i];
        }
        return (maxAbs, maxRel, Math.Sqrt(sq / Math.Max(refSq, 1e-30)));
    }
}
