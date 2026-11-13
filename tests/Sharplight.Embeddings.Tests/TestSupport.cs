using System.Collections.Concurrent;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Runtime;
using Sharplight.Vulkan;

[assembly: Xunit.CollectionBehavior(CollectionBehavior.CollectionPerAssembly)]

namespace Sharplight.Embeddings.Tests;

static class Env
{
    static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    public static string Hf(string name) => Path.Combine(Home, "models/hf", name);
    public static string Gguf(string dir, string file) => Path.Combine(Home, "models/gguf", dir, file);

    static Env()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    public static void Init() { }

    public static JsonElement Reference(string name)
        => JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".json"))).RootElement;

    static readonly Lazy<DeviceSlot> s_cpu = new(() => { var p = new CpuExecutionProvider(); return new DeviceSlot { Device = p.OpenDevice(p.GetDevices()[0]) }; });
    static readonly Lazy<DeviceSlot?> s_vk = new(() =>
    {
        try
        {
            var p = new VulkanProvider();
            if (!p.CheckAvailability().IsAvailable) return null;
            var devs = p.GetDevices();
            var dev = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_VK_DEVICE") is { } e && int.TryParse(e, out int i) && i < devs.Count ? devs[i]
                : devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
            return dev is null ? null : new DeviceSlot { Device = p.OpenDevice(dev) };
        }
        catch { return null; }
    });

    /// <summary>"cpu" or "vulkan" (skips the test when no Vulkan device is available).</summary>
    public static DeviceSlot Device(string kind)
    {
        if (kind == "cpu") return s_cpu.Value;
        Assert.SkipWhen(s_vk.Value is null, "No Vulkan device.");
        return s_vk.Value!;
    }

    public static void RequireModel(string path) => Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"Model not found: {path}");

    public static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++) { ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
        return ab / Math.Sqrt(aa * bb);
    }

    public static float[] Floats(JsonElement e) => [.. e.EnumerateArray().Select(x => x.GetSingle())];
    public static int[] Ints(JsonElement e) => [.. e.EnumerateArray().Select(x => x.GetInt32())];

    /// <summary>Accuracy results of the run, written to accuracy-summary.txt next to the test binary at exit.</summary>
    public static readonly ConcurrentDictionary<string, string> Summary = new();

    static Env2 s_exit = new();
    sealed class Env2
    {
        public Env2() => AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try { File.WriteAllLines(Path.Combine(AppContext.BaseDirectory, "accuracy-summary.txt"), Summary.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key,-60} {kv.Value}")); }
            catch { }
        };
    }
}
