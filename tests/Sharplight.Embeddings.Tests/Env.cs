using System.Collections.Concurrent;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Tests;

[assembly: Xunit.CollectionBehavior(CollectionBehavior.CollectionPerAssembly)]

namespace Sharplight.Embeddings.Tests;

internal static class Env
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    public static string Hf(string name) => Path.Combine(Home, "models/hf", name);
    public static string Gguf(string dir, string file) => Path.Combine(Home, "models/gguf", dir, file);

    static Env()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) => WriteSummary();
    }

    private static void WriteSummary()
    {
        try
        {
            File.WriteAllLines(Path.Combine(AppContext.BaseDirectory, "accuracy-summary.txt"), Summary.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key,-60} {kv.Value}"));
        }
        catch (IOException)
        {
            // Best effort: the summary is informational.
        }
    }

    public static void Init()
    {
    }

    public static JsonElement Reference(string name)
        => JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", name + ".json"))).RootElement;

    private static readonly Lazy<DeviceSlot> s_cpu = new(() => { var p = new CpuExecutionProvider(); return new DeviceSlot { Device = p.OpenDevice(p.GetDevices()[0]) }; });
    // The GPU under test: Vulkan unless SHARPLIGHT_TEST_PROVIDER names another provider (tests/Shared/TestGpu.cs).
    private static readonly Lazy<DeviceSlot?> s_vk = new(() => TestGpu.Open().Device is { } device ? new DeviceSlot { Device = device } : null);

    /// <summary>"cpu" or "vulkan" (the GPU under test: Vulkan, or the provider SHARPLIGHT_TEST_PROVIDER names; skips the test when there is none).</summary>
    public static DeviceSlot Device(string kind)
    {
        if (kind == "cpu")
        {
            return s_cpu.Value;
        }

        Assert.SkipWhen(s_vk.Value is null, $"No {TestGpu.ProviderName} device.");
        return s_vk.Value!;
    }

    public static void RequireModel(string path) => Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"Model not found: {path}");

    public static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            ab += (double)a[i] * b[i];
            aa += (double)a[i] * a[i];
            bb += (double)b[i] * b[i];
        }

        return ab / Math.Sqrt(aa * bb);
    }

    public static float[] Floats(JsonElement e) => [.. e.EnumerateArray().Select(x => x.GetSingle())];
    public static int[] Ints(JsonElement e) => [.. e.EnumerateArray().Select(x => x.GetInt32())];

    /// <summary>Accuracy results of the run, written to accuracy-summary.txt next to the test binary at exit.</summary>
    public static readonly ConcurrentDictionary<string, string> Summary = new();
}
