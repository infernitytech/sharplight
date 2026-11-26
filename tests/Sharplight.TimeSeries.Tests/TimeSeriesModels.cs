using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tests;

namespace Sharplight.TimeSeries.Tests;

/// <summary>
/// Locates the time-series models (<c>~/models/timeseries/&lt;name&gt;</c>, Hub downloads) and their reference forecasts
/// (<c>~/models/timeseries/refs/&lt;name&gt;.json</c>, written by <c>tools/timeseries-reference/reference_chronos.py</c>) and opens devices; tests skip when a file is missing.
/// </summary>
internal static class TimeSeriesModels
{
    private static readonly Lazy<ServiceProvider> s_services = new(() => SharplightDefaults.CreateServiceProvider());
    private static readonly string s_root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "timeseries");
    private static readonly Lazy<DeviceSlot?> s_cpu = new(() => OpenFirst(gpu: false));
    private static readonly Lazy<DeviceSlot?> s_gpu = new(() => OpenFirst(gpu: true));

    /// <summary>The container with <c>services.AddSharplight()</c>, built once per test assembly.</summary>
    public static IServiceProvider Services => s_services.Value;

    /// <summary>A model directory.</summary>
    public static string Model(string name) => Path.Combine(s_root, name);

    /// <summary>The reference file of a model.</summary>
    public static string Reference(string name) => Path.Combine(s_root, "refs", name + ".json");

    /// <summary>Skips the test unless every file or directory exists.</summary>
    public static void Require(params string[] paths)
    {
        foreach (string path in paths)
        {
            Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"{path} not present");
        }
    }

    /// <summary>The reference cases of a model.</summary>
    public static IReadOnlyList<ReferenceCase> Cases(string name)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Reference(name)));
        var cases = new List<ReferenceCase>();
        foreach (var c in document.RootElement.GetProperty("cases").EnumerateArray())
        {
            cases.Add(new ReferenceCase(
                c.GetProperty("name").GetString()!,
                [.. c.GetProperty("history").EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? float.NaN : v.GetSingle())],
                c.GetProperty("horizon").GetInt32(),
                [.. c.GetProperty("levels").EnumerateArray().Select(v => v.GetSingle())],
                [.. c.GetProperty("quantiles").EnumerateArray().Select(row => row.EnumerateArray().Select(v => v.GetSingle()).ToArray())],
                [.. c.GetProperty("mean").EnumerateArray().Select(v => v.GetSingle())]));
        }

        return cases;
    }

    /// <summary>
    /// The largest error of <paramref name="actual"/> against <paramref name="expected"/>, relative to the spread of the expected values (but at least a thousandth
    /// of their largest magnitude, so that a flat forecast does not divide by zero).
    /// </summary>
    public static double RelativeError(IReadOnlyList<float[]> actual, IReadOnlyList<float[]> expected)
    {
        double low = double.MaxValue;
        double high = double.MinValue;
        double magnitude = 0;
        double error = 0;
        for (int q = 0; q < expected.Count; q++)
        {
            Assert.Equal(expected[q].Length, actual[q].Length);
            for (int t = 0; t < expected[q].Length; t++)
            {
                low = Math.Min(low, expected[q][t]);
                high = Math.Max(high, expected[q][t]);
                magnitude = Math.Max(magnitude, Math.Abs(expected[q][t]));
                error = Math.Max(error, Math.Abs(actual[q][t] - expected[q][t]));
            }
        }

        return error / Math.Max(high - low, Math.Max(1e-3 * magnitude, 1e-6));
    }

    /// <summary>The CPU device or the GPU under test, opened once per assembly (skipping the test when there is none).</summary>
    public static DeviceSlot Open(bool gpu)
    {
        var slot = (gpu ? s_gpu : s_cpu).Value;
        Assert.SkipWhen(slot is null, gpu ? "No GPU provider is available" : "The CPU provider is not available");
        return slot!;
    }

    /// <summary>Loads a model directory or file as a task model.</summary>
    public static TaskModel Load(DeviceSlot slot, string path)
        => TaskModel.Load(Services.GetRequiredService<ModelLoader>(), new TaskLoadOptions { ModelPath = path, Devices = [slot] });

    /// <summary>The client of kind <typeparamref name="TClient"/> of a loaded model.</summary>
    public static TClient Client<TClient>(TaskModel model)
        where TClient : class, ITaskClient
        => Services.GetRequiredService<TaskClientProvider>().Create<TClient>(model);

    private static DeviceSlot? OpenFirst(bool gpu)
    {
        if (gpu)
        {
            // The GPU under test (SHARPLIGHT_TEST_PROVIDER=vulkan|cuda|metal, default Vulkan; tests/Shared/TestGpu.cs).
            var (device, _) = TestGpu.Open();
            return device is null ? null : new DeviceSlot { Device = device };
        }

        foreach (var (provider, availability) in Services.GetRequiredService<ExecutionProviderCatalog>().Discover())
        {
            if (availability.IsAvailable && provider.Name == "cpu")
            {
                return new DeviceSlot { Device = provider.OpenDevice(provider.GetDevices()[0]) };
            }
        }

        return null;
    }
}
