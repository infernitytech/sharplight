using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Tests;

namespace Sharplight.AudioModels.Tests;

/// <summary>Locates the downloaded audio models and reference tensors and opens devices; tests skip when a file is missing.</summary>
internal static class AudioModels
{
    private static readonly Lazy<ServiceProvider> s_services = new(() => SharplightDefaults.CreateServiceProvider());
    private static readonly string s_root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "audio-models");

    /// <summary>The container with <c>services.AddSharplight()</c>, built once per test assembly.</summary>
    public static IServiceProvider Services => s_services.Value;

    /// <summary>A model directory of the audio model root.</summary>
    public static string Model(string name) => Path.Combine(s_root, name);

    /// <summary>A reference tensor file (raw little-endian arrays written by the Python reference scripts).</summary>
    public static string Reference(string name) => Path.Combine(s_root, "refs", name);

    /// <summary>Skips the test unless every file or directory exists.</summary>
    public static void Require(params string[] paths)
    {
        foreach (string path in paths)
        {
            Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"{path} not present");
        }
    }

    /// <summary>Reads a raw f32 reference.</summary>
    public static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Reference(name))).ToArray();

    /// <summary>Reads a raw i32 reference.</summary>
    public static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Reference(name))).ToArray();

    /// <summary>The relative RMS error of <paramref name="actual"/> against <paramref name="expected"/>.</summary>
    public static double RelativeRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double error = 0;
        double energy = 0;
        int n = Math.Min(actual.Length, expected.Length);
        for (int i = 0; i < n; i++)
        {
            double d = actual[i] - expected[i];
            error += d * d;
            energy += (double)expected[i] * expected[i];
        }

        return Math.Sqrt(error / Math.Max(energy, 1e-30));
    }

    private static readonly Lazy<DeviceSlot?> s_cpu = new(() => OpenFirst(gpu: false));
    private static readonly Lazy<DeviceSlot?> s_gpu = new(() => OpenFirst(gpu: true));

    /// <summary>
    /// The CPU device or the GPU under test, opened once and shared by every test of the assembly (skipping the test when there is
    /// none). Models do not own their devices: a device per test would stay alive until the process exits, and the GPU driver refuses
    /// new devices after a few dozen live ones (<c>ErrorInitializationFailed</c>).
    /// </summary>
    public static DeviceSlot Open(bool gpu)
    {
        var slot = (gpu ? s_gpu : s_cpu).Value;
        Assert.SkipWhen(slot is null, gpu ? "No GPU provider is available" : "The CPU provider is not available");
        return slot!;
    }

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

    /// <summary>Loads a model directory or file as a task model.</summary>
    public static TaskModel Load(DeviceSlot slot, string path)
        => TaskModel.Load(Services.GetRequiredService<ModelLoader>(), new TaskLoadOptions { ModelPath = path, Devices = [slot] });

    /// <summary>The client of kind <typeparamref name="TClient"/> of a loaded model.</summary>
    public static TClient Client<TClient>(TaskModel model)
        where TClient : class, ITaskClient
        => Services.GetRequiredService<TaskClientProvider>().Create<TClient>(model);
}
