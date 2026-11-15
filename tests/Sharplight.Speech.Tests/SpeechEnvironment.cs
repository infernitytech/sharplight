using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Cpu;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Models, audio and devices of the speech tests. Real checkpoints live under <c>~/models</c> (not in git): tests skip when they are
/// missing. <c>SHARPLIGHT_VK_DEVICES</c> pins the Vulkan device.
/// </summary>
internal static class SpeechEnvironment
{
    private static readonly string s_models = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models");
    private static readonly Lazy<IComputeDevice> s_cpu = new(() =>
    {
        var provider = new CpuExecutionProvider();
        return provider.OpenDevice(provider.GetDevices()[0]);
    });

    private static readonly Lazy<IComputeDevice?> s_vulkan = new(() =>
    {
        var provider = new VulkanProvider();
        if (!provider.CheckAvailability().IsAvailable || provider.GetDevices().Count == 0)
        {
            return null;
        }

        return provider.OpenDevice(provider.GetDevices()[0]);
    });

    static SpeechEnvironment() => new ServiceCollection().AddSharplight();

    /// <summary>The JFK sample (11 s, 16 kHz mono).</summary>
    public static string Jfk => Path.Combine(s_models, "audio", "jfk.wav");

    /// <summary>A Hugging Face checkpoint directory under <c>~/models/hf</c>.</summary>
    /// <param name="name">The directory name, e.g. <c>whisper-tiny</c>.</param>
    public static string HuggingFace(string name) => Path.Combine(s_models, "hf", name);

    /// <summary>A whisper.cpp model file under <c>~/models/whisper.cpp</c>.</summary>
    /// <param name="name">The name without extension, e.g. <c>ggml-tiny.en</c>.</param>
    public static string WhisperCpp(string name) => Path.Combine(s_models, "whisper.cpp", name + ".bin");

    /// <summary>A device by name (<c>cpu</c> or <c>vulkan</c>), or null when it is not available.</summary>
    /// <param name="name">The device name.</param>
    public static IComputeDevice? Device(string name) => name == "cpu" ? s_cpu.Value : s_vulkan.Value;

    /// <summary>Opens a new device on the Vulkan provider (for callers that dispose it, such as the precompiler).</summary>
    /// <param name="options">The device options.</param>
    public static IComputeDevice OpenVulkan(DeviceOptions options)
    {
        var provider = new VulkanProvider();
        return provider.OpenDevice(provider.GetDevices()[0], options);
    }

    /// <summary>Reads a fixture file of the test assembly.</summary>
    /// <param name="name">The file name under Fixtures.</param>
    public static JsonDocument Fixture(string name) => JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", name)));

    /// <summary>True when the model and the audio exist and the device is available; otherwise the test is skipped by the caller.</summary>
    /// <param name="model">The model path.</param>
    /// <param name="device">The device name.</param>
    public static bool Available(string model, string device) => (File.Exists(model) || Directory.Exists(model)) && File.Exists(Jfk) && Device(device) is not null;
}
