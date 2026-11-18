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

    /// <summary>The name of the JFK sample (11 s, 16 kHz mono).</summary>
    public const string JfkName = "jfk";

    /// <summary>The name of the longer recording: LibriSpeech dev-clean 1272-128104-0004 (29.4 s, 16 kHz mono, CC BY 4.0).</summary>
    public const string LibriSpeechName = "librispeech-1272-128104-0004";

    /// <summary>The JFK sample (11 s, 16 kHz mono).</summary>
    public static string Jfk => Audio(JfkName);

    /// <summary>A recording under <c>~/models/audio</c>.</summary>
    /// <param name="name">The file name without the <c>.wav</c> extension.</param>
    public static string Audio(string name) => Path.Combine(s_models, "audio", name + ".wav");

    /// <summary>A Hugging Face checkpoint directory under <c>~/models/hf</c>.</summary>
    /// <param name="name">The directory name, e.g. <c>whisper-tiny</c>.</param>
    public static string HuggingFace(string name) => Path.Combine(s_models, "hf", name);

    /// <summary>A speech GGUF file under <c>~/models/speech-gguf</c> (transcribe.cpp layout).</summary>
    /// <param name="name">The file name, e.g. <c>parakeet-tdt-0.6b-v3-F16.gguf</c>.</param>
    /// <summary>A checkpoint folder of the hot-model sweep (<c>models/hot-b</c>).</summary>
    public static string HotModel(string name) => Path.Combine(s_models, "hot-b", name);

    public static string SpeechGguf(string name) => Path.Combine(s_models, "speech-gguf", name);

    /// <summary>A file or folder under <c>~/models/vibevoice</c> (the Microsoft checkpoints and the reference tensors of the VibeVoice tests).</summary>
    /// <param name="relative">The path below the folder.</param>
    public static string VibeVoice(string relative) => Path.Combine(s_models, "vibevoice", relative);

    /// <summary>A whisper.cpp model file under <c>~/models/whisper.cpp</c>.</summary>
    /// <param name="name">The name without extension, e.g. <c>ggml-tiny.en</c>.</param>
    public static string WhisperCpp(string name) => Path.Combine(s_models, "whisper.cpp", name + ".bin");

    /// <summary>A device by name (<c>cpu</c> or <c>vulkan</c>), or null when it is not available.</summary>
    /// <param name="name">The device name.</param>
    public static IComputeDevice? Device(string name) => name == "cpu" ? s_cpu.Value : s_vulkan.Value;

    /// <summary>Reads a fixture file of the test assembly.</summary>
    /// <param name="name">The file name under Fixtures.</param>
    public static JsonDocument Fixture(string name) => JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", name)));

    /// <summary>The largest difference to sampled reference values (<c>count</c>, <c>stride</c>, <c>values</c>), and the largest reference magnitude.</summary>
    /// <param name="actual">The full tensor.</param>
    /// <param name="reference">The fixture's sample of the reference tensor.</param>
    /// <param name="scale">The largest reference magnitude.</param>
    public static double MaxError(float[] actual, JsonElement reference, out double scale)
    {
        Xunit.Assert.Equal(reference.GetProperty("count").GetInt32(), actual.Length);
        int stride = reference.GetProperty("stride").GetInt32();
        double error = 0;
        scale = 0;
        int index = 0;
        foreach (var value in reference.GetProperty("values").EnumerateArray())
        {
            double expected = value.GetDouble();
            error = Math.Max(error, Math.Abs(actual[index] - expected));
            scale = Math.Max(scale, Math.Abs(expected));
            index += stride;
        }

        return error;
    }

    /// <summary>True when the model and the audio exist and the device is available; otherwise the test is skipped by the caller.</summary>
    /// <param name="model">The model path.</param>
    /// <param name="device">The device name.</param>
    public static bool Available(string model, string device) => (File.Exists(model) || Directory.Exists(model)) && File.Exists(Jfk) && Device(device) is not null;
}
