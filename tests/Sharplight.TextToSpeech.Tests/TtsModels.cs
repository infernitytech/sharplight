using Microsoft.Extensions.DependencyInjection;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>Locates the downloaded text-to-speech models and opens devices; tests skip when a file is missing.</summary>
internal static class TtsModels
{
    private static readonly string s_root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "tts");

    /// <summary>OuteTTS 0.2 500M (Q8_0).</summary>
    public static string OuteTts => Path.Combine(s_root, "outetts-0.2", "OuteTTS-0.2-500M-Q8_0.gguf");

    /// <summary>The WavTokenizer decoder (F16).</summary>
    public static string WavTokenizer => Path.Combine(s_root, "wavtokenizer", "WavTokenizer-Large-75-F16.gguf");

    /// <summary>A fixture file next to the test assembly.</summary>
    public static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    /// <summary>Skips the test unless every file exists.</summary>
    public static void Require(params string[] paths)
    {
        foreach (string path in paths)
        {
            Assert.SkipUnless(File.Exists(path), $"{path} not present");
        }
    }

    /// <summary>Opens the CPU device or the first GPU device (skipping the test when there is none).</summary>
    public static DeviceSlot Open(bool gpu)
    {
        _ = new ServiceCollection().AddSharplight();
        foreach (var (provider, availability) in ProviderRegistry.Discover())
        {
            if (availability.IsAvailable && (provider.Name == "cpu") != gpu)
            {
                return new DeviceSlot { Device = provider.OpenDevice(provider.GetDevices()[0]) };
            }
        }

        Assert.Skip(gpu ? "No GPU provider is available" : "The CPU provider is not available");
        return null!;
    }

    /// <summary>Loads the OuteTTS pair as a task model.</summary>
    public static TaskModel LoadOuteTts(DeviceSlot slot, string? modelPath = null)
        => TaskModel.Load(new TaskLoadOptions
        {
            ModelPath = modelPath ?? OuteTts,
            Devices = [slot],
            Auxiliary = modelPath is null ? [("vocoder", WavTokenizer)] : [],
        });

    /// <summary>The speech synthesizer of a loaded model, from the container's factories.</summary>
    public static ISpeechSynthesizer Synthesizer(TaskModel model)
    {
        var services = new ServiceCollection().AddSharplight();
        using var provider = services.BuildServiceProvider();
        return provider.GetRequiredService<TaskClientProvider>().Create<ISpeechSynthesizer>(model);
    }
}
