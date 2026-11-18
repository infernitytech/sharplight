using Microsoft.Extensions.DependencyInjection;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Speech;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>Locates the downloaded text-to-speech models and opens devices; tests skip when a file is missing.</summary>
internal static class TtsModels
{
    private static readonly Lazy<ServiceProvider> s_services = new(() => SharplightDefaults.CreateServiceProvider());
    private static readonly string s_root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "tts");

    /// <summary>The container with <c>services.AddSharplight()</c>, built once per test assembly.</summary>
    public static IServiceProvider Services => s_services.Value;

    /// <summary>OuteTTS 0.2 500M (Q8_0).</summary>
    public static string OuteTts => Path.Combine(s_root, "outetts-0.2", "OuteTTS-0.2-500M-Q8_0.gguf");

    /// <summary>The WavTokenizer decoder (F16).</summary>
    public static string WavTokenizer => Path.Combine(s_root, "wavtokenizer", "WavTokenizer-Large-75-F16.gguf");

    /// <summary>The Kokoro-82M checkpoint directory.</summary>
    public static string Kokoro => Path.Combine(s_root, "kokoro");

    /// <summary>Reference tensors of the kokoro Python package (raw little-endian arrays, see <c>KokoroTests</c>).</summary>
    public static string KokoroReference => Path.Combine(s_root, "kokoro-ref");

    /// <summary>Orpheus 3B (the checkpoint directory).</summary>
    public static string Orpheus => Path.Combine(s_root, "orpheus-3b");

    /// <summary>The SNAC 24 kHz codec checkpoint directory.</summary>
    public static string Snac => Path.Combine(s_root, "snac-24khz");

    /// <summary>Reference tensors of the snac Python package (raw arrays).</summary>
    public static string SnacReference => Path.Combine(s_root, "snac-ref");

    /// <summary>Reference tokens and logits of transformers' Orpheus (raw arrays).</summary>
    public static string OrpheusReference => Path.Combine(s_root, "orpheus-ref");

    /// <summary>Qwen3-TTS 12 Hz 0.6B CustomVoice (the checkpoint directory, with its codec in <c>speech_tokenizer/</c>).</summary>
    public static string Qwen3Tts => Path.Combine(s_root, "Qwen3-TTS-12Hz-0.6B-CustomVoice");

    /// <summary>Reference prompt, codes and logits of the official qwen_tts package (raw arrays, see <c>Qwen3TtsTests</c>).</summary>
    public static string Qwen3TtsReference => Path.Combine(s_root, "qwen3-tts-ref");

    /// <summary>CSM-1B (the transformers checkpoint directory, with the Mimi codec bundled).</summary>
    public static string Csm => Path.Combine(s_root, "csm-1b");

    /// <summary>Reference ids, frames and logits of transformers' CSM (raw arrays, see <c>CsmTests</c>).</summary>
    public static string CsmReference => Path.Combine(s_root, "csm-ref");

    /// <summary>VibeVoice-1.5B (Microsoft's checkpoint directory with the Qwen2.5 <c>tokenizer.json</c> next to it).</summary>
    public static string VibeVoice => Path.Combine(s_root, "..", "vibevoice", "VibeVoice-1.5B");

    /// <summary>Reference tensors of transformers' VibeVoice (raw arrays, see <c>VibeVoiceTests</c>).</summary>
    public static string VibeVoiceReference => Path.Combine(s_root, "..", "vibevoice", "ref");

    /// <summary>OmniVoice (the checkpoint directory, with its audio tokenizer in <c>audio_tokenizer/</c>).</summary>
    public static string OmniVoice => Path.Combine(s_root, "omnivoice");

    /// <summary>Reference tokens, logits and audio of the official omnivoice package (raw arrays, see <c>OmniVoiceTests</c>).</summary>
    public static string OmniVoiceReference => Path.Combine(s_root, "omnivoice-ref");

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

    private static readonly Lazy<DeviceSlot?> s_cpu = new(() => OpenFirst(gpu: false));
    private static readonly Lazy<DeviceSlot?> s_gpu = new(() => OpenFirst(gpu: true));

    /// <summary>
    /// The CPU device or the first GPU device, opened once and shared by every test of the assembly (skipping the test when there is
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
        foreach (var (provider, availability) in Services.GetRequiredService<ExecutionProviderCatalog>().Discover())
        {
            if (availability.IsAvailable && (provider.Name == "cpu") != gpu)
            {
                return new DeviceSlot { Device = provider.OpenDevice(provider.GetDevices()[0]) };
            }
        }

        return null;
    }

    /// <summary>Loads the OuteTTS pair as a task model.</summary>
    public static TaskModel LoadOuteTts(DeviceSlot slot, string? modelPath = null)
    {
        var options = new TaskLoadOptions { ModelPath = modelPath ?? OuteTts, Devices = [slot] };
        if (modelPath is null)
        {
            options.Auxiliary[AuxiliaryRoles.Vocoder] = WavTokenizer;
        }

        return TaskModel.Load(Services.GetRequiredService<ModelLoader>(), options);
    }

    /// <summary>Loads a model directory or file with its own importer as a task model.</summary>
    public static TaskModel Load(DeviceSlot slot, string path)
        => TaskModel.Load(Services.GetRequiredService<ModelLoader>(), new TaskLoadOptions { ModelPath = path, Devices = [slot] });

    /// <summary>Loads Orpheus with its SNAC decoder as a task model.</summary>
    public static TaskModel LoadOrpheus(DeviceSlot slot)
    {
        var options = new TaskLoadOptions { ModelPath = Orpheus, Devices = [slot] };
        options.Auxiliary[AuxiliaryRoles.Vocoder] = Snac;
        return TaskModel.Load(Services.GetRequiredService<ModelLoader>(), options);
    }

    /// <summary>The speech synthesizer of a loaded model, from the container's factories.</summary>
    public static ISpeechSynthesizer Synthesizer(TaskModel model)
        => Services.GetRequiredService<TaskClientProvider>().Create<ISpeechSynthesizer>(model);
}
