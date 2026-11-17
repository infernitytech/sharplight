using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Models of the hot-model sweep (docs/model-support.md, "Hot-model sweep 2026-11-17") through Studio's own load path: library
/// recognition (<see cref="ModelInspector"/>), loading on the engine runtime and a greedy chat turn with the model's template. A hybrid
/// gated-delta-net model as Hugging Face safetensors (Qwen3.5-4B) and as GGUF, and Gemma 4 31B (GGUF). Each case skips when its files
/// are not on disk.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class HotSweepRealModelTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    public static TheoryData<string> Models =>
    [
        "hot-a/qwen3.5-4b",
        "hot-a/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf",
        "hot-a/gemma-4-31b-gguf/gemma-4-31B-it-Q4_K_M.gguf",
    ];

    [Theory]
    [MemberData(nameof(Models))]
    public async Task LoadsAndChats_ThroughTheStudioRuntime(string relativePath)
    {
        string path = Path.Combine(s_home, "models", relativePath);
        Assert.SkipUnless(File.Exists(path) || Directory.Exists(path), $"{relativePath} not found");
        using var runtime = new EngineModelRuntime(EngineServices.From(TestHost.Services), () => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var ct = TestContext.Current.CancellationToken;

        var inspected = ModelInspector.Inspect(path);
        Assert.Equal(ModelKinds.TextGeneration, inspected.Kind);
        var settings = new LoadSettings { DeviceId = runtime.Devices[0].Id, ContextLength = 1024, BatchTokens = 128, Slots = 1 };
        var model = await runtime.LoadAsync(new ModelLoadRequest("sweep", path, settings, []), null, ct);

        var input = new ChatInput([ChatInputMessage.FromText(ChatRole.User, "What is the capital of France? Answer in one word.")],
            new SamplingSettings { Temperature = 0, MaxTokens = 48 }, false);
        string answer = "";
        await foreach (var update in model.StreamChatAsync(input, ct))
        {
            if (update.Kind == ChatStreamKind.Content)
            {
                answer += update.Text;
            }
        }

        Assert.Contains("Paris", answer, StringComparison.OrdinalIgnoreCase);
        await runtime.UnloadAsync("sweep");
    }
}
