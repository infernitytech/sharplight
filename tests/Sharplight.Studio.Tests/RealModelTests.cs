using System.Text.Json;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Server;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Studio's engine runtime on real small models (skipped when they are not on disk): two resident models with load progress, chat on
/// each, and the embedded server routing by the <c>model</c> field.
/// </summary>
public sealed class RealModelTests
{
    private static readonly string s_home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly string s_qwen = Path.Combine(s_home, "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");
    private static readonly string s_smol = Path.Combine(s_home, "models/gguf/SmolLM2-360M-Instruct/smollm2-360m-instruct-q8_0.gguf");

    private static ChatInput Ask(string text) => new([ChatInputMessage.FromText(ChatRole.User, text)], new SamplingSettings { Temperature = 0, MaxTokens = 24 }, false);

    private static async Task<(string Text, MessageStats? Stats)> ChatAsync(ILoadedModel model, string text)
    {
        var answer = "";
        MessageStats? stats = null;
        await foreach (var update in model.StreamChatAsync(Ask(text), TestContext.Current.CancellationToken))
        {
            answer += update.Kind == ChatStreamKind.Content ? update.Text : "";
            stats = update.Stats ?? stats;
        }

        return (answer, stats);
    }

    [Fact]
    public async Task TwoResidentModels_ChatAndServe()
    {
        Assert.SkipUnless(File.Exists(s_qwen) && File.Exists(s_smol), "small test models not found");
        using var runtime = new EngineModelRuntime(() => new ResidencySettings());
        Assert.SkipWhen(runtime.Devices.Count == 0, "no compute device");
        var device = runtime.Devices[0];
        var load = new LoadSettings { DeviceId = device.Id, ContextLength = 1024, BatchTokens = 128, Slots = 1 };
        var statuses = new List<LoadStatus>();
        var ct = TestContext.Current.CancellationToken;

        var qwen = await runtime.LoadAsync(new ModelLoadRequest("qwen", s_qwen, load, []), new SynchronousProgress<LoadStatus>(statuses.Add), ct);
        var smol = await runtime.LoadAsync(new ModelLoadRequest("smol", s_smol, load, []), null, ct);

        Assert.Equal(["qwen", "smol"], runtime.LoadedModels.Select(model => model.Id).Order());
        Assert.Contains(statuses, status => status.Text.Contains("Compiling", StringComparison.Ordinal) || status.Text.Contains("Uploading", StringComparison.Ordinal));
        Assert.True(statuses[^1].IsDone);
        Assert.True(qwen.Capabilities.HasFlag(ModelCapabilities.Reasoning));
        Assert.True(qwen.ResidentBytes > 0);
        Assert.Contains(runtime.GetDeviceUsage(), usage => usage.UsedBytes > 0);

        var recorded = new List<RequestRecord>();
        runtime.RequestCompleted += (_, record) =>
        {
            lock (recorded)
            {
                recorded.Add(record);
            }
        };
        var answers = await Task.WhenAll(ChatAsync(qwen, "What is the capital of France?"), ChatAsync(smol, "What is the capital of France?"));
        Assert.All(answers, answer => Assert.Contains("Paris", answer.Text, StringComparison.OrdinalIgnoreCase));
        Assert.All(answers, answer => Assert.True(answer.Stats?.CompletionTokens > 0));

        await using var server = new KestrelApiServer(runtime.Host);
        int port = 38000 + Random.Shared.Next(2000);
        await server.StartAsync(new ServerSettings { Port = port, Anthropic = false }, ct);
        using var http = new HttpClient { BaseAddress = new Uri(server.BaseUrl!) };
        var models = await http.GetFromJsonAsync<JsonElement>("/v1/models", ct);
        Assert.Equal(["qwen", "smol"], models.GetProperty("data").EnumerateArray().Select(model => model.GetProperty("id").GetString()).Order());

        server.SetServedModels(["smol"]);
        var hidden = await http.PostAsJsonAsync("/v1/chat/completions", new { model = "qwen", messages = new[] { new { role = "user", content = "hi" } }, max_tokens = 4 }, ct);
        Assert.False(hidden.IsSuccessStatusCode);
        var response = await http.PostAsJsonAsync("/v1/chat/completions", new { model = "smol", messages = new[] { new { role = "user", content = "Say hi" } }, max_tokens = 8, temperature = 0 }, ct);
        response.EnsureSuccessStatusCode();
        var completion = await response.Content.ReadFromJsonAsync<JsonElement>(ct);
        Assert.Equal("smol", completion.GetProperty("model").GetString());
        await server.StopAsync();

        // Stopping the server leaves the models usable for chat.
        Assert.Contains("Paris", (await ChatAsync(smol, "What is the capital of France?")).Text, StringComparison.OrdinalIgnoreCase);
        lock (recorded)
        {
            Assert.Contains(recorded, record => record.Model == "qwen");
            Assert.Contains(recorded, record => record.Model == "smol" && record.Source == RequestSource.Api);
        }

        await runtime.UnloadAsync("qwen");
        Assert.Equal(["smol"], runtime.LoadedModels.Select(model => model.Id));
    }
}
