using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Serving.Tests;

/// <summary>End-to-end on a real small model (skipped when the file is absent): engine + chat layer on SessionBatchDecoder.</summary>
public class RealModelSmokeTests
{
    private static readonly string ModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf");

    [Fact]
    public async Task Qwen3_Chat_Streams_Stops_And_ParsesToolCall()
    {
        Assert.SkipUnless(File.Exists(ModelPath), "model not found: " + ModelPath);
        ProviderRegistry.Register(() => new VulkanProvider());
        CpuExecutionProvider.Register();
        GgufImporter.Register();
        var avail = ProviderRegistry.Discover().Where(x => x.Availability.IsAvailable).ToList();
        var provider = avail.Where(x => x.Provider.Name != "cpu").Concat(avail).First().Provider;
        using var device = provider.OpenDevice(provider.GetDevices()[0], new DeviceOptions());
        using var model = LlmModel.Load(ModelPath, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 2048, MaxBatchTokens = 256 });
        using var engine = new InferenceEngine(new SessionBatchDecoder(model.CreateSession()), model.Tokenizer);
        var svc = new ModelService(engine, model.ChatTemplate, new ModelServiceOptions { ModelName = "qwen3" });

        // Greedy, thinking off: a short deterministic answer that mentions Paris; the same request twice gives identical output.
        ChatRequest Ask(string q, IReadOnlyList<string>? stop = null) => new()
        {
            Messages = [new ChatTurn("user", q)], Sampling = SamplingParams.GreedyDefault, MaxTokens = 40, EnableThinking = false, StopStrings = stop ?? [],
        };
        var a = await svc.StartChat(Ask("What is the capital of France? Answer in one word.")).CompleteAsync();
        Assert.Contains("Paris", a.Choices[0].Content);
        Assert.Equal(ChatFinishReason.Stop, a.Choices[0].Finish);
        var b = await svc.StartChat(Ask("What is the capital of France? Answer in one word.")).CompleteAsync();
        Assert.Equal(a.Choices[0].Content, b.Choices[0].Content);

        // Streaming deltas concatenate to the non-streaming text, and a stop string cuts the output.
        var run = svc.StartChat(Ask("Count from 1 to 10, separated by commas.", ["5"]));
        var text = new System.Text.StringBuilder();
        await foreach (var e in run.ReadAllAsync())
        {
            if (e.Content is not null)
            {
                text.Append(e.Content);
            }
        }

        Assert.Contains("1, 2, 3, 4", text.ToString());
        Assert.DoesNotContain("5", text.ToString());

        // Two concurrent requests are served (serialized on one slot) and both finish.
        var both = await Task.WhenAll(svc.StartChat(Ask("Say hello.")).CompleteAsync(), svc.StartChat(Ask("Say goodbye.")).CompleteAsync());
        Assert.All(both, r => Assert.NotEmpty(r.Choices[0].Content));
        Assert.True(engine.GetMetrics().RequestsFinished[FinishReason.Stop] >= 4);

        // Tool calling in the Hermes format the model's template instructs.
        var tools = new ToolSpec("get_weather", "Get the current weather for a city", """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");
        var t = await svc.StartChat(new ChatRequest
        {
            Messages = [new ChatTurn("user", "What's the weather in Paris?")], Tools = [tools], Sampling = SamplingParams.GreedyDefault, MaxTokens = 100, EnableThinking = false,
        }).CompleteAsync();
        var call = Assert.Single(t.Choices[0].ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Contains("Paris", call.ArgumentsJson);
        Assert.Equal(ChatFinishReason.ToolCalls, t.Choices[0].Finish);
    }
}
