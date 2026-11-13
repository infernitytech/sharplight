using System.ComponentModel;
using Microsoft.Extensions.AI;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>Loads Qwen3.5-0.8B once (Vulkan if available, else CPU). Tests skip when the file is missing.</summary>
public sealed class QwenFixture : IDisposable
{
    public static readonly string Path = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf");
    public SharplightChatClient? Client { get; }
    private readonly DeviceSlot? _slot;

    public QwenFixture()
    {
        if (!File.Exists(Path))
        {
            return;
        }

        GgufImporter.Register();
        var vk = new VulkanProvider();
        if (vk.CheckAvailability().IsAvailable && vk.GetDevices().Count > 0)
        {
            var devs = vk.GetDevices();
            var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs[0];
            _slot = new DeviceSlot { Device = vk.OpenDevice(dev) };
        }
        else
        {
            var cpu = new CpuExecutionProvider();
            _slot = new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) };
        }
        Client = SharplightChatClient.Load(Path, new LlmLoadOptions { Devices = [_slot], ContextLength = 1024, MaxBatchTokens = 256 },
            new SharplightChatClientOptions { Sampling = SamplingOptionsGreedy });
    }

    private static SamplingOptions SamplingOptionsGreedy => SamplingOptions.Greedy;

    public void Dispose()
    {
        Client?.Dispose();
        _slot?.Device.Dispose();
    }
}

public class ModelTests(QwenFixture fx) : IClassFixture<QwenFixture>
{
    private SharplightChatClient Client
    {
        get
        {
            Assert.SkipUnless(fx.Client is not null, $"model not found: {QwenFixture.Path}");
            return fx.Client!;
        }
    }

    [Fact]
    public async Task ChatNonStreaming()
    {
        var r = await Client.GetResponseAsync("What is the capital of France? Answer in one word.", new ChatOptions { MaxOutputTokens = 12, Temperature = 0 });
        Assert.Contains("Paris", r.Text);
        Assert.True(r.Usage!.InputTokenCount > 5);
        Assert.True(r.Usage.OutputTokenCount > 0);
        Assert.NotNull(r.FinishReason);
    }

    [Fact]
    public async Task ChatStreamingMatchesNonStreamingAndTokenLimitIsReported()
    {
        var opts = new ChatOptions { MaxOutputTokens = 6, Temperature = 0 };
        var text = new System.Text.StringBuilder();
        int n = 0;
        ChatFinishReason? finish = null;
        await foreach (var u in Client.GetStreamingResponseAsync("Count from one to twenty.", opts))
        {
            text.Append(u.Text); n++;
            finish = u.FinishReason ?? finish;
        }
        var full = await Client.GetResponseAsync("Count from one to twenty.", opts);
        Assert.Equal(full.Text, text.ToString());
        Assert.True(n >= 2);
        Assert.Equal(ChatFinishReason.Length, finish);
    }

    [Fact]
    public async Task ToolCallIsRenderedIntoTemplateAndRoundTrips()
    {
        var weather = AIFunctionFactory.Create(([Description("City name")] string city) => $"Sunny, 21C in {city}", "get_weather", "Get the current weather for a city");
        var opts = new ChatOptions { Tools = [weather], MaxOutputTokens = 80, Temperature = 0 };
        var r = await Client.GetResponseAsync("What is the weather in Paris? Use the tool.", opts);
        // A 0.8B model may or may not call the tool; when it does, the call must be parsed.
        var call = r.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>().FirstOrDefault();
        if (call is not null)
        {
            Assert.Equal("get_weather", call.Name);
            Assert.Equal(ChatFinishReason.ToolCalls, r.FinishReason);
            // Feed a result back: the second turn must render (assistant tool call + tool response) and generate.
            var history = new List<ChatMessage> { new(ChatRole.User, "What is the weather in Paris? Use the tool.") };
            history.AddRange(r.Messages);
            history.Add(new ChatMessage(ChatRole.Tool, [new FunctionResultContent(call.CallId, "Sunny, 21C in Paris")]));
            var r2 = await Client.GetResponseAsync(history, new ChatOptions { Tools = [weather], MaxOutputTokens = 32, Temperature = 0 });
            Assert.True(r2.Usage!.InputTokenCount > 0);
        }
        else
        {
            Assert.False(string.IsNullOrEmpty(r.Text));
        }
    }
}
