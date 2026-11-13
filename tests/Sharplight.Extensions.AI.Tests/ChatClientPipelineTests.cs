using System.ComponentModel;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI.Tests;

public class ChatClientPipelineTests
{
    private static ChatOptions Weather() => new() { Tools = [AIFunctionFactory.Create(([Description("City")] string city) => "sunny", "get_weather", "Weather")] };

    [Fact]
    public async Task NonStreamingReturnsTextUsageAndFinishReason()
    {
        using var backend = new FakeBackend("Hel", "lo ", "there");
        using var client = new SharplightChatClient(backend);
        var r = await client.GetResponseAsync("hi", new ChatOptions { Temperature = 0, MaxOutputTokens = 8 });
        Assert.Equal("Hello there", r.Text);
        Assert.Equal(ChatFinishReason.Stop, r.FinishReason);
        Assert.Equal(11, r.Usage!.InputTokenCount);
        Assert.Equal(5, r.Usage.OutputTokenCount);
        Assert.Equal("fake", r.ModelId);
        Assert.Contains("<|im_start|>user\nhi<|im_end|>", backend.Last!.Prompt);
        Assert.EndsWith("<|im_start|>assistant\n", backend.Last.Prompt);
        Assert.Equal(8, backend.Last.Options.MaxNewTokens);
        Assert.Equal(0, backend.Last.Options.Sampling.Temperature);
    }

    [Fact]
    public async Task StreamingYieldsIncrementalUpdatesThenUsage()
    {
        using var backend = new FakeBackend("a", "b", "c") { ReachedLimit = true };
        using var client = new SharplightChatClient(backend);
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync("x"))
        {
            updates.Add(u);
        }

        Assert.Equal(["a", "b", "c"], updates.Where(u => u.Text.Length > 0).Select(u => u.Text));
        Assert.Contains(updates.Last().Contents, c => c is UsageContent);
        Assert.Equal(ChatFinishReason.Length, updates.Last().FinishReason);
    }

    [Fact]
    public async Task ToolCallsAreParsedAndFinishReasonIsToolCalls()
    {
        using var backend = new FakeBackend("<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>");
        using var client = new SharplightChatClient(backend);
        var r = await client.GetResponseAsync("weather in Berlin?", Weather());
        var call = Assert.Single(r.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Berlin", call.Arguments!["city"]);
        Assert.Equal(ChatFinishReason.ToolCalls, r.FinishReason);
        Assert.Contains("# Tools", backend.Last!.Prompt);
        Assert.Contains("\"get_weather\"", backend.Last.Prompt);
    }

    [Fact]
    public async Task FunctionInvokingChatClientRunsTheToolLoop()
    {
        var weather = AIFunctionFactory.Create(([Description("City")] string city) => $"sunny in {city}", "get_weather", "Weather");
        using var inner = new ScriptedClient(i => i == 0
            ? "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>"
            : "It is sunny in Rome.");
        using var client = inner.AsBuilder().UseFunctionInvocation().Build();
        var r = await client.GetResponseAsync("weather?", new ChatOptions { Tools = [weather] });
        Assert.Equal("It is sunny in Rome.", r.Text);
        Assert.Equal(2, inner.Prompts.Count);
        Assert.Contains("[id:", inner.Prompts[1]);
        Assert.Contains("sunny in Rome", inner.Prompts[1]);
    }

    [Fact]
    public async Task ThinkingIsReturnedAsReasoningContent()
    {
        using var backend = new FakeBackend("hmm\n</think>\n\n42");
        using var client = new SharplightChatClient(backend);
        var r = await client.GetResponseAsync("q", new ChatOptions { AdditionalProperties = new() { ["enable_thinking"] = true } });
        Assert.EndsWith("<think>\n", backend.Last!.Prompt);
        Assert.Equal("42", r.Text);
        Assert.Equal("hmm", r.Messages.SelectMany(m => m.Contents).OfType<TextReasoningContent>().Single().Text);
    }

    [Fact]
    public async Task StopSequencesTrimAndStop()
    {
        using var backend = new FakeBackend("one two ", "STOP three");
        using var client = new SharplightChatClient(backend);
        var r = await client.GetResponseAsync("q", new ChatOptions { StopSequences = ["STOP"] });
        Assert.Equal("one two", r.Text);
        Assert.Equal(ChatFinishReason.Stop, r.FinishReason);
    }

    [Fact]
    public async Task JsonResponseFormatAddsInstruction()
    {
        using var backend = new FakeBackend("{}");
        using var client = new SharplightChatClient(backend);
        await client.GetResponseAsync("q", new ChatOptions { ResponseFormat = ChatResponseFormat.Json });
        Assert.Contains("valid JSON", backend.Last!.Prompt);
    }

    [Fact]
    public async Task CancellationIsHonored()
    {
        using var backend = new FakeBackend("a", "b");
        using var client = new SharplightChatClient(backend);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await client.GetResponseAsync("x", cancellationToken: cts.Token));
    }

    /// <summary>An IChatClient (through the real pipeline) whose model output is scripted per call.</summary>
    private sealed class ScriptedClient(Func<int, string> script) : IChatClient
    {
        public List<string> Prompts { get; } = [];

        public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            var backend = new FakeBackend(script(Prompts.Count));
            using var c = new SharplightChatClient(backend);
            var r = await c.GetResponseAsync(messages, options, cancellationToken);
            Prompts.Add(backend.Last!.Prompt);
            return r;
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
            => throw new NotSupportedException();
        public object? GetService(Type serviceType, object? serviceKey = null) => null;
        public void Dispose() { }
    }
}
