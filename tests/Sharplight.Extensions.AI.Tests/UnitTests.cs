using System.ComponentModel;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI.Tests;

public class OutputParserTests
{
    private static (List<AIContent> All, OutputParser P) Run(OutputParser p, params string[] pieces)
    {
        var all = new List<AIContent>();
        foreach (var s in pieces)
        {
            all.AddRange(p.Push(s));
        }

        all.AddRange(p.Complete());
        return (all, p);
    }

    private static string Text(IEnumerable<AIContent> c) => string.Concat(c.OfType<TextContent>().Select(t => t.Text));
    private static string Reasoning(IEnumerable<AIContent> c) => string.Concat(c.OfType<TextReasoningContent>().Select(t => t.Text));

    private static OutputParser Plain(bool reasoning = false, string[]? stops = null)
        => new(reasoning, stops, null, null, null, null);

    private static OutputParser WithTools(params AITool[] tools)
        => new(false, null, "<tool_call>", "</tool_call>", [new JsonToolCallParser(), new XmlToolCallParser()], new ToolSchemas(tools));

    [Fact]
    public void PlainTextPassesThroughAndTrims()
    {
        var (all, _) = Run(Plain(), "  Hello", ", wor", "ld!\n\n");
        Assert.Equal("Hello, world!", Text(all));
    }

    [Fact]
    public void StopSequenceIsTrimmedEvenAcrossPieces()
    {
        var (all, p) = Run(Plain(stops: ["END"]), "abc E", "N", "D tail");
        Assert.Equal("abc", Text(all));
        Assert.True(p.Stopped);
    }

    [Fact]
    public void PartialStopPrefixIsFlushedWhenStreamEnds()
    {
        var (all, p) = Run(Plain(stops: ["END"]), "abc EN");
        Assert.Equal("abc EN", Text(all));
        Assert.False(p.Stopped);
    }

    [Fact]
    public void ReasoningIsSplitFromAnswer()
    {
        var (all, _) = Run(Plain(), "<thi", "nk>\nLet me think.\n</think>\n\nThe answer is 4.");
        Assert.Equal("Let me think.", Reasoning(all));
        Assert.Equal("The answer is 4.", Text(all));
    }

    [Fact]
    public void ReasoningStartingInPrompt()
    {
        var (all, _) = Run(Plain(reasoning: true), "hmm</think>\n\nok");
        Assert.Equal("hmm", Reasoning(all));
        Assert.Equal("ok", Text(all));
    }

    [Fact]
    public void QwenXmlToolCall()
    {
        var tool = AIFunctionFactory.Create(([Description("City")] string city, int days) => "x", "get_weather", "Weather");
        var (all, p) = Run(WithTools(tool), "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>");
        Assert.Equal("Let me check.", Text(all));
        var call = Assert.Single(all.OfType<FunctionCallContent>());
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("New York", call.Arguments!["city"]);
        Assert.Equal(3, ((JsonElement)call.Arguments["days"]!).GetInt32());
        Assert.Equal(1, p.ToolCallCount);
    }

    [Fact]
    public void StringParameterStaysStringEvenIfNumeric()
    {
        var tool = AIFunctionFactory.Create((string zip) => "x", "lookup");
        var (all, _) = Run(WithTools(tool), "<tool_call>\n<function=lookup>\n<parameter=zip>\n10001\n</parameter>\n</function>\n</tool_call>");
        Assert.Equal("10001", Assert.Single(all.OfType<FunctionCallContent>()).Arguments!["zip"]);
    }

    [Fact]
    public void HermesJsonToolCallsInSplitPieces()
    {
        var (all, _) = Run(WithTools(), "<tool_ca", "ll>\n{\"name\": \"add\", \"argum", "ents\": {\"a\": 1, \"b\": \"x\"}}\n</tool_call>\n<tool_call>{\"name\":\"two\",\"arguments\":{}}</tool_call>");
        var calls = all.OfType<FunctionCallContent>().ToList();
        Assert.Equal(["add", "two"], calls.Select(c => c.Name));
        Assert.Equal(1, ((JsonElement)calls[0].Arguments!["a"]!).GetInt32());
        Assert.NotEqual(calls[0].CallId, calls[1].CallId);
    }

    [Fact]
    public void ToolMarkersAreTextWhenNoToolsOffered()
    {
        var (all, _) = Run(Plain(), "<tool_call>x</tool_call>");
        Assert.Equal("<tool_call>x</tool_call>", Text(all));
    }

    [Fact]
    public void UnparsableToolCallBecomesText()
    {
        var (all, _) = Run(WithTools(), "<tool_call>garbage</tool_call>");
        Assert.Empty(all.OfType<FunctionCallContent>());
        Assert.Contains("garbage", Text(all));
    }

    [Fact]
    public void TruncatedToolCallIsStillParsedAtEnd()
    {
        var (all, _) = Run(WithTools(), "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>");
        Assert.Equal("f", Assert.Single(all.OfType<FunctionCallContent>()).Name);
    }
}

public class PromptMappingTests
{
    private static readonly SharplightChatClientOptions Opts = new();

    [Fact]
    public void ChatOptionsMapToSampling()
    {
        var g = PromptBuilder.MapOptions(new ChatOptions { Temperature = 0.2f, TopP = 0.5f, TopK = 7, Seed = 42, MaxOutputTokens = 33, StopSequences = ["a", "b"] }, Opts);
        Assert.Equal(0.2f, g.Sampling.Temperature);
        Assert.Equal(0.5f, g.Sampling.TopP);
        Assert.Equal(7, g.Sampling.TopK);
        Assert.Equal(42, g.Sampling.Seed);
        Assert.Equal(33, g.MaxNewTokens);
        Assert.Equal(["a", "b"], g.StopStrings);
    }

    [Fact]
    public void DefaultsComeFromClientOptions()
    {
        var o = new SharplightChatClientOptions { DefaultMaxOutputTokens = 99, Sampling = new() { Temperature = 0.3f } };
        var g = PromptBuilder.MapOptions(null, o);
        Assert.Equal(99, g.MaxNewTokens);
        Assert.Equal(0.3f, g.Sampling.Temperature);
    }

    [Fact]
    public void SystemInstructionsAreMergedAndHoisted()
    {
        var msgs = PromptBuilder.MapMessages(
            [new(ChatRole.User, "hi"), new(ChatRole.System, "be brief"), new(ChatRole.Assistant, "hello")],
            new ChatOptions { Instructions = "Be nice." }, Opts, false, []);
        Assert.Equal(["system", "user", "assistant"], msgs.Select(m => m.Role));
        Assert.Equal("Be nice.\n\nbe brief", msgs[0].Content);
    }

    [Fact]
    public void FunctionCallsAndResults()
    {
        var args = new Dictionary<string, object?> { ["city"] = JsonDocument.Parse("\"Paris\"").RootElement, ["n"] = JsonDocument.Parse("2").RootElement };
        var msgs = PromptBuilder.MapMessages(
        [
            new(ChatRole.User, "weather?"),
            new(ChatRole.Assistant, [new FunctionCallContent("c1", "get_weather", args)]),
            new(ChatRole.Tool, [new FunctionResultContent("c1", "sunny")]),
        ], null, Opts, false, []);
        Assert.Equal(["user", "assistant", "tool"], msgs.Select(m => m.Role));
        Assert.Null(msgs[1].Content);
        var call = Assert.Single(msgs[1].ToolCalls!);
        Assert.Equal("Paris", call.Arguments!["city"]);
        Assert.Equal(2L, call.Arguments["n"]);
        Assert.Equal("sunny", msgs[2].Content);
        Assert.Equal("c1", msgs[2].ToolCallId);
    }

    [Fact]
    public void ImagesRequireVisionSupport()
    {
        var img = new DataContent(new byte[] { 1, 2, 3 }, "image/png");
        var ex = Assert.Throws<NotSupportedException>(() => PromptBuilder.MapMessages([new(ChatRole.User, [new TextContent("what?"), img])], null, Opts, false, []));
        Assert.Contains("vision", ex.Message);
    }

    [Fact]
    public void ImagesBecomePartsInOrder()
    {
        var o = new SharplightChatClientOptions { ImageDecoder = _ => new(1, 1, [1, 2, 3]) };
        var images = new List<Sharplight.Vision.ImageInput>();
        var msgs = PromptBuilder.MapMessages([new(ChatRole.User, [new DataContent(new byte[] { 0 }, "image/png"), new TextContent("what?")])], null, o, true, images);
        Assert.Single(images);
        Assert.Equal(["image", "text"], msgs[0].Parts!.Select(p => p.Type));
    }

    [Fact]
    public void ToolDefinitionsCarryNameDescriptionSchema()
    {
        var tool = AIFunctionFactory.Create(([Description("The city")] string city) => "x", "get_weather", "Gets weather");
        var def = Assert.Single(PromptBuilder.MapTools([tool]));
        Assert.Equal("get_weather", def.Name);
        Assert.Equal("Gets weather", def.Description);
        Assert.True(def.Parameters!.ContainsKey("properties"));
    }
}

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
