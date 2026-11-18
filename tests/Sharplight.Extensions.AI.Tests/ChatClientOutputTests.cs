using System.ComponentModel;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI.Tests;

/// <summary>
/// How <see cref="SharplightChatClient"/> splits model output (through the shared <c>ChatOutputParser</c> of Core): stop strings,
/// reasoning, tool calls and whitespace, with the output arriving in arbitrary pieces.
/// </summary>
public class ChatClientOutputTests
{
    private static async Task<(List<AIContent> All, ChatFinishReason? Finish)> RunAsync(ChatOptions? options, bool thinking, params string[] pieces)
    {
        using var backend = new FakeBackend(pieces);
        using var client = new SharplightChatClient(backend, new SharplightChatClientOptions { EnableThinking = thinking });
        var all = new List<AIContent>();
        ChatFinishReason? finish = null;
        await foreach (var update in client.GetStreamingResponseAsync("q", options, TestContext.Current.CancellationToken))
        {
            all.AddRange(update.Contents.Where(content => content is not UsageContent));
            finish = update.FinishReason ?? finish;
        }

        return (all, finish);
    }

    private static string Text(IEnumerable<AIContent> contents) => string.Concat(contents.OfType<TextContent>().Select(t => t.Text));

    private static string Reasoning(IEnumerable<AIContent> contents) => string.Concat(contents.OfType<TextReasoningContent>().Select(t => t.Text));

    private static ChatOptions Tools(params AITool[] tools) => new() { Tools = tools };

    [Fact]
    public async Task PlainTextPassesThroughWithoutLeadingWhitespace()
    {
        var (all, _) = await RunAsync(null, false, "  Hello", ", wor", "ld!\n\n");
        Assert.Equal("Hello, world!\n\n", Text(all));
    }

    [Fact]
    public async Task StopSequenceIsTrimmedEvenAcrossPieces()
    {
        var (all, finish) = await RunAsync(new ChatOptions { StopSequences = ["END"] }, false, "abc E", "N", "D tail");
        Assert.Equal("abc ", Text(all));
        Assert.Equal(ChatFinishReason.Stop, finish);
    }

    [Fact]
    public async Task PartialStopPrefixIsFlushedWhenStreamEnds()
    {
        var (all, _) = await RunAsync(new ChatOptions { StopSequences = ["END"] }, false, "abc EN");
        Assert.Equal("abc EN", Text(all));
    }

    [Fact]
    public async Task ReasoningIsSplitFromAnswer()
    {
        var (all, _) = await RunAsync(null, false, "<thi", "nk>\nLet me think.\n</think>\n\nThe answer is 4.");
        Assert.Equal("Let me think.", Reasoning(all));
        Assert.Equal("The answer is 4.", Text(all));
    }

    [Fact]
    public async Task ReasoningStartingInPrompt()
    {
        var (all, _) = await RunAsync(null, true, "hmm</", "think>\n\nok");
        Assert.Equal("hmm", Reasoning(all));
        Assert.Equal("ok", Text(all));
    }

    [Fact]
    public async Task QwenXmlToolCall()
    {
        var tool = AIFunctionFactory.Create(([Description("City")] string city, int days) => "x", "get_weather", "Weather");
        var (all, finish) = await RunAsync(Tools(tool), false,
            "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>");
        Assert.Equal("Let me check.", Text(all));
        var call = Assert.Single(all.OfType<FunctionCallContent>());
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("New York", call.Arguments!["city"]);
        Assert.Equal(3, ((JsonElement)call.Arguments["days"]!).GetInt32());
        Assert.Equal(ChatFinishReason.ToolCalls, finish);
    }

    [Fact]
    public async Task StringParameterStaysStringEvenIfNumeric()
    {
        var tool = AIFunctionFactory.Create((string zip) => "x", "lookup");
        var (all, _) = await RunAsync(Tools(tool), false, "<tool_call>\n<function=lookup>\n<parameter=zip>\n10001\n</parameter>\n</function>\n</tool_call>");
        Assert.Equal("10001", Assert.Single(all.OfType<FunctionCallContent>()).Arguments!["zip"]);
    }

    [Fact]
    public async Task HermesJsonToolCallsInSplitPieces()
    {
        var add = AIFunctionFactory.Create((int a, string b) => "x", "add");
        var two = AIFunctionFactory.Create(() => "x", "two");
        var (all, _) = await RunAsync(Tools(add, two), false,
            "<tool_ca", "ll>\n{\"name\": \"add\", \"argum", "ents\": {\"a\": 1, \"b\": \"x\"}}\n</tool_call>\n<tool_call>{\"name\":\"two\",\"arguments\":{}}</tool_call>");
        var calls = all.OfType<FunctionCallContent>().ToList();
        Assert.Equal(["add", "two"], calls.Select(c => c.Name));
        Assert.Equal(1, ((JsonElement)calls[0].Arguments!["a"]!).GetInt32());
        Assert.NotEqual(calls[0].CallId, calls[1].CallId);
    }

    [Fact]
    public async Task ToolMarkersAreTextWhenNoToolsOffered()
    {
        var (all, _) = await RunAsync(null, false, "<tool_call>x</tool_call>");
        Assert.Equal("<tool_call>x</tool_call>", Text(all));
    }

    [Fact]
    public async Task CallsToToolsThatWereNotOfferedStayText()
    {
        var tool = AIFunctionFactory.Create(() => "x", "offered");
        var (all, _) = await RunAsync(Tools(tool), false, "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>");
        Assert.Empty(all.OfType<FunctionCallContent>());
        Assert.Contains("other", Text(all));
    }

    [Fact]
    public async Task UnparsableToolCallBecomesText()
    {
        var tool = AIFunctionFactory.Create(() => "x", "f");
        var (all, _) = await RunAsync(Tools(tool), false, "<tool_call>garbage</tool_call>");
        Assert.Empty(all.OfType<FunctionCallContent>());
        Assert.Contains("garbage", Text(all));
    }

    [Fact]
    public async Task TruncatedToolCallIsStillParsedAtEnd()
    {
        var tool = AIFunctionFactory.Create((int a) => "x", "f");
        var (all, _) = await RunAsync(Tools(tool), false, "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>");
        Assert.Equal("f", Assert.Single(all.OfType<FunctionCallContent>()).Name);
    }
}
