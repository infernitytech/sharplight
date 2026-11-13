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
