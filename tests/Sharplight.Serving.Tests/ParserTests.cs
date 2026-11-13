using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Serving.Tests;

public class ParserTests
{
    static readonly string[] Tools = ["get_weather", "add"];

    /// <summary>Feeds <paramref name="text"/> in pieces of <paramref name="step"/> chars and returns all pieces.</summary>
    static List<ParsedPiece> Parse(string text, int step, IReadOnlyList<IToolCallFormat>? formats = null, bool reasoning = false, string[]? tools = null)
    {
        var p = new ChatOutputParser(formats ?? ToolCallFormats.All, tools ?? Tools, reasoning);
        var o = new List<ParsedPiece>();
        for (int i = 0; i < text.Length; i += step) o.AddRange(p.Push(text.Substring(i, Math.Min(step, text.Length - i))));
        o.AddRange(p.Finish());
        return o;
    }

    static string Join(List<ParsedPiece> ps, PieceKind k) => string.Concat(ps.Where(p => p.Kind == k).Select(p => p.Text));
    static List<ToolCall> Calls(List<ParsedPiece> ps) => [.. ps.Where(p => p.Kind == PieceKind.ToolCall).Select(p => p.Call!)];

    [Theory]
    [InlineData(1)] [InlineData(3)] [InlineData(1000)]
    public void Hermes_JsonToolCall(int step)
    {
        var ps = Parse("Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", step);
        Assert.Equal("Let me check.", Join(ps, PieceKind.Content));
        var c = Assert.Single(Calls(ps));
        Assert.Equal("get_weather", c.Name);
        Assert.Equal("""{"city": "Paris"}""", c.ArgumentsJson);
        Assert.StartsWith("call_", c.Id);
    }

    [Theory]
    [InlineData(1)] [InlineData(7)] [InlineData(1000)]
    public void Hermes_ParallelCalls(int step)
    {
        var ps = Parse("<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call>\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"X\"}}</tool_call>", step);
        Assert.Equal(["add", "get_weather"], Calls(ps).Select(c => c.Name));
        Assert.Equal("", Join(ps, PieceKind.Content));
        Assert.Equal(2, Calls(ps).Select(c => c.Id).Distinct().Count());
    }

    [Fact]
    public void Hermes_XmlStyleFunctionCall_Qwen35()
    {
        var ps = Parse("<tool_call>\n<function=add>\n<parameter=a>\n5\n</parameter>\n<parameter=b>\nseven\n</parameter>\n</function>\n</tool_call>", 4);
        var c = Assert.Single(Calls(ps));
        Assert.Equal("add", c.Name);
        Assert.Equal("""{"a":5,"b":"seven"}""", c.ArgumentsJson);
    }

    [Fact]
    public void Mistral_ToolCallsArray_RunsToEndOfOutput()
    {
        var ps = Parse("[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}, {\"name\": \"add\", \"arguments\": {\"a\": 1, \"b\": 2}}]", 5);
        Assert.Equal(["get_weather", "add"], Calls(ps).Select(c => c.Name));
        Assert.Equal("""{"a": 1, "b": 2}""", Calls(ps)[1].ArgumentsJson);
    }

    [Fact]
    public void Llama3_BareJsonAtStart_WithParametersKey()
    {
        var ps = Parse("""{"name": "get_weather", "parameters": {"city": "Oslo"}}""", 6);
        var c = Assert.Single(Calls(ps));
        Assert.Equal("get_weather", c.Name);
        Assert.Equal("""{"city": "Oslo"}""", c.ArgumentsJson);
        Assert.Equal("", Join(ps, PieceKind.Content));
    }

    [Fact]
    public void Llama3_PythonTag()
    {
        var ps = Parse("<|python_tag|>{\"name\": \"add\", \"parameters\": {\"a\": 1, \"b\": 2}}", 3);
        Assert.Equal("add", Assert.Single(Calls(ps)).Name);
    }

    [Fact]
    public void ArgumentsGivenAsJsonString_AreKept()
    {
        var ps = Parse("<tool_call>{\"name\":\"add\",\"arguments\":\"{\\\"a\\\": 1}\"}</tool_call>", 100);
        Assert.Equal("""{"a": 1}""", Assert.Single(Calls(ps)).ArgumentsJson);
    }

    [Fact]
    public void InvalidJson_FallsBackToContent()
    {
        var ps = Parse("<tool_call>{\"name\": \"add\", oops</tool_call>", 4);
        Assert.Empty(Calls(ps));
        Assert.Equal("<tool_call>{\"name\": \"add\", oops</tool_call>", Join(ps, PieceKind.Content));
    }

    [Fact]
    public void UnknownToolName_StaysText()
    {
        var ps = Parse("<tool_call>{\"name\":\"rm_rf\",\"arguments\":{}}</tool_call>", 100);
        Assert.Empty(Calls(ps));
        Assert.Contains("rm_rf", Join(ps, PieceKind.Content));
    }

    [Fact]
    public void NoTools_NothingIsParsed()
    {
        var p = new ChatOutputParser(null, null);
        var o = p.Push("<tool_call>{\"name\":\"add\"}</tool_call>");
        o.AddRange(p.Finish());
        Assert.Empty(Calls(o));
        Assert.Contains("<tool_call>", Join(o, PieceKind.Content));
    }

    [Fact]
    public void PartialMarkerLookalike_IsReleasedAsText()
    {
        var ps = Parse("a <tool is not a call", 1);
        Assert.Equal("a <tool is not a call", Join(ps, PieceKind.Content));
    }

    [Fact]
    public void PlainJsonAnswer_WithoutMatchingTool_IsContent()
    {
        var ps = Parse("""{"result": 42}""", 3);
        Assert.Empty(Calls(ps));
        Assert.Equal("""{"result": 42}""", Join(ps, PieceKind.Content));
    }

    [Theory]
    [InlineData(1)] [InlineData(5)] [InlineData(1000)]
    public void Reasoning_IsSeparatedFromContent(int step)
    {
        var ps = Parse("<think>\nLet me think...\n</think>\n\nThe answer is 5.", step);
        Assert.Equal("Let me think...", Join(ps, PieceKind.Reasoning));
        Assert.Equal("The answer is 5.", Join(ps, PieceKind.Content));
    }

    [Theory]
    [InlineData(1)] [InlineData(1000)]
    public void Reasoning_StartingInPrompt_OnlyHasClosingTag(int step)
    {
        var ps = Parse("Hmm, so.\n</think>\n\nDone.", step, reasoning: true);
        Assert.Equal("Hmm, so.", Join(ps, PieceKind.Reasoning));
        Assert.Equal("Done.", Join(ps, PieceKind.Content));
    }

    [Fact]
    public void Reasoning_Unclosed_IsFlushedAsReasoning()
    {
        var ps = Parse("<think>still thinking", 3);
        Assert.Equal("still thinking", Join(ps, PieceKind.Reasoning));
        Assert.Equal("", Join(ps, PieceKind.Content));
    }

    [Fact]
    public void ReasoningThenToolCall()
    {
        var ps = Parse("<think>need weather</think>\n\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rio\"}}\n</tool_call>", 2);
        Assert.Equal("need weather", Join(ps, PieceKind.Reasoning));
        Assert.Equal("get_weather", Assert.Single(Calls(ps)).Name);
        Assert.Equal("", Join(ps, PieceKind.Content));
    }

    [Fact]
    public void StreamingAndWholeParse_Agree()
    {
        const string text = "<think>a\nb</think>\nHello <b>world</b>!\n<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1}}</tool_call>";
        var whole = Parse(text, 10000);
        for (int step = 1; step < 12; step++)
        {
            var s = Parse(text, step);
            Assert.Equal(Join(whole, PieceKind.Content), Join(s, PieceKind.Content));
            Assert.Equal(Join(whole, PieceKind.Reasoning), Join(s, PieceKind.Reasoning));
            Assert.Equal(Calls(whole).Select(c => (c.Name, c.ArgumentsJson)), Calls(s).Select(c => (c.Name, c.ArgumentsJson)));
        }
    }

    [Fact]
    public void FormatDetection_FromTemplate()
    {
        Assert.Same(ToolCallFormats.Hermes, Assert.Single(ToolCallFormats.ForTemplate("... <tool_call> ...")));
        Assert.Same(ToolCallFormats.Mistral, Assert.Single(ToolCallFormats.ForTemplate("[TOOL_CALLS]")));
        Assert.Same(ToolCallFormats.Llama3, Assert.Single(ToolCallFormats.ForTemplate("Environment: ipython")));
        Assert.Equal(3, ToolCallFormats.ForTemplate("nothing").Count);
    }

    [Fact]
    public void CustomFormat_IsPluggable()
    {
        var ps = Parse("say hi @@call add 1 2@@ bye", 3, [new AtFormat()]);
        var c = Assert.Single(Calls(ps));
        Assert.Equal("add", c.Name);
        Assert.Equal("say hibye", Join(ps, PieceKind.Content));
    }

    sealed class AtFormat : IToolCallFormat
    {
        public string Name => "at";
        public string? StartMarker => "@@call ";
        public string? EndMarker => "@@";
        public bool AllowLeadingJson => false;
        public bool TryParse(string block, out IReadOnlyList<ParsedCall> calls)
        {
            var parts = block.Split(' ');
            calls = [new ParsedCall(parts[0], $"{{\"a\":{parts[1]},\"b\":{parts[2]}}}")];
            return true;
        }
    }
}
