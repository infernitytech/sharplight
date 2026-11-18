using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Serving.Tests;

public class ParserTests
{
    private static readonly string[] Tools = ["get_weather", "add"];

    /// <summary>Feeds <paramref name="text"/> in pieces of <paramref name="step"/> chars and returns all pieces.</summary>
    private static List<ChatOutputPiece> Parse(string text, int step, IReadOnlyList<IToolCallFormat>? formats = null, bool reasoning = false, string[]? tools = null)
    {
        var p = new ChatOutputParser(new ChatOutputParserOptions { ToolCallFormats = formats ?? ToolCallFormats.All, ToolNames = tools ?? Tools, StartsInReasoning = reasoning });
        var o = new List<ChatOutputPiece>();
        for (int i = 0; i < text.Length; i += step)
        {
            o.AddRange(p.Push(text.Substring(i, Math.Min(step, text.Length - i))));
        }

        o.AddRange(p.Complete());
        return o;
    }

    private static string Join(List<ChatOutputPiece> ps, ChatOutputPieceKind k) => string.Concat(ps.Where(p => p.Kind == k).Select(p => p.Text));
    private static List<ToolCall> Calls(List<ChatOutputPiece> ps) => [.. ps.Where(p => p.Kind == ChatOutputPieceKind.ToolCall).Select(p => p.Call!)];

    [Theory]
    [InlineData(1)] [InlineData(3)] [InlineData(1000)]
    public void Hermes_JsonToolCall(int step)
    {
        var ps = Parse("Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", step);
        Assert.Equal("Let me check.", Join(ps, ChatOutputPieceKind.Content));
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
        Assert.Equal("", Join(ps, ChatOutputPieceKind.Content));
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
        Assert.Equal("", Join(ps, ChatOutputPieceKind.Content));
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
        Assert.Equal("<tool_call>{\"name\": \"add\", oops</tool_call>", Join(ps, ChatOutputPieceKind.Content));
    }

    [Fact]
    public void UnknownToolName_StaysText()
    {
        var ps = Parse("<tool_call>{\"name\":\"rm_rf\",\"arguments\":{}}</tool_call>", 100);
        Assert.Empty(Calls(ps));
        Assert.Contains("rm_rf", Join(ps, ChatOutputPieceKind.Content));
    }

    [Fact]
    public void NoTools_NothingIsParsed()
    {
        var p = new ChatOutputParser();
        var o = new List<ChatOutputPiece>(p.Push("<tool_call>{\"name\":\"add\"}</tool_call>"));
        o.AddRange(p.Complete());
        Assert.Empty(Calls(o));
        Assert.Contains("<tool_call>", Join(o, ChatOutputPieceKind.Content));
    }

    [Fact]
    public void PartialMarkerLookalike_IsReleasedAsText()
    {
        var ps = Parse("a <tool is not a call", 1);
        Assert.Equal("a <tool is not a call", Join(ps, ChatOutputPieceKind.Content));
    }

    [Fact]
    public void PlainJsonAnswer_WithoutMatchingTool_IsContent()
    {
        var ps = Parse("""{"result": 42}""", 3);
        Assert.Empty(Calls(ps));
        Assert.Equal("""{"result": 42}""", Join(ps, ChatOutputPieceKind.Content));
    }

    [Theory]
    [InlineData(1)] [InlineData(5)] [InlineData(1000)]
    public void Reasoning_IsSeparatedFromContent(int step)
    {
        var ps = Parse("<think>\nLet me think...\n</think>\n\nThe answer is 5.", step);
        Assert.Equal("Let me think...", Join(ps, ChatOutputPieceKind.Reasoning));
        Assert.Equal("The answer is 5.", Join(ps, ChatOutputPieceKind.Content));
    }

    [Theory]
    [InlineData(1)] [InlineData(1000)]
    public void Reasoning_StartingInPrompt_OnlyHasClosingTag(int step)
    {
        var ps = Parse("Hmm, so.\n</think>\n\nDone.", step, reasoning: true);
        Assert.Equal("Hmm, so.", Join(ps, ChatOutputPieceKind.Reasoning));
        Assert.Equal("Done.", Join(ps, ChatOutputPieceKind.Content));
    }

    [Fact]
    public void Reasoning_Unclosed_IsFlushedAsReasoning()
    {
        var ps = Parse("<think>still thinking", 3);
        Assert.Equal("still thinking", Join(ps, ChatOutputPieceKind.Reasoning));
        Assert.Equal("", Join(ps, ChatOutputPieceKind.Content));
    }

    [Fact]
    public void ReasoningThenToolCall()
    {
        var ps = Parse("<think>need weather</think>\n\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rio\"}}\n</tool_call>", 2);
        Assert.Equal("need weather", Join(ps, ChatOutputPieceKind.Reasoning));
        Assert.Equal("get_weather", Assert.Single(Calls(ps)).Name);
        Assert.Equal("", Join(ps, ChatOutputPieceKind.Content));
    }

    [Fact]
    public void StreamingAndWholeParse_Agree()
    {
        const string text = "<think>a\nb</think>\nHello <b>world</b>!\n<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1}}</tool_call>";
        var whole = Parse(text, 10000);
        for (int step = 1; step < 12; step++)
        {
            var s = Parse(text, step);
            Assert.Equal(Join(whole, ChatOutputPieceKind.Content), Join(s, ChatOutputPieceKind.Content));
            Assert.Equal(Join(whole, ChatOutputPieceKind.Reasoning), Join(s, ChatOutputPieceKind.Reasoning));
            Assert.Equal(Calls(whole).Select(c => (c.Name, c.ArgumentsJson)), Calls(s).Select(c => (c.Name, c.ArgumentsJson)));
        }
    }

    /// <summary>
    /// Qwen3.5 / Qwen3.8 structured thinking (indented lists, which markdown shows as code when they leak into the answer) arriving
    /// token by token, with the close marker split over several pieces: all of it is reasoning, only the reply is content.
    /// </summary>
    [Fact]
    public void StructuredThinking_TokenByToken_StaysReasoning()
    {
        string[] tokens =
        [
            "Thinking", " Process", ":\n\n", "1", ".", "  **", "Draft", ":**\n", "    ", "*", "   ", "*Option", " 1", " (Casual", "):*",
            " \"Haha", ", yeah", ", I try", "!\"\n\n", "    ", "*", "   ", "Acknowledge", " the speed", ".\n", "    ", "*", "   ", "Offer",
            " assistance", ".\n", "</", "think", ">", "\n\n", "Haha", ", yeah", ", I try", "!",
        ];
        var parser = new ChatOutputParser(new ChatOutputParserOptions { StartsInReasoning = true });
        var pieces = new List<ChatOutputPiece>();
        foreach (var token in tokens)
        {
            pieces.AddRange(parser.Push(token));
        }

        pieces.AddRange(parser.Complete());
        string reasoning = Join(pieces, ChatOutputPieceKind.Reasoning);
        Assert.StartsWith("Thinking Process:", reasoning);
        Assert.EndsWith("    *   Offer assistance.", reasoning);
        Assert.Equal("Haha, yeah, I try!", Join(pieces, ChatOutputPieceKind.Content));
    }

    [Theory]
    [InlineData("<|im_start|>assistant\n<think>\n", true)]                         // Qwen3 / Qwen3.5 / Qwen3.8, thinking on
    [InlineData("<|im_start|>assistant\n<think>\n\n</think>\n\n", false)]          // thinking off
    [InlineData("<｜Assistant｜><think>\n", true)]                                  // DeepSeek-R1
    [InlineData("<|im_start|>assistant\n", false)]                                 // the model opens reasoning itself (Qwen3 hybrid)
    [InlineData("<|start|>assistant", false)]                                      // gpt-oss harmony
    public void PromptEndsInReasoning_OnlyForAnOpenBlock(string promptEnd, bool expected)
    {
        Assert.Equal(expected, ChatOutputParser.PromptEndsInReasoning("<|im_start|>user\nhi<|im_end|>\n" + promptEnd));
    }

    [Fact]
    public void HarmonyTemplate_IsDetected()
    {
        Assert.True(ChatOutputParser.IsHarmonyTemplate("{{ '<|start|>assistant<|channel|>final<|message|>' }}"));
        Assert.False(ChatOutputParser.IsHarmonyTemplate("{{ '<|im_start|>assistant\n<think>\n' }}"));
        Assert.False(ChatOutputParser.IsHarmonyTemplate(null));
    }

    [Fact]
    public void StopStrings_CutTheOutput_AcrossPieces()
    {
        var parser = new ChatOutputParser(new ChatOutputParserOptions { StopStrings = ["END"] });
        var pieces = new List<ChatOutputPiece>();
        foreach (var piece in new[] { "one two E", "N", "D three" })
        {
            pieces.AddRange(parser.Push(piece));
        }

        pieces.AddRange(parser.Complete());
        Assert.Equal("one two ", Join(pieces, ChatOutputPieceKind.Content));
        Assert.True(parser.Stopped);
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
        Assert.Equal("say hibye", Join(ps, ChatOutputPieceKind.Content));
    }

    private sealed class AtFormat : IToolCallFormat
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
