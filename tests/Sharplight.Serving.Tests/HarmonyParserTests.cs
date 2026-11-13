using Sharplight.Llm.Serving.Chat;

namespace Sharplight.Serving.Tests;

/// <summary>OpenAI harmony output (gpt-oss) through <see cref="ChatOutputParser"/>: analysis, final and tool-call messages.</summary>
public class HarmonyParserTests
{
    private static readonly string[] Tools = ["get_weather"];

    private static List<ParsedPiece> Parse(string text, int step)
    {
        var p = new ChatOutputParser([ToolCallFormats.Hermes], Tools, harmony: true);
        var o = new List<ParsedPiece>();
        for (int i = 0; i < text.Length; i += step)
        {
            o.AddRange(p.Push(text.Substring(i, Math.Min(step, text.Length - i))));
        }

        o.AddRange(p.Finish());
        return o;
    }

    private static string Join(List<ParsedPiece> ps, PieceKind k) => string.Concat(ps.Where(p => p.Kind == k).Select(p => p.Text));

    [Theory]
    [InlineData(1)] [InlineData(5)] [InlineData(1000)]
    public void Analysis_becomes_reasoning_and_final_becomes_content(int step)
    {
        var ps = Parse("<|channel|>analysis<|message|>The user asks 2+3. So answer: 5.<|end|><|start|>assistant<|channel|>final<|message|>5<|return|>", step);
        Assert.Equal("The user asks 2+3. So answer: 5.", Join(ps, PieceKind.Reasoning).Trim());
        Assert.Equal("5", Join(ps, PieceKind.Content).Trim());
    }

    [Theory]
    [InlineData(1)] [InlineData(4)] [InlineData(1000)]
    public void Commentary_to_function_becomes_tool_call(int step)
    {
        var ps = Parse("<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}<|call|>", step);
        Assert.Equal("Need weather.", Join(ps, PieceKind.Reasoning).Trim());
        var call = Assert.Single(ps.Where(p => p.Kind == PieceKind.ToolCall).Select(p => p.Call!));
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("""{"city":"Paris"}""", call.ArgumentsJson);
        Assert.Equal("", Join(ps, PieceKind.Content).Trim());
    }

    [Fact]
    public void Recipient_before_channel_and_plain_output_are_handled()
    {
        var ps = Parse("<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Rome\"}<|call|>", 3);
        Assert.Equal("get_weather", Assert.Single(ps.Where(p => p.Kind == PieceKind.ToolCall)).Call!.Name);

        var plain = Parse("Just text without channels.", 2);
        Assert.Equal("Just text without channels.", Join(plain, PieceKind.Content));
    }

    [Fact]
    public void Unfinished_analysis_is_closed_at_the_end()
    {
        var ps = Parse("<|channel|>analysis<|message|>Thinking until the token limit", 6);
        Assert.Equal("Thinking until the token limit", Join(ps, PieceKind.Reasoning).Trim());
        Assert.Equal("", Join(ps, PieceKind.Content));
    }
}
