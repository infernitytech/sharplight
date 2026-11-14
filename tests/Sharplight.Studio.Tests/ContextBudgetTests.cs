using Sharplight.Studio.Services.Documents;

namespace Sharplight.Studio.Tests;

public sealed class ContextBudgetTests
{
    private static int Count(string text) => (text.Length + 3) / 4;

    [Fact]
    public void AnswerReserve_IsCappedAtAQuarterOfTheContext()
    {
        Assert.Equal(1000, ContextBudget.AnswerReserve(8192, 1000));
        Assert.Equal(2048, ContextBudget.AnswerReserve(8192, 100_000));
        Assert.Equal(2048, ContextBudget.AnswerReserve(8192, 0));
    }

    [Fact]
    public void Available_SubtractsAnswerConversationAndMargin()
    {
        Assert.Equal(8192 - 2048 - 500 - ContextBudget.TemplateMargin, ContextBudget.Available(8192, 2048, 500));
        Assert.Equal(0, ContextBudget.Available(1000, 2048, 500));
    }

    [Fact]
    public void Fit_KeepsTextThatFits()
    {
        var (text, tokens, truncated) = ContextBudget.Fit("short", 100, Count);
        Assert.Equal("short", text);
        Assert.Equal(2, tokens);
        Assert.False(truncated);
    }

    [Fact]
    public void Fit_TruncatesToTheBudget_WithANote()
    {
        var document = string.Join(' ', Enumerable.Range(0, 2000).Select(i => $"word{i}"));
        var (text, tokens, truncated) = ContextBudget.Fit(document, 300, Count);
        Assert.True(truncated);
        Assert.True(tokens <= 300);
        Assert.True(tokens > 200, $"only {tokens} tokens kept");
        Assert.EndsWith(ContextBudget.TruncationNote, text);
        Assert.StartsWith("word0 word1", text);
    }

    [Fact]
    public void Fit_DropsTheText_WhenNotEvenTheNoteFits()
    {
        Assert.Equal(("", 0, true), ContextBudget.Fit(new string('x', 1000), 3, Count));
    }
}
