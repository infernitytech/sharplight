using Sharplight.Studio.Models;

namespace Sharplight.Studio.Tests;

public sealed class StatsSummaryTests
{
    private static RequestRecord Record(int prompt, int cached, int completion, double queue, double prefill, double decode, int drafted = 0, int accepted = 0) => new()
    {
        Model = "m",
        Stats = new MessageStats
        {
            PromptTokens = prompt, CachedTokens = cached, CompletionTokens = completion, QueueMs = queue, PrefillMs = prefill, DecodeMs = decode,
            DraftTokens = drafted, AcceptedDraftTokens = accepted,
        },
    };

    [Fact]
    public void Empty_IsAllZero()
    {
        var summary = StatsSummary.From([]);
        Assert.Equal(0, summary.Requests);
        Assert.Null(summary.AcceptanceRate);
    }

    [Fact]
    public void Rates_AreTokenWeighted_AndExcludeCachedPrompt()
    {
        var summary = StatsSummary.From(
        [
            Record(prompt: 1100, cached: 100, completion: 101, queue: 0, prefill: 1000, decode: 1000, drafted: 10, accepted: 5),
            Record(prompt: 100, cached: 0, completion: 201, queue: 10, prefill: 100, decode: 3000, drafted: 10, accepted: 10),
            Record(prompt: 50, cached: 50, completion: 1, queue: 5, prefill: 0, decode: 0),
        ]);

        Assert.Equal(3, summary.Requests);
        Assert.Equal(1250, summary.PromptTokens);
        Assert.Equal(303, summary.CompletionTokens);
        Assert.Equal(300.0 / 4, summary.DecodeTokensPerSecond, 3);      // (100 + 200) tokens over 4 s; the 1-token answer has no decode
        Assert.Equal(1100.0 / 1.1, summary.PrefillTokensPerSecond, 3);  // (1000 + 100) uncached tokens over 1.1 s
        Assert.Equal(110, summary.MedianTimeToFirstTokenMs);            // ttfts 5, 110, 1000
        Assert.Equal(1000, summary.P95TimeToFirstTokenMs);
        Assert.Equal(0.75, summary.AcceptanceRate);
    }
}
