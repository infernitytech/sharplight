using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>The prefix cache is partitioned by <see cref="GenerationRequest.CachePartition"/>: no reuse or block sharing across scopes.</summary>
public sealed class CachePartitionTests
{
    private static GenerationRequest Req(string prompt, string? partition) => new()
    {
        PromptTokens = T.Prompt(prompt), MaxTokens = 100, Sampling = SamplingParams.GreedyDefault, CachePartition = partition,
    };

    [Fact]
    public async Task IdenticalPromptsOfDifferentScopes_DoNotShareSlotContexts()
    {
        var d = new FakeBatchDecoder(2);
        d.Script("r");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        await e.Submit(Req("the same long prompt", "scope-a")).CompleteAsync();
        var other = await e.Submit(Req("the same long prompt", "scope-b")).CompleteAsync();
        var same = await e.Submit(Req("the same long prompt", "scope-a")).CompleteAsync();
        Assert.Equal(0, other.Usage.CachedTokens);
        Assert.True(same.Usage.CachedTokens > 0);
    }

    [Fact]
    public async Task TheDefaultScopeIsUnchanged()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("r");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var p = T.Prompt("same prompt");
        await e.Submit(new GenerationRequest { PromptTokens = p }).CompleteAsync();
        var r = await e.Submit(new GenerationRequest { PromptTokens = p }).CompleteAsync();
        Assert.Equal(p.Length - 1, r.Usage.CachedTokens);
    }

    [Fact]
    public async Task PagedBlocksAreNotSharedAcrossScopes()
    {
        var d = new FakeBatchDecoder(2, kvBlockSize: 4, kvBlocks: 64) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("0123456789");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        string system = new('s', 37);
        var first = e.Submit(Req(system + "A", "scope-a"));
        Assert.True(await T.WaitUntil(() => d.Steps.Count >= 2));
        var second = e.Submit(Req(system + "B", "scope-b"));
        var rs = await Task.WhenAll(first.CompleteAsync(), second.CompleteAsync());
        Assert.Equal(0, rs[1].Usage.CachedTokens);
        Assert.Equal(0, e.GetMetrics().SharedPrefixTokens);

        var third = await e.Submit(Req(system + "C", "scope-a")).CompleteAsync();
        Assert.True(third.Usage.CachedTokens >= 36);
    }
}
