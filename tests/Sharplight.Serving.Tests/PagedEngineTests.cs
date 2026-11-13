using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Scheduling over a paged KV cache (<see cref="FakeBatchDecoder"/> with blocks): admission by free blocks, preemption with
/// recomputation when running sequences outgrow the pool, and prompt prefixes shared between slots as blocks.
/// </summary>
public class PagedEngineTests
{
    private static GenerationRequest Req(string prompt, int max = 100) => new()
    {
        PromptTokens = T.Prompt(prompt), MaxTokens = max, Sampling = SamplingParams.GreedyDefault,
    };

    [Fact]
    public async Task ManySequences_ShareASmallPool_AndAllFinishCorrectly()
    {
        // 8 slots but only 12 blocks of 4 tokens: each sequence needs 4 blocks (3 prompt + 10 reply tokens), so at most three
        // fit at once; the rest wait for blocks, not slots.
        var d = new FakeBatchDecoder(8, kvBlockSize: 4, kvBlocks: 12);
        d.Script("0123456789");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var hs = Enumerable.Range(0, 8).Select(i => e.Submit(Req("p" + i))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal("0123456789", r.Text));
        Assert.All(rs, r => Assert.Equal(FinishReason.Stop, r.Finish));
        Assert.True(d.PeakBlocksHeld <= 12);
        Assert.Equal(12 * 4, e.GetMetrics().KvCapacityTokens);
    }

    [Fact]
    public async Task GrowingSequences_ArePreempted_AndRecomputed()
    {
        // Two sequences of 13 tokens (4 blocks each) in a pool of 5 blocks: both are admitted on their prompt blocks, and when
        // they outgrow the pool the newer one gives its blocks back and recomputes its context once the other has finished.
        var d = new FakeBatchDecoder(2, kvBlockSize: 4, kvBlocks: 5);
        d.Script("0123456789");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var a = e.Submit(Req("pa"));
        var b = e.Submit(Req("pb"));
        var rs = await Task.WhenAll(a.CompleteAsync(), b.CompleteAsync());
        Assert.All(rs, r => Assert.Equal("0123456789", r.Text));
        Assert.True(e.GetMetrics().Preemptions >= 1, "expected a preemption");
        // Replayed rows are fed without logits.
        Assert.Contains(d.Steps, s => s.Any(x => !x.Logits && x.Rows > 1));
    }

    [Fact]
    public async Task Admission_ReservesTheRestOfRunningPrompts()
    {
        // Two 30-token prompts prefilled in chunks of 4 in a pool of 11 blocks: each sequence needs 9 blocks, so the second
        // is admitted only after the first finishes (counting just the blocks the first holds after its first chunk would
        // admit both and preempt one of them).
        var d = new FakeBatchDecoder(2, kvBlockSize: 4, kvBlocks: 11);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { PrefillChunk = 4 });
        var a = e.Submit(Req(new string('a', 29)));
        var b = e.Submit(Req(new string('b', 29)));
        var rs = await Task.WhenAll(a.CompleteAsync(), b.CompleteAsync());
        Assert.All(rs, r => Assert.Equal("0123", r.Text));
        Assert.Equal(0, e.GetMetrics().Preemptions);
        Assert.True(d.PeakBlocksHeld <= 11);
    }

    [Fact]
    public async Task SequenceLongerThanThePool_IsCappedByIt()
    {
        var d = new FakeBatchDecoder(1, contextLength: 4096, kvBlockSize: 4, kvBlocks: 4);
        d.Script("0123456789abcdefghijklmnopqrstuvwxyz", loop: true);
        using var e = new InferenceEngine(d, new FakeTokenizer());
        Assert.Equal(16, e.ContextLength);
        var r = await e.Submit(Req("p")).CompleteAsync();
        Assert.Equal(FinishReason.Length, r.Finish);
        Assert.Equal(16 - 2, r.Usage.CompletionTokens);   // prompt "p" + marker
    }

    [Fact]
    public async Task ConcurrentRequests_ShareTheirCommonPromptBlocks()
    {
        var d = new FakeBatchDecoder(2, kvBlockSize: 4, kvBlocks: 64) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("0123456789");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        string system = new('s', 37);
        var first = e.Submit(Req(system + "A"));
        Assert.True(await T.WaitUntil(() => d.Steps.Count >= 2));   // the first prompt is in the cache
        var second = e.Submit(Req(system + "B"));
        var rs = await Task.WhenAll(first.CompleteAsync(), second.CompleteAsync());
        Assert.All(rs, r => Assert.Equal("0123456789", r.Text));
        Assert.Equal(36, rs[1].Usage.CachedTokens);   // the 37 common tokens, rounded down to whole blocks
        Assert.Equal(36, e.GetMetrics().SharedPrefixTokens);
    }
}
