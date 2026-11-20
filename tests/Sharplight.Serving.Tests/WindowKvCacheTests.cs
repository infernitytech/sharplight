using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Window-sized KV caches (<see cref="KvWindowBlocks"/>, the caches of sliding-window layers): only the blocks a slot's window reaches
/// are held, yet prefix caching, forks, preemption, truncation by speculative decoding and slot reuse keep working. Every step of the
/// <see cref="FakeBatchDecoder"/> checks, through the window block table, that each query's window rows hold the tokens of its context.
/// </summary>
public sealed class WindowKvCacheTests
{
    private const int Window = 6;
    private const int BlockSize = 4;

    private static GenerationRequest Req(string prompt, int max = 10) => new()
    {
        PromptTokens = T.Prompt(prompt), MaxTokens = max, Sampling = SamplingParams.GreedyDefault,
    };

    private static StepCostModel CostlyRows() => new(fixedMs: 1, perRowMs: 1, copyMs: 0, frozen: true);

    [Fact]
    public async Task LongSequences_HoldOnlyTheirWindow()
    {
        var d = new FakeBatchDecoder(4, kvBlockSize: BlockSize, kvBlocks: 256, maxBatchTokens: 32, kvWindow: Window);
        d.Script("0123456789abcdefghijklmnopqrstuvwxyz", loop: true);
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { PrefillChunk = 16 });
        var hs = Enumerable.Range(0, 4).Select(i => e.Submit(Req(new string((char)('a' + i), 90), max: 120))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal(120, r.Usage.CompletionTokens));
        Assert.True(d.WindowKeysChecked > 4 * 200 * Window / 2, $"only {d.WindowKeysChecked} window keys checked");
        // 4 slots of ~210 tokens hold ~53 blocks each in the main pool; the window pool never held more than window + margin + a step.
        Assert.True(d.PeakWindowBlocksHeld <= 4 * 6 + 32 / BlockSize, $"window blocks held: {d.PeakWindowBlocksHeld}");
    }

    [Fact]
    public async Task ManySequences_ShareASmallPool()
    {
        var d = new FakeBatchDecoder(8, kvBlockSize: BlockSize, kvBlocks: 12, kvWindow: Window);
        d.Script("0123456789");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var hs = Enumerable.Range(0, 8).Select(i => e.Submit(Req("p" + i))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal("0123456789", r.Text));
        Assert.True(d.WindowKeysChecked > 0);
    }

    [Fact]
    public async Task CachedPrefix_BringsItsWindowBlocks()
    {
        // As PrefixCacheTests.CachedBlocks_OutliveTheSlotThatComputedThem: the third prompt attaches the 64-token prefix, so the window
        // blocks at its end must have stayed cached when they left the first sequence's window.
        var d = new FakeBatchDecoder(1, kvBlockSize: BlockSize, kvBlocks: 64, kvWindow: Window);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        string system = new('s', 64);
        await e.Submit(Req(system + "A", max: 4)).CompleteAsync();
        var other = await e.Submit(Req(new string('x', 40), max: 4)).CompleteAsync();
        var again = await e.Submit(Req(system + "B", max: 4)).CompleteAsync();
        Assert.Equal(0, other.Usage.CachedTokens);
        Assert.Equal(64, again.Usage.CachedTokens);
        Assert.Equal("0123", again.Text);
    }

    [Fact]
    public async Task ConversationTurns_ReuseTheirSlot()
    {
        var d = new FakeBatchDecoder(2, kvBlockSize: BlockSize, kvBlocks: 128, kvWindow: Window);
        d.Script("0123456789");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        string turn = new string('u', 30);
        var first = await e.Submit(Req(turn)).CompleteAsync();
        var second = await e.Submit(Req(turn + first.Text + "next")).CompleteAsync();
        Assert.Equal("0123456789", second.Text);
        Assert.True(second.Usage.CachedTokens >= 28, $"cached {second.Usage.CachedTokens}");
    }

    [Fact]
    public async Task ConcurrentPrompts_AttachAPrefixAnotherPromptComputes()
    {
        var d = new FakeBatchDecoder(8, kvBlockSize: BlockSize, kvBlocks: 256, maxBatchTokens: 64, kvWindow: Window);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { PrefillChunk = 16, StepCost = CostlyRows() });
        string prefix = new('s', 128);
        var hs = Enumerable.Range(0, 8).Select(i => e.Submit(Req(prefix + (char)('a' + i), max: 4))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal("0123", r.Text));
        Assert.True(rs.Count(r => r.Usage.CachedTokens >= 128) >= 6, string.Join(",", rs.Select(r => r.Usage.CachedTokens)));
    }

    [Fact]
    public async Task SamplesOfOnePrompt_ForkTheirWindow()
    {
        var d = new FakeBatchDecoder(4, kvBlockSize: BlockSize, kvBlocks: 128, kvWindow: Window);
        d.Script("0123456789");
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { StepCost = CostlyRows() });
        string prompt = new('p', 41);
        var hs = Enumerable.Range(0, 4).Select(_ => e.Submit(Req(prompt))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal("0123456789", r.Text));
        Assert.Equal(3, d.Forks);
    }

    [Fact]
    public async Task PreemptedSequences_Recompute()
    {
        var d = new FakeBatchDecoder(2, kvBlockSize: BlockSize, kvBlocks: 9, kvWindow: Window);
        d.Script("0123456789abcdefghij");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var a = e.Submit(Req("pa", max: 20));
        var b = e.Submit(Req("pb", max: 20));
        var rs = await Task.WhenAll(a.CompleteAsync(), b.CompleteAsync());
        Assert.All(rs, r => Assert.Equal("0123456789abcdefghij", r.Text));
        Assert.True(e.GetMetrics().Preemptions >= 1);
    }

    [Fact]
    public async Task SpeculativeDrafts_AreRolledBackInsideTheMargin()
    {
        // Drafts that are often wrong: every verification truncates the slot by the rejected rows, which the window margin keeps.
        var d = new FakeBatchDecoder(4, maxLogitsRows: 32, kvBlockSize: BlockSize, kvBlocks: 256, kvWindow: Window);
        d.Script("0123456789abcdefghijklmnopqrstuvwxyz", loop: true);
        var spec = new GreedyGuessSpeculator(3, c => (c[^1] % 3 == 0) ? c[^1] + 1 : c[^1] + 2);
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { Speculator = spec });
        var hs = Enumerable.Range(0, 4).Select(i => e.Submit(Req(new string((char)('a' + i), 20), max: 60))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal(60, r.Usage.CompletionTokens));
        Assert.All(rs, r => Assert.StartsWith("0123456789abcdefghijklmnopqrstuvwxyz0123", r.Text));
        Assert.Contains(d.Truncations, t => t.Length > 0);
    }

    [Fact]
    public void Truncation_BeforeTheHeldWindow_StartsOver()
    {
        var window = new KvWindowBlocks(1, 4, 16, 16, window: 6, margin: 4, copyKv: null);
        var kv = new KvBlockCache(1, 4, 16, 16, window: window);
        kv.Grow(0, 40);
        kv.Trim(0, 40);                       // keeps from position 40 - 6 + 1 - 4 = 31: blocks 7..9
        Assert.Equal(36, kv.Truncate(0, 36)); // the window of a query at 36 (31..35) is held
        Assert.Equal(0, kv.Truncate(0, 20));  // the window of a query at 20 is gone: the slot starts over
        Assert.Equal(16, window.Free);
    }

    [Fact]
    public void Find_NeedsTheWindowBlocksAtTheEndOfTheRun()
    {
        var window = new KvWindowBlocks(2, 4, 16, 16, window: 6, margin: 0, copyKv: null);
        var kv = new KvBlockCache(2, 4, 32, 16, window: window);
        kv.Grow(0, 24);
        kv.Publish(0, 0, [10, 11, 12, 13, 14, 15]);
        kv.Trim(0, 24);                                  // the window blocks 0..3 leave (published: they stay cached)
        Assert.Equal(6, kv.Find([10, 11, 12, 13, 14, 15], out _));
        kv.Release(0, 0);
        window.Grow(1, 64);                              // takes every free window block, evicting the cached ones
        Assert.Equal(0, kv.Find([10, 11, 12, 13, 14, 15], out _));
    }
}
