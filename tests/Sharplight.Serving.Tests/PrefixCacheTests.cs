using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Prefix caching by block content over a paged KV cache (<see cref="FakeBatchDecoder"/> on the engine's <see cref="KvBlockCache"/>):
/// cached blocks outlive the slots that computed them, namespaces (partition and salt) and non-text content keep prefixes apart,
/// prompts attach blocks mid-prompt and wait for blocks another prompt is computing, preempted sequences take their blocks back,
/// and models with recurrent state attach blocks together with state snapshots.
/// </summary>
public sealed class PrefixCacheTests
{
    private static GenerationRequest Req(string prompt, string? partition = null, string? salt = null, int max = 4) => new()
    {
        PromptTokens = T.Prompt(prompt), MaxTokens = max, Sampling = SamplingParams.GreedyDefault, CachePartition = partition, CacheSalt = salt,
    };

    private static readonly string System = new('s', 64);

    [Fact]
    public async Task CachedBlocks_OutliveTheSlotThatComputedThem()
    {
        // One slot: the second prompt takes the slot, yet the first prompt's blocks stay cached in the pool for the third.
        var d = new FakeBatchDecoder(1, kvBlockSize: 4, kvBlocks: 64);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        await e.Submit(Req(System + "A")).CompleteAsync();
        var other = await e.Submit(Req(new string('x', 40))).CompleteAsync();
        var again = await e.Submit(Req(System + "B")).CompleteAsync();
        Assert.Equal(0, other.Usage.CachedTokens);
        Assert.Equal(64, again.Usage.CachedTokens);
        Assert.Equal("0123", again.Text);
        var m = e.GetMetrics();
        Assert.Equal(64, m.SharedPrefixTokens);
        Assert.Equal(16, m.SharedPrefixBlocks);
        Assert.True(m.KvBlocksCached > 0);
    }

    [Theory]
    [InlineData("a", null, "b", null)]
    [InlineData(null, "s1", null, "s2")]
    [InlineData("a", "s", "b", "s")]
    [InlineData(null, null, "", null)]
    public async Task DifferentNamespaces_NeverShareBlocks(string? p1, string? s1, string? p2, string? s2)
    {
        var d = new FakeBatchDecoder(1, kvBlockSize: 4, kvBlocks: 64);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        await e.Submit(Req(System + "A", p1, s1)).CompleteAsync();
        var other = await e.Submit(Req(System + "B", p2, s2)).CompleteAsync();
        var same = await e.Submit(Req(System + "C", p1, s1)).CompleteAsync();
        Assert.Equal(0, other.Usage.CachedTokens);
        Assert.True(same.Usage.CachedTokens >= 64);
    }

    [Fact]
    public async Task ConcurrentNamespaces_NeverShareBlocksOrWaitForEachOther()
    {
        var d = new FakeBatchDecoder(4, kvBlockSize: 4, kvBlocks: 128) { StepDelay = TimeSpan.FromMilliseconds(1) };
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { PrefillChunk = 8 });
        var hs = new[] { "a", "b", "c", "d" }.Select(p => e.Submit(Req(System + p, partition: p))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal(0, r.Usage.CachedTokens));
        Assert.Equal(0, e.GetMetrics().SharedPrefixTokens);
    }

    [Fact]
    public async Task DifferentPromptContent_DoesNotShareTheBlocksItOverlaps()
    {
        var d = new FakeBatchDecoder(1, kvBlockSize: 4, kvBlocks: 64);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        // The same placeholder tokens at positions 32..47 stand for two different images.
        GenerationRequest WithImage(UInt128 image) => Req(System + "A") with { PromptContent = [new PromptContentKey(32, 16, image)] };
        await e.Submit(WithImage(1)).CompleteAsync();
        var other = await e.Submit(WithImage(2)).CompleteAsync();
        var same = await e.Submit(WithImage(1)).CompleteAsync();
        Assert.Equal(32, other.Usage.CachedTokens);   // the blocks before the image
        Assert.True(same.Usage.CachedTokens >= 64);
    }

    [Fact]
    public async Task BypassPrefixCache_ComputesThePrompt_AndLeavesItsBlocks()
    {
        var d = new FakeBatchDecoder(2, kvBlockSize: 4, kvBlocks: 64);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        await e.Submit(Req(System + "A")).CompleteAsync();
        var bypass = await e.Submit(Req(System + "A") with { BypassPrefixCache = true }).CompleteAsync();
        Assert.Equal(0, bypass.Usage.CachedTokens);
        Assert.Equal("0123", bypass.Text);
        var after = await e.Submit(Req(System + "B")).CompleteAsync();
        Assert.True(after.Usage.CachedTokens >= 64);
    }

    [Fact]
    public async Task ConcurrentPrompts_WaitForBlocksAnotherPromptComputes_InsteadOfComputingThemTwice()
    {
        // Eight prompts with one long shared prefix arrive together; prefill runs in chunks of 16 rows. Only the first computes the
        // prefix; the others attach its blocks as they are published.
        var d = new FakeBatchDecoder(8, kvBlockSize: 4, kvBlocks: 256, maxBatchTokens: 64);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { PrefillChunk = 16 });
        string prefix = new('s', 128);
        var hs = Enumerable.Range(0, 8).Select(i => e.Submit(Req(prefix + (char)('a' + i)))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal("0123", r.Text));
        Assert.Equal(7, rs.Count(r => r.Usage.CachedTokens >= 128));
        int promptRows = d.Steps.Sum(s => s.Where(x => x.Rows > 1 || !x.Logits).Sum(x => x.Rows));
        Assert.True(promptRows < 2 * 130, $"prompt rows {promptRows}: the prefix was computed more than once");
    }

    [Fact]
    public async Task PreemptedSequences_TakeTheirCachedBlocksBack()
    {
        // As GrowingSequences_ArePreempted_AndRecomputed: the newer sequence is preempted, and its blocks are still cached when it
        // resumes, so it recomputes less than its whole context.
        var d = new FakeBatchDecoder(2, kvBlockSize: 4, kvBlocks: 9);
        d.Script("0123456789abcdefghij");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var a = e.Submit(Req("pa", max: 20));
        var b = e.Submit(Req("pb", max: 20));
        var rs = await Task.WhenAll(a.CompleteAsync(), b.CompleteAsync());
        Assert.All(rs, r => Assert.Equal("0123456789abcdefghij", r.Text));
        var m = e.GetMetrics();
        Assert.True(m.Preemptions >= 1);
        Assert.True(m.SharedPrefixTokens > 0, "the preempted context's cached blocks were not attached again");
    }

    [Fact]
    public void EvictionOrder_TailBlocksGoBeforePrefixBlocks()
    {
        var kv = new KvBlockCache(2, 4, 6, 8);
        kv.Grow(0, 12);                                  // blocks 0, 1, 2 for slot 0
        kv.Publish(0, 0, [10, 11, 12]);
        kv.Release(0, 0);                                // released last-first: queue = 3, 4, 5 (empty) then 2, 1, 0 (cached)
        Assert.Equal(3, kv.Cached);
        kv.Grow(1, 16);                                  // four blocks: the three empty ones, then the oldest cached: the tail block
        Assert.Equal(1, kv.Evictions);
        Assert.Equal(2, kv.Find([10, 11, 12], out _));   // the prefix blocks are still cached
    }

    [Fact]
    public void PublishedBlocks_AreNeverRewritten_ByATruncation()
    {
        var kv = new KvBlockCache(1, 4, 8, 8);
        kv.Grow(0, 8);
        kv.Publish(0, 0, [1, 2]);
        Assert.Equal(4, kv.Truncate(0, 6));   // cutting into the published second block drops it
    }

    [Fact]
    public async Task RecurrentModel_AttachesBlocksWithStateSnapshots()
    {
        // A model with recurrent state: the first prompt leaves blocks but no snapshot at the shared prefix; the second prompt finds
        // the shared prefix and snapshots its end; the third restores it and computes only its own suffix.
        var d = new FakeBatchDecoder(1, canTruncate: false, kvBlockSize: 4, kvBlocks: 128, stateSnapshots: 4);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var first = await e.Submit(Req(System + "first prompt")).CompleteAsync();
        var second = await e.Submit(Req(System + "second one")).CompleteAsync();
        var third = await e.Submit(Req(System + "third")).CompleteAsync();
        Assert.Equal("0123", third.Text);
        Assert.Equal(0, first.Usage.CachedTokens);
        Assert.Equal(64, third.Usage.CachedTokens);
        Assert.True(d.SnapshotRestores >= 1);
        Assert.True(e.GetMetrics().StateSnapshots >= 1);
    }

    [Fact]
    public async Task RecurrentModel_ContinuesAConversation_FromThePromptEndSnapshot()
    {
        // Turn two resends turn one (prompt + reply) plus a new message; after another prompt took the slot, turn two restores the
        // snapshot taken at the end of turn one's prompt.
        var d = new FakeBatchDecoder(1, canTruncate: false, kvBlockSize: 4, kvBlocks: 128, stateSnapshots: 4);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        string turn1 = new('u', 47);
        var r1 = await e.Submit(Req(turn1)).CompleteAsync();
        await e.Submit(Req(new string('z', 30))).CompleteAsync();
        var r2 = await e.Submit(Req(turn1 + "\u0001" + r1.Text + "next question")).CompleteAsync();
        Assert.Equal("0123", r2.Text);
        Assert.True(r2.Usage.CachedTokens >= 44, $"cached {r2.Usage.CachedTokens}");
    }

    [Fact]
    public async Task RecurrentModel_WithoutSnapshots_SharesNothing()
    {
        var d = new FakeBatchDecoder(1, canTruncate: false, kvBlockSize: 4, kvBlocks: 128);
        d.Script("0123");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        await e.Submit(Req(System + "A")).CompleteAsync();
        await e.Submit(Req(System + "B")).CompleteAsync();
        var r = await e.Submit(Req(System + "C")).CompleteAsync();
        Assert.Equal(0, r.Usage.CachedTokens);
        Assert.Equal(0, d.SnapshotRestores);
    }
}
