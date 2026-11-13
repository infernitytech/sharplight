using System.Text;
using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

public class EngineTests
{
    private static InferenceEngine Engine(FakeBatchDecoder d, EngineOptions? o = null) => new(d, new FakeTokenizer(), o);

    private static GenerationRequest Req(string prompt, int max = 100, IReadOnlyList<string>? stops = null, int priority = 0, TimeSpan? timeout = null) => new()
    {
        PromptTokens = T.Prompt(prompt), MaxTokens = max, Sampling = SamplingParams.GreedyDefault,
        StopStrings = stops ?? [], Priority = priority, Timeout = timeout,
    };

    [Fact]
    public async Task SingleRequest_ProducesScriptedReply_AndUsage()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("hello");
        using var e = Engine(d);
        var r = await e.Submit(Req("hi")).CompleteAsync();
        Assert.Equal("hello", r.Text);
        Assert.Equal(FinishReason.Stop, r.Finish);
        Assert.Equal(new GenerationUsage(3, 5, 0), r.Usage);
    }

    [Fact]
    public async Task MaxTokens_FinishesWithLength()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("abcdefghij");
        using var e = Engine(d);
        var r = await e.Submit(Req("x", max: 4)).CompleteAsync();
        Assert.Equal("abcd", r.Text);
        Assert.Equal(FinishReason.Length, r.Finish);
        Assert.Equal(4, r.Usage.CompletionTokens);
    }

    [Fact]
    public async Task ContinuousBatching_RunsSequencesInTheSameSteps()
    {
        var d = new FakeBatchDecoder(4) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("0123456789");
        using var e = Engine(d);
        var hs = Enumerable.Range(0, 4).Select(i => e.Submit(Req("p" + i))).ToList();
        var rs = await Task.WhenAll(hs.Select(h => h.CompleteAsync()));
        Assert.All(rs, r => Assert.Equal("0123456789", r.Text));
        // Far fewer steps than 4 * 11 sequential ones, and some step carried several slots.
        Assert.True(d.Steps.Count < 30, $"steps: {d.Steps.Count}");
        Assert.Contains(d.Steps, s => s.Count(x => x.Rows == 1) >= 3);
        Assert.Equal(4, e.GetMetrics().RequestsFinished[FinishReason.Stop]);
    }

    [Fact]
    public async Task SingleSlot_SerializesRequests()
    {
        var d = new FakeBatchDecoder(1, batches: false);
        d.Script("abc");
        using var e = Engine(d);
        var rs = await Task.WhenAll(Enumerable.Range(0, 3).Select(i => e.Submit(Req("q" + i)).CompleteAsync()));
        Assert.All(rs, r => Assert.Equal("abc", r.Text));
        Assert.All(d.Steps, s => Assert.Single(s));
    }

    [Fact]
    public async Task DecodeFirst_PrefillIsChunkedAroundRunningSequences()
    {
        var d = new FakeBatchDecoder(2, maxBatchTokens: 64) { StepDelay = TimeSpan.FromMilliseconds(3) };
        d.Script("x", loop: true);
        using var e = Engine(d, new EngineOptions { PrefillChunk = 16 });
        var longRun = e.Submit(Req("a", max: 400));
        await T.WaitUntil(() => d.Steps.Count > 3);
        var big = e.Submit(Req(new string('z', 100), max: 3));
        await big.CompleteAsync();
        longRun.Cancel();
        // Every step that processed a prompt chunk of slot 1 also decoded slot 0, and chunks respect the limit.
        var prefillSteps = d.Steps.Where(s => s.Any(x => x.Slot == 1 && x.Rows > 1)).ToList();
        Assert.True(prefillSteps.Count >= 100 / 16);
        Assert.All(prefillSteps, s =>
        {
            Assert.Contains(s, x => x.Slot == 0 && x.Rows == 1);
            Assert.True(s.Single(x => x.Slot == 1).Rows <= 16);
        });
        // Non-final chunks do not ask for logits.
        Assert.Contains(d.Steps.SelectMany(s => s), x => x.Slot == 1 && x.Rows == 16 && !x.Logits);
    }

    [Fact]
    public async Task Cancel_RunningRequest_FreesSlot()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("a", loop: true);
        using var e = Engine(d);
        using var cts = new CancellationTokenSource();
        var h = e.Submit(Req("x", max: 100000), cts.Token);
        var chunks = new List<GenerationChunk>();
        await foreach (var c in h.ReadAllAsync())
        {
            chunks.Add(c);
            if (chunks.Count == 3)
            {
                cts.Cancel();
            }
        }
        Assert.Equal(FinishReason.Cancelled, chunks[^1].Finish);
        var r2 = await e.Submit(Req("y", max: 2)).CompleteAsync();
        Assert.Equal("aa", r2.Text);
        Assert.Equal(0, e.GetMetrics().ActiveSequences);
    }

    [Fact]
    public async Task Cancel_QueuedRequest_NeverRuns()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("a", loop: true);
        using var e = Engine(d);
        var first = e.Submit(Req("first", max: 100000));
        await T.WaitUntil(() => e.GetMetrics().ActiveSequences == 1);
        using var cts = new CancellationTokenSource();
        var queued = e.Submit(Req("queued-prompt"), cts.Token);
        Assert.Equal(1, e.GetMetrics().QueueDepth);
        cts.Cancel();
        var r = await queued.CompleteAsync();
        Assert.Equal(FinishReason.Cancelled, r.Finish);
        Assert.Equal(0, e.GetMetrics().QueueDepth);
        first.Cancel();
        await first.CompleteAsync();
        Assert.DoesNotContain(d.Steps, s => s.Any(x => x.Rows == T.Prompt("queued-prompt").Length));
    }

    [Fact]
    public async Task Timeout_FinishesWithTimeout()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(5) };
        d.Script("a", loop: true);
        using var e = Engine(d);
        var r = await e.Submit(Req("x", max: 100000, timeout: TimeSpan.FromMilliseconds(80))).CompleteAsync();
        Assert.Equal(FinishReason.Timeout, r.Finish);
        Assert.True(r.Usage.CompletionTokens > 0);
    }

    [Fact]
    public async Task QueueTimeout_ExpiresWaitingRequest()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(5) };
        d.Script("a", loop: true);
        using var e = Engine(d);
        var first = e.Submit(Req("first", max: 100000));
        await T.WaitUntil(() => e.GetMetrics().ActiveSequences == 1);
        var waiting = e.Submit(new GenerationRequest { PromptTokens = T.Prompt("w"), QueueTimeout = TimeSpan.FromMilliseconds(50) });
        var r = await waiting.CompleteAsync();
        Assert.Equal(FinishReason.Timeout, r.Finish);
        Assert.Equal(0, r.Usage.CompletionTokens);
        first.Cancel();
    }

    [Fact]
    public async Task Admission_QueueFull_And_PromptTooLong()
    {
        var d = new FakeBatchDecoder(1, contextLength: 64) { StepDelay = TimeSpan.FromMilliseconds(5) };
        d.Script("a", loop: true);
        using var e = Engine(d, new EngineOptions { MaxQueue = 1 });
        var running = e.Submit(Req("1", max: 50));
        await T.WaitUntil(() => e.GetMetrics().ActiveSequences == 1);
        var queued = e.Submit(Req("2", max: 1));
        Assert.Throws<QueueFullException>(() => e.Submit(Req("3")));
        Assert.Throws<PromptTooLongException>(() => e.Submit(new GenerationRequest { PromptTokens = new int[64] }));
        Assert.Equal(2, e.GetMetrics().RequestsRejected);
        running.Cancel();
        await queued.CompleteAsync();
    }

    [Fact]
    public async Task Context_Limit_EndsWithLength()
    {
        var d = new FakeBatchDecoder(1, contextLength: 20);
        d.Script("a", loop: true);
        using var e = Engine(d);
        var r = await e.Submit(Req("0123456789", max: 1000)).CompleteAsync();   // prompt 11 tokens -> at most 9 generated
        Assert.Equal(FinishReason.Length, r.Finish);
        Assert.Equal(9, r.Usage.CompletionTokens);
    }

    [Fact]
    public async Task Priority_HigherRunsFirst()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("ab");
        using var e = Engine(d);
        var blocker = e.Submit(Req("blocker", max: 3));
        await T.WaitUntil(() => e.GetMetrics().ActiveSequences == 1);
        var order = new List<string>();
        var low = e.Submit(Req("low", priority: 0));
        var high = e.Submit(Req("high", priority: 5));
        var tl = low.CompleteAsync().ContinueWith(_ => { lock (order) { order.Add("low"); } });
        var th = high.CompleteAsync().ContinueWith(_ => { lock (order) { order.Add("high"); } });
        await Task.WhenAll(tl, th);
        Assert.Equal(["high", "low"], order);
    }

    [Fact]
    public async Task EqualPriority_IsFifo()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("ab");
        using var e = Engine(d);
        var blocker = e.Submit(Req("blocker", max: 3));
        await T.WaitUntil(() => e.GetMetrics().ActiveSequences == 1);
        var order = new List<int>();
        var tasks = Enumerable.Range(0, 4).Select(i => e.Submit(Req("r" + i)).CompleteAsync().ContinueWith(_ => { lock (order) { order.Add(i); } })).ToArray();
        await Task.WhenAll(tasks);
        Assert.Equal([0, 1, 2, 3], order);
    }

    // ---- stop strings / detokenization ---------------------------------------------------------------------------------

    [Fact]
    public async Task StopString_AcrossTokens_IsCutAndNotEmitted()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("hello STOP world");
        using var e = Engine(d);
        var r = await e.Submit(Req("x", stops: ["STOP"])).CompleteAsync();
        Assert.Equal("hello ", r.Text);
        Assert.Equal(FinishReason.Stop, r.Finish);
        Assert.Equal("STOP", r.StopSequence);
    }

    [Fact]
    public async Task StopString_PartialMatch_IsHeldBackThenReleased()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("ab STO cd");
        using var e = Engine(d);
        var chunks = new List<string>();
        await foreach (var c in e.Submit(Req("x", stops: ["STOP"])).ReadAllAsync())
        {
            if (c.Text.Length > 0)
            {
                chunks.Add(c.Text);
            }
        }

        Assert.Equal("ab STO cd", string.Concat(chunks));
        // The partial match "STO" is never emitted on its own: no intermediate text ends in a stop-string prefix.
        var acc = "";
        foreach (var c in chunks[..^1])
        {
            acc += c;
            Assert.False(acc.EndsWith('S') || acc.EndsWith("ST") || acc.EndsWith("STO"), acc);
        }
    }

    [Fact]
    public async Task StopString_HeldTextIsFlushedAtEndOfOutput()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("done ST");
        using var e = Engine(d);
        var r = await e.Submit(Req("x", stops: ["STOP"])).CompleteAsync();
        Assert.Equal("done ST", r.Text);
        Assert.Null(r.StopSequence);
    }

    [Fact]
    public async Task StopTokenId_EndsGenerationWithoutEmittingIt()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("abc!def");
        using var e = Engine(d);
        var r = await e.Submit(new GenerationRequest { PromptTokens = T.Prompt("x"), StopTokenIds = [(int)'!'], Sampling = SamplingParams.GreedyDefault }).CompleteAsync();
        Assert.Equal("abc", r.Text);
        Assert.Equal(3, r.Usage.CompletionTokens);
    }

    [Fact]
    public async Task Utf8_SplitAcrossTokens_IsNeverEmittedBroken()
    {
        const string s = "é€😀 ok";
        var d = new FakeBatchDecoder(1);
        d.Script(s);
        using var e = Engine(d);
        var chunks = new List<string>();
        await foreach (var c in e.Submit(Req("x")).ReadAllAsync())
        {
            chunks.Add(c.Text);
        }

        Assert.Equal(s, string.Concat(chunks));
        Assert.DoesNotContain(chunks, c => c.Contains('�'));
        Assert.True(chunks.Count(c => c.Length > 0) < Encoding.UTF8.GetByteCount(s));   // multi-byte characters arrive whole
    }

    [Fact]
    public async Task Logprobs_AreReportedPerToken()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("ab");
        using var e = Engine(d);
        var r = await e.Submit(new GenerationRequest { PromptTokens = T.Prompt("x"), Logprobs = true, TopLogprobs = 2, Sampling = SamplingParams.GreedyDefault }).CompleteAsync();
        Assert.Equal(2, r.Logprobs.Count);
        Assert.Equal((int)'a', r.Logprobs[0].TokenId);
        Assert.True(r.Logprobs[0].Logprob is < 0 and > -0.01f);
        Assert.Equal(2, r.Logprobs[0].TopAlternatives.Count);
        Assert.Equal((int)'a', r.Logprobs[0].TopAlternatives[0].TokenId);
    }

    // ---- prefix cache --------------------------------------------------------------------------------------------------

    [Fact]
    public async Task PrefixCache_TruncatableDecoder_ReusesLongestCommonPrefix()
    {
        var d = new FakeBatchDecoder(2);
        d.Script("r");
        using var e = Engine(d);
        await e.Submit(Req("hello world abc")).CompleteAsync();
        var r = await e.Submit(Req("hello world xyz")).CompleteAsync();
        Assert.Equal(12, r.Usage.CachedTokens);   // "hello world "
        Assert.Contains(d.Truncations, t => t.Length == 12);
        var m = e.GetMetrics();
        Assert.Equal(1, m.PrefixCacheHits);
        Assert.Equal(2, m.PrefixCacheLookups);
        Assert.Equal(12, m.CachedPromptTokens);
        // The reused slot only prefilled the 4 new rows (xyz + marker).
        Assert.Contains(d.Steps.Last(), x => x.Rows == 1);
    }

    [Fact]
    public async Task PrefixCache_IdenticalPrompt_StillRecomputesLastToken()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("r");
        using var e = Engine(d);
        var p = T.Prompt("same prompt");
        await e.Submit(new GenerationRequest { PromptTokens = p }).CompleteAsync();
        var r = await e.Submit(new GenerationRequest { PromptTokens = p }).CompleteAsync();
        Assert.Equal(p.Length - 1, r.Usage.CachedTokens);
        Assert.Equal("r", r.Text);
    }

    [Fact]
    public async Task PrefixCache_NonTruncatable_OnlyExtendsWholeContext()
    {
        var d = new FakeBatchDecoder(1, canTruncate: false);
        d.Script("xyz");
        using var e = Engine(d);
        var p1 = T.Prompt("turn one");
        var r1 = await e.Submit(new GenerationRequest { PromptTokens = p1 }).CompleteAsync();
        // The slot now holds prompt + the 3 generated tokens that were fed back (the final end token is never fed).
        int ctxLen = p1.Length + 3;
        Assert.Equal(ctxLen, d.Length(0));
        int[] p2 = [.. p1, .. "xyz"u8.ToArray().Select(b => (int)b), .. T.Prompt(" and more")];
        var r2 = await e.Submit(new GenerationRequest { PromptTokens = p2 }).CompleteAsync();
        Assert.Equal(ctxLen, r2.Usage.CachedTokens);
        Assert.Empty(d.Resets);
        // A prompt that merely shares a prefix cannot be reused: the slot is reset.
        var r3 = await e.Submit(Req("turn two")).CompleteAsync();
        Assert.Equal(0, r3.Usage.CachedTokens);
        Assert.Contains(0, d.Resets);
    }

    [Fact]
    public async Task PrefixCache_Disabled_NeverReuses()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("r");
        using var e = Engine(d, new EngineOptions { PrefixCache = false });
        await e.Submit(Req("hello world abc")).CompleteAsync();
        var r = await e.Submit(Req("hello world xyz")).CompleteAsync();
        Assert.Equal(0, r.Usage.CachedTokens);
    }

    [Fact]
    public async Task PrefixCache_PrefersSlotWithPrefix_AndEvictsLeastRecentlyUsed()
    {
        var d = new FakeBatchDecoder(2);
        d.Script("r");
        using var e = Engine(d);
        await e.Submit(Req("AAAA first")).CompleteAsync();     // -> slot 0
        await e.Submit(Req("BBBB second")).CompleteAsync();    // -> slot 1 (empty slots are used first)
        var again = await e.Submit(Req("AAAA first more")).CompleteAsync();   // must pick slot 0 over the older-or-equal slot 1
        Assert.True(again.Usage.CachedTokens >= 10);
        // Now slot 0 was used most recently; a third unrelated prompt evicts slot 1 (the LRU one).
        d.Resets.Clear();
        await e.Submit(Req("CCCC third")).CompleteAsync();
        Assert.Equal([1], d.Resets);
    }

    // ---- failure & shutdown --------------------------------------------------------------------------------------------

    [Fact]
    public async Task DecoderFailure_FailsRunningRequests_AndEngineKeepsWorking()
    {
        var d = new FakeBatchDecoder(1) { FailAfterSteps = 2 };
        d.Script("abcdef");
        using var e = Engine(d);
        var r = await e.Submit(Req("x")).CompleteAsync();
        Assert.Equal(FinishReason.Error, r.Finish);
        Assert.Contains("injected", r.Error);
        d.FailAfterSteps = -1;
        var r2 = await e.Submit(Req("y")).CompleteAsync();
        Assert.Equal("abcdef", r2.Text);
    }

    [Fact]
    public async Task Drain_WaitsForRunningRequests_ThenRejectsNewOnes()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(3) };
        d.Script("0123456789");
        using var e = Engine(d);
        var h = e.Submit(Req("x"));
        var drain = e.DrainAsync(TimeSpan.FromSeconds(10));
        Assert.Throws<EngineStoppedException>(() => e.Submit(Req("late")));
        Assert.True(await drain);
        Assert.Equal("0123456789", (await h.CompleteAsync()).Text);
        Assert.True(e.IsDraining);
    }

    [Fact]
    public async Task Drain_Timeout_CancelsRemaining()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(3) };
        d.Script("a", loop: true);
        using var e = Engine(d);
        var h = e.Submit(Req("x", max: 100000));
        Assert.False(await e.DrainAsync(TimeSpan.FromMilliseconds(100)));
        Assert.Equal(FinishReason.Cancelled, (await h.CompleteAsync()).Finish);
    }

    [Fact]
    public async Task Dispose_CancelsEverything()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(3) };
        d.Script("a", loop: true);
        var e = Engine(d);
        var run = e.Submit(Req("x", max: 100000));
        var queued = e.Submit(Req("y"));
        e.Dispose();
        Assert.Equal(FinishReason.Cancelled, (await run.CompleteAsync()).Finish);
        Assert.Equal(FinishReason.Cancelled, (await queued.CompleteAsync()).Finish);
    }

    [Fact]
    public async Task Metrics_TrackLatencyAndCounts()
    {
        var d = new FakeBatchDecoder(2) { StepDelay = TimeSpan.FromMilliseconds(1) };
        d.Script("hello");
        using var e = Engine(d);
        await Task.WhenAll(e.Submit(Req("a")).CompleteAsync(), e.Submit(Req("b")).CompleteAsync());
        var m = e.GetMetrics();
        Assert.Equal(2, m.RequestsSubmitted);
        Assert.Equal(10, m.CompletionTokens);
        Assert.Equal(2, m.TimeToFirstToken.Count);
        Assert.True(m.InterTokenLatency.Count >= 8);
        Assert.True(m.Steps > 0 && m.StepRows.Count == m.Steps);
        Assert.Equal(2, m.SlotCount);
        Assert.Equal(0, m.ActiveSequences);
        Assert.Equal(m.PromptTokens, 2 * 2);
    }

    [Fact]
    public async Task GenerateAsync_AbandonedEnumeration_CancelsRequest()
    {
        var d = new FakeBatchDecoder(1) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("a", loop: true);
        using var e = Engine(d);
        int n = 0;
        await foreach (var c in e.GenerateAsync(Req("x", max: 100000)))
        {
            if (++n == 3)
            {
                break;
            }
        }
        Assert.True(await T.WaitUntil(() => e.GetMetrics().ActiveSequences == 0));
        Assert.Equal(1, e.GetMetrics().RequestsFinished[FinishReason.Cancelled]);
    }
}
