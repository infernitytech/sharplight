using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Llm.Structured;

namespace Sharplight.Serving.Tests;

/// <summary>Minimum length, ignore-EOS, fair queuing, speculative decoding and constraint masks of the engine.</summary>
public class EngineFeatureTests
{
    private static InferenceEngine Engine(FakeBatchDecoder d, EngineOptions? o = null) => new(d, new FakeTokenizer(), o);

    private static GenerationRequest Req(string prompt, int max = 100) => new()
    {
        PromptTokens = T.Prompt(prompt), MaxTokens = max, Sampling = SamplingParams.GreedyDefault,
    };

    [Fact]
    public async Task IgnoreEos_GeneratesThroughEndOfGeneration()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("abc");
        using var e = Engine(d);
        var r = await e.Submit(Req("x", max: 6) with { StopAtEog = false }).CompleteAsync(TestContext.Current.CancellationToken);
        Assert.Equal(FinishReason.Length, r.Finish);
        Assert.Equal(6, r.Usage.CompletionTokens);
        Assert.StartsWith("abc</s>", r.Text);
    }

    [Fact]
    public async Task MinTokens_SuppressesEndOfGenerationUntilReached()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("ab");
        using var e = Engine(d);
        var r = await e.Submit(Req("x") with { MinTokens = 4 }).CompleteAsync(TestContext.Current.CancellationToken);
        Assert.Equal(FinishReason.Stop, r.Finish);
        Assert.Equal(4, r.Usage.CompletionTokens);
        Assert.StartsWith("ab", r.Text);
    }

    [Fact]
    public async Task MinTokens_AlsoSuppressesStopTokenIds()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("abcdef");
        using var e = Engine(d);
        var r = await e.Submit(Req("x") with { MinTokens = 3, StopTokenIds = ['b'] }).CompleteAsync(TestContext.Current.CancellationToken);
        Assert.DoesNotContain('b', r.Text);
        Assert.True(r.Usage.CompletionTokens >= 3);
    }

    [Fact]
    public async Task FairQueuing_LightTenantOvertakesHeavyTenantsBacklog()
    {
        var d = new FakeBatchDecoder(1, batches: false) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("0123456789");
        using var e = Engine(d);
        var order = new List<string>();
        var ct = TestContext.Current.CancellationToken;
        async Task Run(string name, string tenant)
        {
            await e.Submit(Req(name) with { Tenant = tenant }).CompleteAsync(ct);
            lock (order)
            {
                order.Add(name);
            }
        }

        var a1 = Run("a1", "heavy");
        await T.WaitUntil(() => d.Steps.Count > 0);
        var a2 = Run("a2", "heavy");
        var a3 = Run("a3", "heavy");
        await T.WaitUntil(() => e.GetMetrics().QueueDepth == 2);
        var b1 = Run("b1", "light");
        await Task.WhenAll(a1, a2, a3, b1);
        Assert.Equal(["a1", "b1", "a2", "a3"], order);
    }

    [Fact]
    public async Task Priority_StillBeatsFairShare()
    {
        var d = new FakeBatchDecoder(1, batches: false) { StepDelay = TimeSpan.FromMilliseconds(2) };
        d.Script("0123456789");
        using var e = Engine(d);
        var ct = TestContext.Current.CancellationToken;
        var first = e.Submit(Req("a") with { Tenant = "x" });
        await T.WaitUntil(() => d.Steps.Count > 0);
        var low = e.Submit(Req("b") with { Tenant = "fresh" });
        var high = e.Submit(Req("c") with { Tenant = "x", Priority = 5 });
        var highDone = high.CompleteAsync(ct);
        var lowDone = low.CompleteAsync(ct);
        await first.CompleteAsync(ct);
        var winner = await Task.WhenAny(highDone, lowDone);
        Assert.Same(highDone, winner);
        await lowDone;
    }

    [Fact]
    public async Task NgramSpeculation_SameOutput_FewerSteps()
    {
        const string Reply = "abcdabcdabcdabcdabcdabcdabcdabcd";
        var ct = TestContext.Current.CancellationToken;
        var plain = new FakeBatchDecoder(2, maxLogitsRows: 10);
        plain.Script(Reply);
        string expected;
        using (var e = Engine(plain))
        {
            expected = (await e.Submit(Req("abcd")).CompleteAsync(ct)).Text;
        }

        var d = new FakeBatchDecoder(2, maxLogitsRows: 10);
        d.Script(Reply);
        using var spec = Engine(d, new EngineOptions { Speculator = new NgramSpeculator(maxDraftTokens: 4, minNgram: 2) });
        Assert.True(spec.Speculates);
        var r = await spec.Submit(Req("abcd")).CompleteAsync(ct);
        Assert.Equal(expected, r.Text);
        Assert.Equal(Reply, r.Text);
        var m = spec.GetMetrics();
        Assert.True(m.AcceptedDraftTokens > 10, $"accepted {m.AcceptedDraftTokens} of {m.DraftTokens}");
        Assert.True(d.Steps.Count < plain.Steps.Count / 2, $"{d.Steps.Count} steps vs {plain.Steps.Count}");
        Assert.Contains(d.LogitsRowsPerStep, rows => rows.Any(n => n > 1));
    }

    [Fact]
    public async Task NgramSpeculation_RejectedDraftsAreRolledBack()
    {
        // The prompt suggests "xyz" continuations that the model never produces: every draft is rejected.
        const string Reply = "xyq xyr xys xyt xyu";
        var d = new FakeBatchDecoder(1, maxLogitsRows: 6);
        d.Script(Reply);
        using var e = Engine(d, new EngineOptions { Speculator = new NgramSpeculator(maxDraftTokens: 3, minNgram: 2) });
        var r = await e.Submit(Req("xyz xyz xyz")).CompleteAsync(TestContext.Current.CancellationToken);
        Assert.Equal(Reply, r.Text);
        Assert.NotEmpty(d.Truncations);
        // The decoder context is exactly prompt + reply (the end-of-generation token is never fed).
        Assert.Equal(T.Prompt("xyz xyz xyz").Length + Reply.Length, d.Context(0).Count);
    }

    [Fact]
    public void Speculation_IsDisabledWhenTheDecoderCannotRollBack()
    {
        var d = new FakeBatchDecoder(2, canTruncate: false, maxLogitsRows: 8);
        using var e = Engine(d, new EngineOptions { Speculator = new NgramSpeculator() });
        Assert.False(e.Speculates);
    }

    [Fact]
    public async Task Speculation_RespectsMaxTokensAndStopStrings()
    {
        var d = new FakeBatchDecoder(1, maxLogitsRows: 6);
        d.Script("abab.abab.abab.abab.");
        using var e = Engine(d, new EngineOptions { Speculator = new NgramSpeculator(maxDraftTokens: 5, minNgram: 1) });
        var ct = TestContext.Current.CancellationToken;
        var limited = await e.Submit(Req("abab") with { MaxTokens = 7 }).CompleteAsync(ct);
        Assert.Equal("abab.ab", limited.Text);
        Assert.Equal(7, limited.Usage.CompletionTokens);
        var stopped = await e.Submit(Req("abab") with { StopStrings = ["b.a"] }).CompleteAsync(ct);
        Assert.Equal("aba", stopped.Text);
        Assert.Equal("b.a", stopped.StopSequence);
    }

    [Fact]
    public async Task Constraint_MasksDisallowedTokens()
    {
        var d = new FakeBatchDecoder(1);
        d.Script("a1b2c3");
        using var e = Engine(d);
        var r = await e.Submit(Req("x") with { Constraint = new DigitsConstraint(3) }).CompleteAsync(TestContext.Current.CancellationToken);
        Assert.Equal(3, r.Text.Length);
        Assert.All(r.Text, c => Assert.True(char.IsAsciiDigit(c)));
        Assert.Equal(FinishReason.Stop, r.Finish);
    }

    [Fact]
    public async Task Metrics_ReportKvUsageAndBatchSize()
    {
        var d = new FakeBatchDecoder(2, contextLength: 128);
        d.Script("hello");
        using var e = Engine(d);
        await e.Submit(Req("hi")).CompleteAsync(TestContext.Current.CancellationToken);
        var m = e.GetMetrics();
        Assert.Equal(256, m.KvCapacityTokens);
        Assert.True(m.KvTokens > 0);
        Assert.Equal(0, m.KvActiveTokens);
        Assert.True(m.StepSequences.Count > 0);
    }

    /// <summary>Exactly <c>length</c> ASCII digits, then the end.</summary>
    private sealed class DigitsConstraint(int length) : IOutputConstraint
    {
        public IConstraintMatcher CreateMatcher() => new Matcher(length);

        private sealed class Matcher(int length) : IConstraintMatcher
        {
            private int _count;
            public bool IsComplete => _count == length;
            public bool IsTerminated => _count == length;
            public void FillAllowed(TokenMask mask)
            {
                mask.Clear();
                if (_count < length)
                {
                    for (int c = '0'; c <= '9'; c++)
                    {
                        mask.Allow(c);
                    }
                }
            }

            public bool Accept(int token)
            {
                if (token < '0' || token > '9' || _count >= length)
                {
                    return false;
                }

                _count++;
                return true;
            }

            public void Rollback(int count) => _count -= count;
        }
    }
}
