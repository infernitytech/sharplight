using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// The two admission policies of a paged KV cache that runs short (<see cref="EngineOptions.KvAdmission"/>): queueing (default) waits for
/// room for a prompt and its growth; preemption admits on the prompt alone and recomputes the newest sequences when the running ones
/// outgrow the pool. Replies here depend on the whole context, so a recomputation that lost or reordered a token changes the text.
/// </summary>
public class KvAdmissionTests
{
    private static readonly TimeSpan Limit = TimeSpan.FromSeconds(60);

    [Fact]
    public void KvAdmission_DefaultsToQueue()
    {
        Assert.Equal(KvAdmissionPolicy.Queue, new EngineOptions().KvAdmission);
        Assert.Equal(KvAdmissionPolicy.Queue, default(KvAdmissionPolicy));
    }

    /// <summary>
    /// Two 30-token prompts that generate 10 tokens each (10 blocks of 4) in a pool of 16 blocks. Queueing admits the second only when
    /// the first is done (its prompt plus an 8-token growth reserve does not fit next to the first's); preemption admits both on their
    /// prompts, and when they outgrow the pool the newer one gives its blocks back and is recomputed. The texts are the same.
    /// </summary>
    [Theory]
    [InlineData(KvAdmissionPolicy.Queue)]
    [InlineData(KvAdmissionPolicy.Preempt)]
    public async Task TwoPromptsThatOutgrowThePool(KvAdmissionPolicy policy)
    {
        string[] prompts = [new string('a', 29), new string('b', 29)];
        var expected = await Reference(prompts, 10);

        var d = Decoder(2, 16);
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { KvAdmission = policy });
        var results = await Run(e, prompts, 10);

        Assert.Equal(expected, results.Select(r => r.Text));
        Assert.All(results, r => Assert.Equal(10, r.Usage.CompletionTokens));
        Assert.True(d.PeakBlocksHeld <= 16);
        bool together = d.Steps.Any(s => s.Select(x => x.Slot).Distinct().Count() == 2);
        if (policy == KvAdmissionPolicy.Queue)
        {
            Assert.Equal(0, e.GetMetrics().Preemptions);
            Assert.False(together, "queueing admitted the second prompt before the first finished");
        }
        else
        {
            Assert.True(e.GetMetrics().Preemptions >= 1, "expected a preemption");
            Assert.True(together, "preemption should admit both prompts at once");
            // The preempted context is fed again without logits.
            Assert.Contains(d.Steps, s => s.Any(x => !x.Logits && x.Rows > 1));
        }
    }

    /// <summary>
    /// Twelve requests of different lengths over six slots and a pool that holds about two of them at full length: under preemption
    /// sequences are preempted again and again, and every request still finishes with the text it has with a pool large enough for
    /// all (no request starves, no recomputation changes a token).
    /// </summary>
    [Fact]
    public async Task Preempt_ManySequences_AllFinishWithTheirReferenceText()
    {
        string[] prompts = [.. Enumerable.Range(0, 12).Select(i => new string((char)('c' + i), 6 + 5 * (i % 4)))];
        int[] maxTokens = [.. Enumerable.Range(0, 12).Select(i => 12 + 7 * (i % 3))];
        var expected = await Reference(prompts, maxTokens);

        var d = Decoder(6, 22);
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { KvAdmission = KvAdmissionPolicy.Preempt, PrefillChunk = 8 });
        var results = await Run(e, prompts, maxTokens);

        Assert.Equal(expected, results.Select(r => r.Text));
        Assert.All(results, r => Assert.Equal(FinishReason.Length, r.Finish));
        Assert.True(e.GetMetrics().Preemptions >= 2, $"expected repeated preemptions, got {e.GetMetrics().Preemptions}");
        Assert.True(d.PeakBlocksHeld <= 22);
    }

    /// <summary>The same workload under queueing: the same texts with the default policy.</summary>
    [Fact]
    public async Task Queue_ManySequences_AllFinishWithTheirReferenceText()
    {
        string[] prompts = [.. Enumerable.Range(0, 12).Select(i => new string((char)('c' + i), 6 + 5 * (i % 4)))];
        int[] maxTokens = [.. Enumerable.Range(0, 12).Select(i => 12 + 7 * (i % 3))];
        var expected = await Reference(prompts, maxTokens);

        var d = Decoder(6, 22);
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { PrefillChunk = 8 });
        var results = await Run(e, prompts, maxTokens);

        Assert.Equal(expected, results.Select(r => r.Text));
        Assert.True(d.PeakBlocksHeld <= 22);
    }

    /// <summary>A paged fake (blocks of 4 tokens) whose next token is a hash of the whole context.</summary>
    private static FakeBatchDecoder Decoder(int slots, int blocks)
    {
        var d = new FakeBatchDecoder(slots, kvBlockSize: blocks > 0 ? 4 : 0, kvBlocks: blocks);
        d.Next = ContextHash;
        return d;
    }

    private static int ContextHash(IReadOnlyList<int> context)
    {
        uint h = 2166136261;
        foreach (int t in context)
        {
            h = (h ^ (uint)t) * 16777619;
        }

        return 'A' + (int)(h % 26);
    }

    private static Task<string[]> Reference(string[] prompts, int maxTokens) => Reference(prompts, [.. prompts.Select(_ => maxTokens)]);

    /// <summary>The texts of the requests with one slot each and no paged cache (nothing preempted, nothing waits).</summary>
    private static async Task<string[]> Reference(string[] prompts, int[] maxTokens)
    {
        using var e = new InferenceEngine(Decoder(prompts.Length, 0), new FakeTokenizer());
        return [.. (await Run(e, prompts, maxTokens)).Select(r => r.Text)];
    }

    private static Task<GenerationResult[]> Run(InferenceEngine e, string[] prompts, int maxTokens) => Run(e, prompts, [.. prompts.Select(_ => maxTokens)]);

    private static async Task<GenerationResult[]> Run(InferenceEngine e, string[] prompts, int[] maxTokens)
    {
        var handles = prompts.Select((p, i) => e.Submit(new GenerationRequest
        {
            PromptTokens = T.Prompt(p), MaxTokens = maxTokens[i], Sampling = SamplingParams.GreedyDefault,
        })).ToList();
        return await Task.WhenAll(handles.Select(h => h.CompleteAsync())).WaitAsync(Limit);
    }
}
