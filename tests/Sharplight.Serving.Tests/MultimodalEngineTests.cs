using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Image and audio rows in the batching engine (<see cref="GenerationRequest.PromptMedia"/> on the <see cref="FakeBatchDecoder"/>): every row
/// reaches the decoder with its embedding and its rope position (explicit in the prompt, offset after it), media are encoded in one stage per
/// step (each content once) and reused from the encoder-output cache and the prefix cache, preempted sequences feed their media rows again,
/// and bad requests or a failing encoder end only the requests concerned.
/// </summary>
public sealed class MultimodalEngineTests
{
    private const int Width = 4;

    /// <summary>Counts its calls; each content's rows are its value plus the row index (first column), so the decoder sees where they came from.</summary>
    private sealed class FakeEncoder : IMediaEncoder
    {
        public int EmbeddingWidth => Width;
        public List<int> Calls { get; } = [];
        public bool Fail { get; set; }

        public IReadOnlyList<float[]> Encode(IReadOnlyList<MediaContent> contents)
        {
            Calls.Add(contents.Count);
            if (Fail)
            {
                throw new InvalidOperationException("broken encoder");
            }

            return [.. contents.Select(c =>
            {
                var rows = new float[c.Rows * Width];
                for (int r = 0; r < c.Rows; r++)
                {
                    rows[r * Width] = ((FakeContent)c).Value + r;
                }

                return rows;
            })];
        }
    }

    private sealed class FakeContent(FakeEncoder encoder, float value, int rows) : MediaContent((UInt128)value, rows)
    {
        public float Value { get; } = value;
        public override IMediaEncoder Encoder => encoder;
    }

    /// <summary>
    /// "prefix" + an image of <paramref name="rows"/> placeholder rows + "suffix" + marker, with M-RoPE-like positions: the image's rows share
    /// one position each 4 rows (a compressed span), the text after it continues from the image's last position + 1.
    /// </summary>
    private static GenerationRequest Image(FakeEncoder encoder, string prefix, float value, int rows, string suffix, int max = 4, string? partition = null)
    {
        var tokens = new List<int>(T.Prompt(prefix)[..^1]);
        int start = tokens.Count;
        tokens.AddRange(Enumerable.Repeat((int)'#', rows));
        tokens.AddRange(T.Prompt(suffix));
        var positions = new int[tokens.Count];
        int p = 0;
        for (int i = 0; i < tokens.Count; i++)
        {
            bool image = i >= start && i < start + rows;
            positions[i] = image ? start + (i - start) / 4 : p;
            p = image ? start + (i - start) / 4 + 1 : p + 1;
        }

        return new GenerationRequest
        {
            PromptTokens = [.. tokens], MaxTokens = max, Sampling = SamplingParams.GreedyDefault, CachePartition = partition,
            PromptMedia = [new PromptMedia(start, new FakeContent(encoder, value, rows))],
            PromptPositions = positions,
            NextPosition = p,
        };
    }

    private static FakeBatchDecoder Decoder(int slots = 2, int blocks = 64, int batch = 64)
    {
        var d = new FakeBatchDecoder(slots, maxBatchTokens: batch, kvBlockSize: 4, kvBlocks: blocks) { EmbeddingWidth = Width };
        d.Script("0123");
        return d;
    }

    [Fact]
    public async Task MediaRowsAndPositionsReachTheDecoder()
    {
        var encoder = new FakeEncoder();
        var d = Decoder(batch: 16);
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var request = Image(encoder, "look: ", 100, 20, "what?");
        var result = await e.Submit(request).CompleteAsync();
        Assert.Null(result.Error);
        Assert.Equal("0123", result.Text);
        var media = request.PromptMedia[0];
        int n = request.PromptTokens.Length;
        var rows = d.Rows.OrderBy(r => r.Position).ToList();
        Assert.Equal(n + 3, rows.Count);   // the prompt, then the generated tokens fed back (the last one is not fed)
        for (int i = 0; i < rows.Count; i++)
        {
            var (_, position, rope, embedding) = rows[i];
            Assert.Equal(i, position);
            Assert.Equal(i < n ? request.PromptPositions![i] : request.NextPosition!.Value + (i - n), rope);
            Assert.Equal(i >= media.Start && i < media.End ? 100 + i - media.Start : float.NaN, embedding);
        }

        // The prompt was cut into chunks of the step budget: the image's rows crossed chunks and still arrived in order.
        Assert.True(d.Steps.Count(s => s.Any(x => x.Rows > 1)) > 1);
        Assert.Equal([1], encoder.Calls);
    }

    [Fact]
    public async Task OneImageInSeveralPromptsIsEncodedOnceAndCached()
    {
        var encoder = new FakeEncoder();
        var d = Decoder(slots: 4);
        using var e = new InferenceEngine(d, new FakeTokenizer());
        // Same image, different text before it (no shared blocks): encoded once in one stage, then served from the encoder-output cache.
        var together = await Task.WhenAll(
            e.Submit(Image(encoder, "aaaa ", 7, 8, "x")).CompleteAsync(),
            e.Submit(Image(encoder, "bbbb ", 7, 8, "y")).CompleteAsync(),
            e.Submit(Image(encoder, "cccc ", 9, 8, "z")).CompleteAsync());
        Assert.All(together, r => Assert.Null(r.Error));
        // Each distinct content is encoded exactly once. Whether both land in one encoder call depends on whether the engine loop
        // admitted the first prompt before the others were submitted, so the number of calls is not asserted.
        Assert.Equal(2, encoder.Calls.Sum());
        var later = await e.Submit(Image(encoder, "dddd ", 7, 8, "w")).CompleteAsync();
        Assert.Null(later.Error);
        Assert.Equal(2, encoder.Calls.Sum());
        var m = e.GetMetrics();
        Assert.Equal(2, m.MediaEncoded);
        Assert.Equal(1, m.MediaCacheHits);
    }

    [Fact]
    public async Task ARepeatedImagePromptIsNeitherEncodedNorPrefilledAgain()
    {
        var encoder = new FakeEncoder();
        var d = Decoder();
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { MediaCacheBytes = 0 });
        var request = Image(encoder, "describe ", 5, 24, "please");
        await e.Submit(request).CompleteAsync();
        int fed = d.Rows.Count;
        var again = await e.Submit(request).CompleteAsync();
        Assert.Equal("0123", again.Text);
        Assert.Equal([1], encoder.Calls);
        Assert.True(again.Usage.CachedTokens >= request.PromptMedia[0].End);
        Assert.DoesNotContain(d.Rows.Skip(fed), r => !float.IsNaN(r.Embedding));

        // Another image at the same place shares only the blocks before it.
        var other = await e.Submit(Image(encoder, "describe ", 6, 24, "please")).CompleteAsync();
        Assert.Equal([1, 1], encoder.Calls);
        Assert.True(other.Usage.CachedTokens <= request.PromptMedia[0].Start);
    }

    [Fact]
    public async Task APreemptedSequenceFeedsItsImageRowsAgain()
    {
        var encoder = new FakeEncoder();
        // A pool too small for both: the newer sequence is preempted and recomputed (prefix caching off, so it recomputes rows).
        var d = new FakeBatchDecoder(2, maxBatchTokens: 64, kvBlockSize: 4, kvBlocks: 22) { EmbeddingWidth = Width };
        d.Script("0123456789abcdef", loop: true);
        using var e = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { PrefixCache = false });
        var first = Image(encoder, "one ", 100, 16, "a", max: 40);
        var second = Image(encoder, "two ", 200, 16, "b", max: 40);
        var results = await Task.WhenAll(e.Submit(first).CompleteAsync(), e.Submit(second).CompleteAsync());
        Assert.All(results, r => Assert.Null(r.Error));
        Assert.True(e.GetMetrics().Preemptions > 0);
        // Every fed image row carries its own embedding, also when it was fed again, at its place and with the prompt's position.
        foreach (var (_, position, rope, embedding) in d.Rows.Where(r => !float.IsNaN(r.Embedding)))
        {
            var request = embedding < 200 ? first : second;
            float value = embedding < 200 ? 100 : 200;
            Assert.Equal(position - request.PromptMedia[0].Start, embedding - value);
            Assert.Equal(request.PromptPositions![position], rope);
        }

        Assert.True(d.Rows.Count(r => !float.IsNaN(r.Embedding)) > 32);
    }

    [Fact]
    public void BadMediaIsRejectedAtSubmission()
    {
        var encoder = new FakeEncoder();
        using var e = new InferenceEngine(Decoder(), new FakeTokenizer());
        var ok = Image(encoder, "x ", 1, 8, "y");
        var outside = ok with { PromptMedia = [new PromptMedia(ok.PromptTokens.Length - 2, new FakeContent(encoder, 1, 8))] };
        Assert.Equal("invalid_request", Assert.Throws<EngineException>(() => e.Submit(outside)).Code);
        var positions = ok with { PromptPositions = [1, 2, 3] };
        Assert.Equal("invalid_request", Assert.Throws<EngineException>(() => e.Submit(positions)).Code);

        using var text = new InferenceEngine(new FakeBatchDecoder(1), new FakeTokenizer());
        Assert.Contains("without image or audio input", Assert.Throws<EngineException>(() => text.Submit(ok)).Message);
    }

    [Fact]
    public async Task AFailingEncoderEndsOnlyTheRequestsThatNeedIt()
    {
        var encoder = new FakeEncoder { Fail = true };
        using var e = new InferenceEngine(Decoder(), new FakeTokenizer());
        var failed = await e.Submit(Image(encoder, "x ", 1, 8, "y")).CompleteAsync();
        Assert.Equal(FinishReason.Error, failed.Finish);
        Assert.Contains("broken encoder", failed.Error);
        var text = await e.Submit(new GenerationRequest { PromptTokens = T.Prompt("plain"), MaxTokens = 4, Sampling = SamplingParams.GreedyDefault }).CompleteAsync();
        Assert.Equal("0123", text.Text);
    }
}
