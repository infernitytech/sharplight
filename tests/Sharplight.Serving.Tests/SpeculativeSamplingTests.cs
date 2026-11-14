using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Speculative decoding must not change what is sampled: with temperature sampling, drafts that are often but not always right
/// (the most likely token, probability about 0.47), the joint distribution of the first tokens equals the model's exact
/// distribution (chi-square goodness of fit).
/// </summary>
public class SpeculativeSamplingTests
{
    private const int First = 'a';
    private const int Choices = 4;
    private const int Length = 3;
    private const int Samples = 4000;

    // The fake model: tokens 'a'..'d' have logit 9, the context's "next" token 10, everything else -30.
    private static int MostLikely(IReadOnlyList<int> c) => First + ((c[^1] + (2 * c[^2])) % Choices);

    private static float[] Bias()
    {
        var bias = new float[260];
        Array.Fill(bias, -30f);
        for (int t = First; t < First + Choices; t++)
        {
            bias[t] = 9f;
        }

        return bias;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SampledDistribution_Matches_TheModel(bool speculate)
    {
        var d = new FakeBatchDecoder(8, maxLogitsRows: 32) { Bias = Bias(), Next = MostLikely };
        var spec = speculate ? new GreedyGuessSpeculator(3, MostLikely) : null;
        using var engine = new InferenceEngine(d, new FakeTokenizer(), new EngineOptions { Speculator = spec, MaxQueue = 1 << 20 });
        Assert.Equal(speculate, engine.Speculates);
        int[] prompt = T.Prompt("xy");
        var ct = TestContext.Current.CancellationToken;
        var counts = new Dictionary<string, int>();
        for (int batch = 0; batch < Samples; batch += 200)
        {
            var handles = Enumerable.Range(batch, 200).Select(i => engine.Submit(new GenerationRequest
            {
                PromptTokens = prompt, MaxTokens = Length, StopAtEog = false, Logprobs = true,
                Sampling = new SamplingParams { Temperature = 1f, Seed = i },
            }, ct)).ToList();
            foreach (var r in await Task.WhenAll(handles.Select(h => h.CompleteAsync(ct))))
            {
                string key = string.Concat(r.Logprobs.Select(l => (char)l.TokenId));
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }
        }

        // Exact probabilities of every sequence of 'a'..'d' (the rest has negligible mass).
        double chi2 = 0;
        int cells = 0;
        foreach (var (sequence, p) in Sequences(prompt))
        {
            double expected = p * Samples;
            double observed = counts.GetValueOrDefault(sequence);
            chi2 += (observed - expected) * (observed - expected) / expected;
            cells++;
        }

        Assert.Equal(64, cells);
        Assert.Equal(Samples, counts.Where(kv => kv.Key.All(ch => ch >= First && ch < First + Choices)).Sum(kv => kv.Value));
        // 63 degrees of freedom: P(chi2 > 110) is about 2e-4.
        Assert.True(chi2 < 110, $"chi-square {chi2:F1} over {cells} cells");
        if (speculate)
        {
            var m = engine.GetMetrics();
            Assert.True(m.AcceptedDraftTokens > 0 && m.AcceptedDraftTokens < m.DraftTokens, $"accepted {m.AcceptedDraftTokens} of {m.DraftTokens}");
        }
    }

    private static IEnumerable<(string Sequence, double P)> Sequences(int[] prompt)
    {
        var bias = Bias();
        IEnumerable<(List<int> Context, string Text, double P)> level = [([.. prompt], "", 1.0)];
        for (int step = 0; step < Length; step++)
        {
            level = level.SelectMany(item =>
            {
                var logits = bias.Select(x => (double)x).ToArray();
                logits[MostLikely(item.Context)] = 10;
                double z = logits.Sum(Math.Exp);
                return Enumerable.Range(First, Choices).Select(t => (Context: (List<int>)[.. item.Context, t], item.Text + (char)t, item.P * Math.Exp(logits[t]) / z));
            }).ToList();
        }

        return level.Select(x => (x.Text, x.P));
    }
}
