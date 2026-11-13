using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

public class SamplingTests
{
    private static int[] Draw(SamplingParams p, float[] logits, int n, int[]? prompt = null)
    {
        var s = new RequestSampler(p, prompt ?? []);
        var res = new int[n];
        for (int i = 0; i < n; i++) { res[i] = s.Sample(logits); s.Accept(res[i]); }
        return res;
    }

    [Fact]
    public void Greedy_PicksArgMax()
    {
        Assert.All(Draw(SamplingParams.GreedyDefault, [0.1f, 3f, 2f], 5), t => Assert.Equal(1, t));
        Assert.All(Draw(new SamplingParams { Temperature = 0 }, [0.1f, 3f, 2f], 5), t => Assert.Equal(1, t));
    }

    [Fact]
    public void TopK1_EqualsGreedy()
        => Assert.All(Draw(new SamplingParams { TopK = 1, Seed = 1 }, [0.1f, 3f, 2f, 2.9f], 50), t => Assert.Equal(1, t));

    [Fact]
    public void Seed_IsDeterministic_AndDifferentSeedsDiffer()
    {
        var logits = Enumerable.Range(0, 50).Select(i => (float)Math.Sin(i)).ToArray();
        var a = Draw(new SamplingParams { Seed = 7, TopK = 10 }, logits, 40);
        var b = Draw(new SamplingParams { Seed = 7, TopK = 10 }, logits, 40);
        var c = Draw(new SamplingParams { Seed = 8, TopK = 10 }, logits, 40);
        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
    }

    [Fact]
    public void Temperature_FollowsSoftmaxDistribution()
    {
        // p = softmax([0, ln 3]) = [0.25, 0.75]
        var draws = Draw(new SamplingParams { Seed = 3 }, [0f, MathF.Log(3)], 20000);
        double p1 = draws.Count(t => t == 1) / 20000.0;
        Assert.InRange(p1, 0.73, 0.77);
    }

    [Fact]
    public void LowTemperature_Sharpens()
        => Assert.All(Draw(new SamplingParams { Temperature = 0.05f, Seed = 1 }, [1f, 1.5f, 0.2f], 100), t => Assert.Equal(1, t));

    [Fact]
    public void TopK_RestrictsCandidates()
    {
        var draws = Draw(new SamplingParams { TopK = 2, Seed = 5 }, [5f, 4.9f, 4.8f, 4.7f], 2000);
        Assert.DoesNotContain(draws, t => t >= 2);
        Assert.Contains(draws, t => t == 0); Assert.Contains(draws, t => t == 1);
    }

    [Fact]
    public void TopP_KeepsSmallestNucleus()
    {
        // probabilities ~ [0.7, 0.2, 0.05, 0.05]; top_p 0.8 keeps tokens 0 and 1 only.
        float[] logits = [MathF.Log(0.7f), MathF.Log(0.2f), MathF.Log(0.05f), MathF.Log(0.05f)];
        var draws = Draw(new SamplingParams { TopP = 0.8f, Seed = 9 }, logits, 3000);
        Assert.DoesNotContain(draws, t => t >= 2);
        Assert.Contains(draws, t => t == 1);
        var tiny = Draw(new SamplingParams { TopP = 0.1f, Seed = 9 }, logits, 200);
        Assert.All(tiny, t => Assert.Equal(0, t));
    }

    [Fact]
    public void MinP_DropsUnlikelyTokens()
    {
        float[] logits = [MathF.Log(0.6f), MathF.Log(0.3f), MathF.Log(0.05f), MathF.Log(0.05f)];
        var draws = Draw(new SamplingParams { MinP = 0.3f, Seed = 2 }, logits, 3000);   // threshold 0.18 -> tokens 0, 1
        Assert.DoesNotContain(draws, t => t >= 2);
        Assert.Contains(draws, t => t == 1);
    }

    [Fact]
    public void RepetitionPenalty_DiscouragesPromptAndOutputTokens()
    {
        var p = new SamplingParams { Greedy = true, RepetitionPenalty = 2f };
        Assert.Equal(1, Draw(p, [2f, 1.9f], 1, prompt: [0])[0]);
        // A token generated earlier is penalized from then on: two near-tied tokens alternate.
        var seq = Draw(p, [2f, 1.5f], 4);
        Assert.Equal([0, 1, 0, 0], seq);   // 0 (2 vs 1.5); 0 penalized to 1 -> 1; 1 penalized to 0.75 -> 0 again (1 > 0.75)
    }

    [Fact]
    public void RepetitionPenalty_NegativeLogitsAreMultiplied()
    {
        var p = new SamplingParams { Greedy = true, RepetitionPenalty = 2f };
        Assert.Equal(1, Draw(p, [-1f, -1.5f], 1, prompt: [0])[0]);   // -1 * 2 = -2 < -1.5
    }

    [Fact]
    public void PresenceAndFrequencyPenalty_AccumulateOverGeneratedTokens()
    {
        var p = new SamplingParams { Greedy = true, FrequencyPenalty = 1f };
        var seq = Draw(p, [3f, 2.5f, 0f], 6);
        // token 0 loses 1 per use: 3, then 2 (< 2.5) -> switch to 1 ... both converge to alternating.
        Assert.Equal(0, seq[0]);
        Assert.Equal(1, seq[1]);
        var pres = Draw(new SamplingParams { Greedy = true, PresencePenalty = 1f }, [3f, 2.5f, 0f], 3);
        Assert.Equal([0, 1, 0], pres);   // 3 -> 2 (used), 2.5 -> 1.5 (used), then 0:2 vs 1:1.5
    }

    [Fact]
    public void LogitBias_ForcesAndForbidsTokens()
    {
        var forbid = new SamplingParams { Greedy = true, LogitBias = new Dictionary<int, float> { [1] = -100f } };
        Assert.Equal(0, Draw(forbid, [1f, 5f], 1)[0]);
        var force = new SamplingParams { Greedy = true, LogitBias = new Dictionary<int, float> { [0] = 100f } };
        Assert.Equal(0, Draw(force, [1f, 5f], 1)[0]);
        var sampled = new SamplingParams { Seed = 1, LogitBias = new Dictionary<int, float> { [1] = -100f } };
        Assert.All(Draw(sampled, [1f, 5f], 200), t => Assert.Equal(0, t));
    }

    [Fact]
    public void Logprobs_AreNormalizedLogSoftmax()
    {
        float[] logits = [1f, 2f, 3f, 4f];
        var (lp, top) = RequestSampler.Logprobs(logits, 2, 4);
        double z = logits.Sum(x => Math.Exp(x));
        Assert.Equal(Math.Log(Math.Exp(3) / z), lp, 4);
        Assert.Equal([3, 2, 1, 0], top.Select(t => t.Id));
        Assert.Equal(1.0, top.Sum(t => Math.Exp(t.Logprob)), 4);
    }

    [Fact]
    public void LargeVocabulary_TopK_IsFast()
    {
        var rng = new Random(1);
        var logits = Enumerable.Range(0, 152_000).Select(_ => (float)rng.NextDouble() * 10).ToArray();
        var s = new RequestSampler(new SamplingParams { TopK = 40, TopP = 0.95f, Seed = 1 }, []);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < 50; i++)
        {
            s.Sample(logits);
        }

        Assert.True(sw.Elapsed.TotalMilliseconds / 50 < 10, $"{sw.Elapsed.TotalMilliseconds / 50} ms per sample");
    }
}
