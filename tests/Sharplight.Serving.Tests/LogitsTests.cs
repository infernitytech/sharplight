using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Speculative;
using Sharplight.Llm.Structured;

namespace Sharplight.Serving.Tests;

/// <summary>Vectorized arg-max, token masks and the n-gram lookup.</summary>
public class LogitsTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(33)]
    [InlineData(1000)]
    [InlineData(128_257)]
    public void ArgMax_MatchesScalarScan(int n)
    {
        var rng = new Random(n);
        var v = new float[n];
        for (int i = 0; i < n; i++)
        {
            v[i] = (float)rng.NextDouble();
        }

        int expected = 0;
        for (int i = 1; i < n; i++)
        {
            if (v[i] > v[expected])
            {
                expected = i;
            }
        }

        Assert.Equal(expected, LogitsMath.ArgMax(v));
    }

    [Fact]
    public void ArgMax_FirstOfTies_AndIgnoresNaN()
    {
        var v = new float[100];
        v[40] = 3;
        v[70] = 3;
        v[10] = float.NaN;
        Assert.Equal(40, LogitsMath.ArgMax(v));
        Assert.Equal(0, LogitsMath.ArgMax([float.NaN, float.NaN]));
        Assert.Equal(2, LogitsMath.ArgMax([float.NegativeInfinity, float.NaN, -5f]));
    }

    [Fact]
    public void TokenMask_AllowApplyCount()
    {
        var m = new TokenMask(70);
        m.Allow(3);
        m.Allow(64);
        m.Allow(69);
        m.Allow(500);
        Assert.Equal(3, m.Count());
        Assert.True(m.IsAllowed(64));
        Assert.False(m.IsAllowed(65));
        var logits = Enumerable.Repeat(1f, 72).ToArray();
        m.Apply(logits);
        for (int i = 0; i < logits.Length; i++)
        {
            Assert.Equal(i is 3 or 64 or 69 ? 1f : float.NegativeInfinity, logits[i]);
        }

        m.AllowAll();
        Assert.Equal(70, m.Count());
        m.Disallow(0);
        Assert.Equal(69, m.Count());
    }

    [Fact]
    public void Ngram_ProposesContinuationOfMostRecentMatch()
    {
        var spec = new NgramSpeculator(maxDraftTokens: 3, maxNgram: 3, minNgram: 2);
        var draft = new int[3];
        // ... 1 2 3 4 5 ... 1 2 3 9 9 ... 1 2  -> the most recent "1 2" was followed by 3 9 9.
        int[] ctx = [7, 1, 2, 3, 4, 5, 8, 1, 2, 3, 9, 9, 6, 1, 2];
        Assert.Equal(3, spec.Lookup(ctx, draft));
        Assert.Equal([3, 9, 9], draft);
        Assert.Equal(0, spec.Lookup([1, 2, 3, 4], draft));
    }
}
