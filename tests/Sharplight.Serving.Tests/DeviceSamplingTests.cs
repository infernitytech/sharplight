using Sharplight.Graph;
using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Routing between device and CPU sampling in the engine, the mapping of request parameters onto device sampling rows, and the
/// CPU sampler's distribution against the filters' exact distribution (the device kernel is checked against the same target in
/// the Vulkan tests).
/// </summary>
public class DeviceSamplingTests
{
    private static GenerationRequest Req(string prompt, SamplingParams sampling, bool logprobs = false, int topLogprobs = 0) => new()
    {
        PromptTokens = T.Prompt(prompt), MaxTokens = 100, Sampling = sampling, Logprobs = logprobs, TopLogprobs = topLogprobs,
    };

    [Fact]
    public async Task GreedyRequests_AreSampledOnTheDevice()
    {
        var d = new FakeDeviceSamplingDecoder(new FakeBatchDecoder(2));
        d.Inner.Script("hello");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var r = await e.Submit(Req("hi", SamplingParams.GreedyDefault)).CompleteAsync();
        Assert.Equal("hello", r.Text);
        Assert.Equal(6, d.SampledRows.Count);   // five letters and the end-of-generation token
        Assert.All(d.SampledRows, s => Assert.True(s.Row.IsGreedy));
    }

    [Fact]
    public async Task SampledRequest_AdvancesTheCounterWithAFixedSeed()
    {
        var d = new FakeDeviceSamplingDecoder(new FakeBatchDecoder(1));
        d.Inner.Script("abcd");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var sampling = new SamplingParams { Temperature = 0.7f, TopK = 40, TopP = 0.9f, MinP = 0.05f, Seed = 5 };
        var r = await e.Submit(Req("x", sampling)).CompleteAsync();
        Assert.Equal("abcd", r.Text);
        Assert.Equal([0u, 1u, 2u, 3u, 4u], d.SampledRows.Select(s => s.Row.Counter));
        Assert.All(d.SampledRows, s => Assert.Equal(new SamplingRow
        {
            Temperature = 0.7f, TopK = 40, TopP = 0.9f, MinP = 0.05f, Seed = 5, Counter = s.Row.Counter,
        }, s.Row));
    }

    [Fact]
    public async Task PenaltiesAndBias_FallBackToTheCpu()
    {
        var d = new FakeDeviceSamplingDecoder(new FakeBatchDecoder(2));
        d.Inner.Script("abc");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var penalized = e.Submit(Req("x", SamplingParams.GreedyDefault with { RepetitionPenalty = 1.2f }));
        var biased = e.Submit(Req("y", SamplingParams.GreedyDefault with { LogitBias = new Dictionary<int, float> { [7] = 1f } }));
        var results = await Task.WhenAll(penalized.CompleteAsync(), biased.CompleteAsync());
        Assert.All(results, r => Assert.Equal("abc", r.Text));
        Assert.Empty(d.SampledRows);
    }

    [Fact]
    public async Task Logprobs_UseTheDeviceOnlyWithoutAlternatives()
    {
        var d = new FakeDeviceSamplingDecoder(new FakeBatchDecoder(1));
        d.Inner.Script("ab");
        using var e = new InferenceEngine(d, new FakeTokenizer());
        var device = await e.Submit(Req("x", SamplingParams.GreedyDefault, logprobs: true)).CompleteAsync();
        Assert.Equal(3, d.SampledRows.Count);
        Assert.All(d.SampledRows, s => Assert.True(s.Row.WantLogprob));
        Assert.Equal([-0.5f, -0.5f], device.Logprobs.Select(l => l.Logprob));
        Assert.All(device.Logprobs, l => Assert.Empty(l.TopAlternatives));

        d.SampledRows.Clear();
        var cpu = await e.Submit(Req("x", SamplingParams.GreedyDefault, logprobs: true, topLogprobs: 2)).CompleteAsync();
        Assert.Empty(d.SampledRows);
        Assert.All(cpu.Logprobs, l => Assert.Equal(2, l.TopAlternatives.Count));
    }

    [Fact]
    public void DeviceRows_CoverWhatTheKernelCanExpress()
    {
        static SamplingRow? Row(SamplingParams p) => new RequestSampler(p, []).DeviceRow(3, false, 0);
        Assert.True(Row(SamplingParams.GreedyDefault)!.Value.IsGreedy);
        Assert.Equal(3u, Row(new SamplingParams { Temperature = 1, Seed = 1 })!.Value.Counter);
        Assert.NotNull(Row(new SamplingParams { Temperature = 1, TopK = 50000 }));   // top-k alone has no candidate limit
        Assert.Null(Row(new SamplingParams { Temperature = 1, TopK = 5000, TopP = 0.9f }));
        Assert.Null(Row(new SamplingParams { Temperature = 1, PresencePenalty = 0.5f }));
        Assert.Null(Row(new SamplingParams { Temperature = 1, FrequencyPenalty = 0.5f }));
        Assert.Null(new RequestSampler(SamplingParams.GreedyDefault, []).DeviceRow(0, true, 1));
        // Unseeded requests draw one seed per request: the stream stays the same across the request's tokens.
        var unseeded = new RequestSampler(new SamplingParams { Temperature = 1 }, []);
        Assert.Equal(unseeded.DeviceRow(0, false, 0)!.Value.Seed, unseeded.DeviceRow(1, false, 0)!.Value.Seed);
    }

    public static TheoryData<string, float, int, float, float> Configurations => new()
    {
        { "temperature", 0.8f, 0, 1f, 0f },
        { "top-k 40", 1f, 40, 1f, 0f },
        { "top-p 0.9", 0.7f, 0, 0.9f, 0f },
        { "top-k 40 top-p 0.95", 0.7f, 40, 0.95f, 0f },
        { "min-p 0.05", 1f, 0, 1f, 0.05f },
        { "all", 1.3f, 64, 0.8f, 0.02f },
    };

    [Theory]
    [MemberData(nameof(Configurations))]
    public void CpuSampler_DrawsTheFilteredDistribution(string name, float temperature, int topK, float topP, float minP)
    {
        const int Vocab = 300;
        const int Draws = 25600;
        var rng = new Random(7);
        var x = new float[Vocab];
        for (int i = 0; i < Vocab; i++)
        {
            x[i] = (float)(rng.NextDouble() * 6 - 3);
        }

        for (int j = 0; j < 20; j++)
        {
            x[rng.Next(Vocab)] = 4 + (float)rng.NextDouble() * 8;
        }

        var sampler = new RequestSampler(new SamplingParams { Temperature = temperature, TopK = topK, TopP = topP, MinP = minP, Seed = 3 }, []);
        var counts = new int[Vocab];
        for (int i = 0; i < Draws; i++)
        {
            counts[sampler.Sample(x)]++;
        }

        var target = TargetDistribution(x, temperature, topK, topP, minP);
        double tv = 0;
        double expected = 0;
        for (int i = 0; i < Vocab; i++)
        {
            Assert.False(target[i] == 0 && counts[i] > 0, $"{name}: token {i} is outside the filtered set but was drawn {counts[i]} times");
            tv += Math.Abs(counts[i] / (double)Draws - target[i]) / 2;
            expected += Math.Sqrt(2 * target[i] * (1 - target[i]) / (Math.PI * Draws)) / 2;
        }

        Assert.True(tv < 1.5 * expected + 0.005, $"{name}: total variation {tv:F4} between drawn and target distribution (noise level {expected:F4})");
    }

    /// <summary>The distribution the filters define (see <see cref="SamplingHead"/>), in double precision.</summary>
    private static double[] TargetDistribution(float[] x, float temperature, int topK, float topP, float minP)
    {
        int n = x.Length;
        var order = Enumerable.Range(0, n).OrderByDescending(i => x[i]).ThenBy(i => i).ToArray();
        double m = x[order[0]];
        var p = order.Select(i => Math.Exp((x[i] - m) / temperature)).ToArray();
        int kept = topK > 0 && topK < n ? topK : n;
        double z = p.Take(kept).Sum();
        int cut = kept;
        for (int j = 1; j < kept && minP > 0; j++)
        {
            if (p[j] < minP)
            {
                cut = j;
                break;
            }
        }

        if (topP < 1)
        {
            double acc = 0;
            for (int j = 0; j < cut; j++)
            {
                acc += p[j];
                if (acc >= topP * z)
                {
                    cut = j + 1;
                    break;
                }
            }
        }

        double total = p.Take(cut).Sum();
        var target = new double[n];
        for (int j = 0; j < cut; j++)
        {
            target[order[j]] = p[j] / total;
        }

        return target;
    }
}
