using Sharplight.Llm;
using Sharplight.Llm.Serving;

namespace Sharplight.Serving.Tests;

/// <summary>
/// Seeds became 64-bit in 0.3: a seed within the 32-bit range must reproduce the token choices of the releases with 32-bit seeds. The expected draws were
/// recorded with the 32-bit implementation (host sampling through <see cref="Random"/>, the device row's seed as the seed's bits); both samplers drew the same.
/// </summary>
public class SeedCompatibilityTests
{
    private static readonly float[] s_logits = [0.3f, 1.2f, -0.5f, 2.0f, 0.9f, 1.7f, -1.1f, 0.0f];

    public static TheoryData<long, int[]> Draws => new()
    {
        { 7, [3, 4, 5, 3, 3, 1, 3, 0, 4, 4, 5, 4, 5, 3, 4, 5, 1, 3, 4, 3, 1, 5, 3, 5, 3, 3, 0, 3, 5, 1, 1, 3] },
        { -123, [0, 4, 1, 1, 1, 3, 3, 3, 3, 5, 4, 5, 3, 5, 3, 4, 5, 3, 3, 4, 3, 3, 1, 3, 1, 5, 1, 4, 5, 3, 5, 4] },
        { int.MaxValue, [1, 1, 1, 5, 3, 5, 4, 5, 0, 3, 3, 5, 5, 5, 0, 3, 4, 0, 1, 3, 1, 4, 0, 3, 1, 5, 4, 1, 5, 3, 3, 5] },
        { int.MinValue, [1, 1, 1, 5, 3, 5, 4, 5, 0, 3, 3, 5, 5, 5, 0, 3, 4, 0, 1, 3, 1, 4, 0, 3, 1, 5, 4, 1, 5, 3, 3, 5] },
    };

    [Theory]
    [MemberData(nameof(Draws))]
    public void ServingSamplerDrawsAsWith32BitSeeds(long seed, int[] expected)
    {
        var sampler = new RequestSampler(new SamplingParams { Temperature = 0.9f, TopK = 6, TopP = 0.95f, Seed = seed }, []);
        Assert.Equal(expected, Draw(logits => sampler.Sample(logits), sampler.Accept));
        Assert.Equal(unchecked((uint)(int)seed), sampler.DeviceRow(0, false, 0)!.Value.Seed);
    }

    [Theory]
    [MemberData(nameof(Draws))]
    public void GenerationSamplerDrawsAsWith32BitSeeds(long seed, int[] expected)
    {
        var sampler = new Sampler(new SamplingOptions { Temperature = 0.9f, TopK = 6, TopP = 0.95f, Seed = seed });
        Assert.Equal(expected, Draw(logits => sampler.Sample(logits), sampler.Accept));
        Assert.Equal(unchecked((uint)(int)seed), sampler.DeviceRow(0)!.Value.Seed);
    }

    [Fact]
    public void SeedsBeyond32BitsAreDistinct()
    {
        var a = new RequestSampler(new SamplingParams { Temperature = 1f, Seed = 1L << 40 }, []);
        var b = new RequestSampler(new SamplingParams { Temperature = 1f, Seed = (1L << 40) + 1 }, []);
        Assert.NotEqual(a.DeviceRow(0, false, 0)!.Value.Seed, b.DeviceRow(0, false, 0)!.Value.Seed);
    }

    private static int[] Draw(Func<float[], int> sample, Action<int> accept)
    {
        var tokens = new int[32];
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens[i] = sample(s_logits);
            accept(tokens[i]);
        }

        return tokens;
    }
}
