using System.Runtime.Intrinsics;

namespace Sharplight.Cpu.Tests;

/// <summary><see cref="VectorExp"/> against the double-precision exponential, at every vector width.</summary>
public sealed class VectorExpTests
{
    [Fact]
    public void ExpIsWithinTwoUlp()
    {
        var rnd = new Random(5);
        var inputs = new List<float> { 0f, -0f, 1f, -1f, 0.5f, -87.3f, 88.7f, 1e-8f, -1e-8f };
        // Normal results only: below e^-87.34 the results flush to zero.
        for (int i = 0; i < 100_000; i++)
        {
            inputs.Add((float)(Math.Min(88.72, rnd.NextDouble() * 176 - 87.3)));
        }

        for (int i = 0; i + 16 <= inputs.Count; i += 16)
        {
            var x = inputs.GetRange(i, 16).ToArray();
            var y256 = new float[16];
            var y128 = new float[16];
            var y512 = new float[16];
            VectorExp.Exp(Vector256.Create(x, 0)).CopyTo(y256, 0);
            VectorExp.Exp(Vector256.Create(x, 8)).CopyTo(y256, 8);
            for (int j = 0; j < 16; j += 4)
            {
                VectorExp.Exp(Vector128.Create(x, j)).CopyTo(y128, j);
            }

            VectorExp.Exp(Vector512.Create(x)).CopyTo(y512, 0);
            for (int j = 0; j < 16; j++)
            {
                float expected = (float)Math.Exp(x[j]);
                float ulp = MathF.BitIncrement(expected) - expected;
                foreach (var actual in new[] { y128[j], y256[j], y512[j] })
                {
                    Assert.True(Math.Abs(actual - expected) <= 2 * ulp, $"exp({x[j]}) = {actual}, expected {expected}");
                }
            }
        }
    }

    [Fact]
    public void ExpHandlesTheEnds()
    {
        var x = Vector256.Create(-100f, -87.4f, 89f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 88.72f, -87.33f);
        var y = VectorExp.Exp(x);
        Assert.Equal(0f, y[0]);
        Assert.Equal(0f, y[1]);
        Assert.Equal(float.PositiveInfinity, y[2]);
        Assert.Equal(float.PositiveInfinity, y[3]);
        Assert.Equal(0f, y[4]);
        Assert.True(float.IsNaN(y[5]));
        Assert.True(float.IsFinite(y[6]) && y[6] > 3e38f);
        Assert.True(y[7] > 0 && float.IsNormal(y[7]));
    }
}
