using Sharplight.Audio;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>The mixed-radix transform of the feature extractors against the direct DFT.</summary>
public sealed class FourierTransformTests
{
    /// <summary>Lengths with factors 2, 3, 5 and a prime agree with the direct sum.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(400)]
    [InlineData(512)]
    [InlineData(97)]
    [InlineData(360)]
    public void MatchesDirectTransform(int length)
    {
        var random = new Random(length);
        var signal = new double[length];
        for (int i = 0; i < length; i++)
        {
            signal[i] = (random.NextDouble() * 2) - 1;
        }

        var real = new double[length];
        var imaginary = new double[length];
        new FourierTransform(length).Forward(signal, real, imaginary, new double[2 * length]);
        for (int k = 0; k < length; k++)
        {
            double expectedReal = 0;
            double expectedImaginary = 0;
            for (int j = 0; j < length; j++)
            {
                double angle = -2 * Math.PI * ((long)j * k % length) / length;
                expectedReal += signal[j] * Math.Cos(angle);
                expectedImaginary += signal[j] * Math.Sin(angle);
            }

            Assert.Equal(expectedReal, real[k], 1e-9);
            Assert.Equal(expectedImaginary, imaginary[k], 1e-9);
        }
    }
}
