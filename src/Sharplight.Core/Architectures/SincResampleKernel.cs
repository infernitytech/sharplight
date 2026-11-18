using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The band-limited resampling kernel of torchaudio's <c>resample</c> (<c>sinc_interp_hann</c>, 6 zero crossings, roll-off 0.99) as a convolution weight: a strided convolution with
/// <see cref="NewFrequency"/> output channels whose interleaved output is the resampled signal. Single precision throughout, like torchaudio on f32 audio.
/// </summary>
internal sealed class SincResampleKernel
{
    private const int LowpassFilterWidth = 6;
    private const float Rolloff = 0.99f;

    private SincResampleKernel(int original, int target, int width, float[] values)
    {
        OriginalFrequency = original;
        NewFrequency = target;
        Width = width;
        Values = values;
    }

    /// <summary>The input stride of the convolution (the original frequency divided by the common divisor).</summary>
    public int OriginalFrequency { get; }

    /// <summary>The number of output channels (the new frequency divided by the common divisor).</summary>
    public int NewFrequency { get; }

    /// <summary>Zero padding on each side of the signal.</summary>
    public int Width { get; }

    /// <summary>The kernel taps, laid out [<see cref="NewFrequency"/>, 1, 2 · <see cref="Width"/> + <see cref="OriginalFrequency"/>].</summary>
    public float[] Values { get; }

    /// <summary>The taps per output channel.</summary>
    public int Taps => (2 * Width) + OriginalFrequency;

    /// <summary>Computes the kernel for resampling from <paramref name="original"/> to <paramref name="target"/> Hz.</summary>
    /// <param name="original">The original sample rate.</param>
    /// <param name="target">The new sample rate.</param>
    public static SincResampleKernel Create(int original, int target)
    {
        int gcd = (int)System.Numerics.BigInteger.GreatestCommonDivisor(original, target);
        int orig = original / gcd;
        int next = target / gcd;
        float baseFrequency = MathF.Min(orig, next) * Rolloff;
        int width = (int)Math.Ceiling(LowpassFilterWidth * (double)orig / baseFrequency);
        int taps = (2 * width) + orig;
        var values = new float[next * taps];
        for (int o = 0; o < next; o++)
        {
            for (int i = 0; i < taps; i++)
            {
                float index = (-width + i) / (float)orig;
                float t = (-o / (float)next) + index;
                t *= baseFrequency;
                t = Math.Clamp(t, -LowpassFilterWidth, LowpassFilterWidth);
                float window = MathF.Cos(t * MathF.PI / LowpassFilterWidth / 2f);
                window *= window;
                t *= MathF.PI;
                float scale = baseFrequency / orig;
                float sinc = t == 0f ? 1f : MathF.Sin(t) / t;
                values[(o * taps) + i] = sinc * window * scale;
            }
        }

        return new SincResampleKernel(orig, next, width, values);
    }

    /// <summary>The kernel as a weight named <paramref name="name"/>.</summary>
    /// <param name="name">The weight's name.</param>
    public WeightInfo ToWeight(string name)
    {
        var bytes = new byte[Values.Length * sizeof(float)];
        Buffer.BlockCopy(Values, 0, bytes, 0, bytes.Length);
        return WeightInfo.Derived(name, DataType.F32, new Shape(NewFrequency, 1, Taps), () => bytes);
    }
}
