using System.Runtime.InteropServices;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// A learned relative position bias expanded per key − query distance, the form the attention ops take as their
/// <c>position_bias</c> / <c>position_bias_range</c> attributes: <see cref="Table"/>[h · (2R + 1) + d + R] is added to every score of
/// head h whose key lies d positions after the query (d clamped to ±<see cref="Range"/>).
/// </summary>
/// <param name="Table">Per head the bias of the distances −R..R, [heads · (2R + 1)].</param>
/// <param name="Range">R: distances beyond it share the bias of ±R.</param>
internal sealed record RelativePositionBias(float[] Table, int Range)
{
    /// <summary>The attribute pairs of an attention node.</summary>
    public (string, object)[] Attributes => [("position_bias", Table), ("position_bias_range", Range)];

    /// <summary>
    /// Expands T5's bucketed bias (<c>relative_attention_bias</c>, [buckets, heads] f32) per distance. T5 buckets distances exactly up
    /// to half the buckets (a quarter when bidirectional: the other half then holds the keys after the query) and logarithmically up to
    /// <paramref name="maxDistance"/>, beyond which every distance shares the last bucket; a causal decoder puts every key after the
    /// query into bucket 0. The bucket arithmetic follows Hugging Face's <c>_relative_position_bucket</c> in single precision.
    /// </summary>
    /// <param name="weight">The bucket table [buckets, heads], f32.</param>
    /// <param name="maxDistance">Distance from which on the last bucket applies (T5: 128).</param>
    /// <param name="bidirectional">Encoder (true) or causal decoder (false) bucketing.</param>
    /// <exception cref="InvalidDataException">The table is not a 2-D f32 tensor.</exception>
    public static RelativePositionBias FromBuckets(WeightInfo weight, int maxDistance, bool bidirectional)
    {
        if (weight.Type != DataType.F32 || weight.Shape.Rank != 2)
        {
            throw new InvalidDataException($"{weight.Name}: expected an f32 [buckets, heads] table, got {weight.Type}{weight.Shape}.");
        }

        int buckets = (int)weight.Shape[0].Fixed;
        int heads = (int)weight.Shape[1].Fixed;
        var values = MemoryMarshal.Cast<byte, float>(weight.ReadAll());
        int width = 2 * maxDistance + 1;
        var table = new float[heads * width];
        for (int d = -maxDistance; d <= maxDistance; d++)
        {
            int bucket = Bucket(d, buckets, maxDistance, bidirectional);
            for (int h = 0; h < heads; h++)
            {
                table[h * width + d + maxDistance] = values[bucket * heads + h];
            }
        }

        return new RelativePositionBias(table, maxDistance);
    }

    /// <summary>T5's bucket of the distance <paramref name="relative"/> (key position − query position).</summary>
    public static int Bucket(int relative, int buckets, int maxDistance, bool bidirectional)
    {
        int bucket = 0;
        int n = buckets;
        if (bidirectional)
        {
            n /= 2;
            if (relative > 0)
            {
                bucket += n;
            }

            relative = Math.Abs(relative);
        }
        else
        {
            relative = -Math.Min(relative, 0);
        }

        int maxExact = n / 2;
        if (relative < maxExact)
        {
            return bucket + relative;
        }

        float large = MathF.Log(relative / (float)maxExact) / (float)Math.Log(maxDistance / (double)maxExact) * (n - maxExact);
        return bucket + Math.Min(maxExact + (int)large, n - 1);
    }
}
