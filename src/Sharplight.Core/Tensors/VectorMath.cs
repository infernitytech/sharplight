namespace Sharplight.Tensors;

/// <summary>Small host-side vector helpers shared by the task clients (embedding normalization, classification scores).</summary>
internal static class VectorMath
{
    /// <summary>Scales <paramref name="vector"/> to unit Euclidean length in place; a zero vector stays zero.</summary>
    /// <param name="vector">The vector.</param>
    public static void L2Normalize(Span<float> vector)
    {
        double sum = 0;
        foreach (float v in vector)
        {
            sum += (double)v * v;
        }

        float scale = sum > 0 ? (float)(1.0 / Math.Sqrt(sum)) : 0f;
        for (int i = 0; i < vector.Length; i++)
        {
            vector[i] *= scale;
        }
    }

    /// <summary>The softmax of <paramref name="logits"/> (shifted by the maximum, summed in double precision).</summary>
    /// <param name="logits">The logits.</param>
    /// <returns>The probabilities, one per logit.</returns>
    public static float[] Softmax(ReadOnlySpan<float> logits)
    {
        var scores = new float[logits.Length];
        float max = float.NegativeInfinity;
        foreach (float l in logits)
        {
            max = MathF.Max(max, l);
        }

        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            scores[i] = MathF.Exp(logits[i] - max);
            sum += scores[i];
        }

        for (int i = 0; i < logits.Length; i++)
        {
            scores[i] = (float)(scores[i] / sum);
        }

        return scores;
    }
}
