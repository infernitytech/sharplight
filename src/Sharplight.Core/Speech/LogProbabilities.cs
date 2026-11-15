namespace Sharplight.Speech;

/// <summary>Log-softmax helpers of the speech decoding loops.</summary>
internal static class LogProbabilities
{
    /// <summary>The log probability of <paramref name="index"/> under the softmax of <paramref name="logits"/>.</summary>
    /// <param name="logits">The logits.</param>
    /// <param name="index">The token.</param>
    public static double Of(ReadOnlySpan<float> logits, int index) => logits[index] - Normalizer(logits);

    /// <summary>Replaces <paramref name="logits"/> by their log-softmax.</summary>
    /// <param name="logits">The logits; on return, log probabilities.</param>
    public static void Normalize(Span<float> logits)
    {
        float normalizer = (float)Normalizer(logits);
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] -= normalizer;
        }
    }

    /// <summary>log Σ exp(logits), computed around the largest logit.</summary>
    private static double Normalizer(ReadOnlySpan<float> logits)
    {
        float max = float.NegativeInfinity;
        foreach (float value in logits)
        {
            max = MathF.Max(max, value);
        }

        double sum = 0;
        foreach (float value in logits)
        {
            sum += Math.Exp(value - max);
        }

        return max + Math.Log(sum);
    }
}
