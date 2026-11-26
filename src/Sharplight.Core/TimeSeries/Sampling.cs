namespace Sharplight.TimeSeries;

/// <summary>Token sampling as transformers' <c>generate(do_sample=True)</c> applies it: temperature, then top-k (ties at the cutoff kept), then a draw from the softmax.</summary>
internal sealed class Sampling(float temperature, int topK, Random random)
{
    private float[] _sorted = [];

    /// <summary>Draws a token from <paramref name="logits"/> (masked tokens hold negative infinity).</summary>
    public int Next(ReadOnlySpan<float> logits)
    {
        float cutoff = float.NegativeInfinity;
        if (topK > 0 && topK < logits.Length)
        {
            if (_sorted.Length != logits.Length)
            {
                _sorted = new float[logits.Length];
            }

            logits.CopyTo(_sorted);
            Array.Sort(_sorted);
            cutoff = _sorted[^topK];
        }

        float max = float.NegativeInfinity;
        int best = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
                best = i;
            }
        }

        if (topK == 1)
        {
            return best;
        }

        double total = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] >= cutoff && !float.IsNegativeInfinity(logits[i]))
            {
                total += Math.Exp((logits[i] - max) / temperature);
            }
        }

        double draw = random.NextDouble() * total;
        double running = 0;
        int last = best;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] >= cutoff && !float.IsNegativeInfinity(logits[i]))
            {
                running += Math.Exp((logits[i] - max) / temperature);
                last = i;
                if (running >= draw)
                {
                    return i;
                }
            }
        }

        return last;
    }
}
