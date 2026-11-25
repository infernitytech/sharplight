namespace Sharplight.Architectures;

/// <summary>
/// CosyVoice's repetition-aware sampling (<c>ras_sampling</c>): nucleus sampling over the most probable tokens (added while their cumulative
/// probability is below <c>topP</c> and fewer than <c>topK</c> are kept); when the drawn token already occurs at least <c>window · tauR</c> times among
/// the last <c>window</c> tokens, it is excluded and the token is drawn from the whole distribution instead.
/// </summary>
/// <param name="topP">The nucleus probability (0.8).</param>
/// <param name="topK">The most tokens of the nucleus (25).</param>
/// <param name="window">The repetition window (10).</param>
/// <param name="tauR">The repetition ratio (0.1).</param>
/// <param name="temperature">Divides the logits (1 in the reference).</param>
/// <param name="seed">The seed of the generator, or <see langword="null"/>.</param>
internal sealed class CosyVoiceRasSampler(float topP, int topK, int window, float tauR, float temperature, long? seed)
{
    private readonly Random _random = Tasks.Seeds.CreateRandom(seed);

    /// <summary>Draws the next token.</summary>
    /// <param name="logits">The logits (modified: a repeated token is set to −∞).</param>
    /// <param name="decoded">The tokens generated so far.</param>
    public int Sample(float[] logits, IReadOnlyList<int> decoded)
    {
        var probabilities = Softmax(logits, temperature);
        var order = Enumerable.Range(0, probabilities.Length).OrderByDescending(i => probabilities[i]).ToArray();
        var kept = new List<int>();
        double cumulative = 0;
        foreach (int index in order)
        {
            if (cumulative >= topP || kept.Count >= topK)
            {
                break;
            }

            cumulative += probabilities[index];
            kept.Add(index);
        }

        int token = Draw(kept, probabilities);
        int repeats = 0;
        for (int i = Math.Max(0, decoded.Count - window); i < decoded.Count; i++)
        {
            repeats += decoded[i] == token ? 1 : 0;
        }

        if (repeats >= window * tauR)
        {
            logits[token] = float.NegativeInfinity;
            probabilities = Softmax(logits, temperature);
            token = Draw(Enumerable.Range(0, probabilities.Length).ToList(), probabilities);
        }

        return token;
    }

    private int Draw(List<int> candidates, double[] probabilities)
    {
        double total = 0;
        foreach (int c in candidates)
        {
            total += probabilities[c];
        }

        double target = _random.NextDouble() * total;
        foreach (int c in candidates)
        {
            target -= probabilities[c];
            if (target < 0)
            {
                return c;
            }
        }

        return candidates[^1];
    }

    private static double[] Softmax(float[] logits, float temperature)
    {
        double scale = temperature > 0 ? 1.0 / temperature : 1.0;
        double max = double.NegativeInfinity;
        foreach (float v in logits)
        {
            max = Math.Max(max, v * scale);
        }

        var result = new double[logits.Length];
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = float.IsNegativeInfinity(logits[i]) ? 0 : Math.Exp((logits[i] * scale) - max);
            sum += result[i];
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }
}
