namespace Sharplight.Speech;

/// <summary>
/// Connectionist temporal classification decoding on the host: the per-row token logits of a CTC head become tokens by choosing the best
/// token per row, merging repeats and dropping blanks (greedy, as NeMo and transformers decode), or by a prefix beam search over the
/// collapsed sequences.
/// </summary>
internal static class CtcDecoding
{
    /// <summary>The greedy path: the best token of every row, repeats merged, blanks dropped.</summary>
    /// <param name="logits">Row-major logits [rows, vocabulary].</param>
    /// <param name="vocabulary">Logits per row.</param>
    /// <param name="blank">The blank token.</param>
    public static List<AlignedToken> Greedy(ReadOnlySpan<float> logits, int vocabulary, int blank)
    {
        int rows = logits.Length / vocabulary;
        var tokens = new List<AlignedToken>();
        int previous = -1;
        for (int row = 0; row < rows; row++)
        {
            var scores = logits.Slice(row * vocabulary, vocabulary);
            int best = 0;
            for (int v = 1; v < vocabulary; v++)
            {
                if (scores[v] > scores[best])
                {
                    best = v;
                }
            }

            if (best != blank && best == previous && tokens.Count > 0)
            {
                // A repeat of the running token extends it.
                var last = tokens[^1];
                tokens[^1] = last with { Rows = last.Rows + 1 };
            }
            else if (best != blank)
            {
                tokens.Add(new AlignedToken(best, row, 1, (float)LogProbabilities.Of(scores, best)));
            }

            previous = best;
        }

        return tokens;
    }

    /// <summary>
    /// A prefix beam search over the collapsed token sequences (each prefix keeps the probability of the paths ending in a blank and in its last
    /// token), without a language model. The tokens' rows are those where they first entered the best prefix.
    /// </summary>
    /// <param name="logits">Row-major logits [rows, vocabulary].</param>
    /// <param name="vocabulary">Logits per row.</param>
    /// <param name="blank">The blank token.</param>
    /// <param name="beams">The prefixes kept per row.</param>
    public static List<AlignedToken> BeamSearch(ReadOnlySpan<float> logits, int vocabulary, int blank, int beams)
    {
        int rows = logits.Length / vocabulary;
        var root = new Prefix(null, -1, -1, 0f);
        var current = new Dictionary<Prefix, (double Blank, double Token)> { [root] = (0, double.NegativeInfinity) };
        var children = new Dictionary<(Prefix, int), Prefix>();
        var logProbabilities = new float[vocabulary];
        for (int row = 0; row < rows; row++)
        {
            var scores = logits.Slice(row * vocabulary, vocabulary);
            scores.CopyTo(logProbabilities);
            LogProbabilities.Normalize(logProbabilities);
            // Only the likeliest tokens of the row can extend a prefix into the beam.
            int[] candidates = [.. Enumerable.Range(0, vocabulary).Where(v => v != blank).OrderByDescending(v => logProbabilities[v]).Take(beams)];
            var next = new Dictionary<Prefix, (double Blank, double Token)>();
            void Add(Prefix prefix, double blankScore, double tokenScore)
            {
                var (b, t) = next.TryGetValue(prefix, out var existing) ? existing : (double.NegativeInfinity, double.NegativeInfinity);
                next[prefix] = (LogAdd(b, blankScore), LogAdd(t, tokenScore));
            }

            foreach (var (prefix, (pb, pt)) in current)
            {
                double total = LogAdd(pb, pt);
                Add(prefix, total + logProbabilities[blank], double.NegativeInfinity);
                if (prefix.Token >= 0)
                {
                    // The last token repeated without a blank in between stays one token.
                    Add(prefix, double.NegativeInfinity, pt + logProbabilities[prefix.Token]);
                }

                foreach (int v in candidates)
                {
                    if (!children.TryGetValue((prefix, v), out var child))
                    {
                        child = new Prefix(prefix, v, row, logProbabilities[v]);
                        children[(prefix, v)] = child;
                    }

                    // A repeat of the last token only starts a new token after a blank.
                    double score = v == prefix.Token ? pb + logProbabilities[v] : total + logProbabilities[v];
                    Add(child, double.NegativeInfinity, score);
                }
            }

            current = next.OrderByDescending(e => LogAdd(e.Value.Blank, e.Value.Token)).Take(beams).ToDictionary(e => e.Key, e => e.Value);
        }

        var best = current.MaxBy(e => LogAdd(e.Value.Blank, e.Value.Token)).Key;
        var tokens = new List<AlignedToken>();
        for (var p = best; p.Parent is not null; p = p.Parent)
        {
            tokens.Add(new AlignedToken(p.Token, p.Row, 1, p.LogProbability));
        }

        tokens.Reverse();
        return tokens;
    }

    private static double LogAdd(double a, double b)
    {
        if (double.IsNegativeInfinity(a))
        {
            return b;
        }

        if (double.IsNegativeInfinity(b))
        {
            return a;
        }

        double max = Math.Max(a, b);
        return max + Math.Log(Math.Exp(a - max) + Math.Exp(b - max));
    }

    /// <summary>A collapsed token sequence as a node of a prefix tree (reference identity).</summary>
    private sealed class Prefix(Prefix? parent, int token, int row, float logProbability)
    {
        public Prefix? Parent { get; } = parent;

        public int Token { get; } = token;

        public int Row { get; } = row;

        public float LogProbability { get; } = logProbability;
    }
}
