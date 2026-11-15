namespace Sharplight.Speech;

/// <summary>
/// Beam search over the speech decoder of one window, with the selection rules of Hugging Face transformers (length penalty 1, no early
/// stopping): every step expands each running beam by its log-softmax scores after the logit rules, keeps the best 2 × beams continuations,
/// moves those ending in the end token (or at the length limit) among the best beams-many into the finished set, and continues with the best
/// beams-many others; it stops when no running beam can beat the worst of a full finished set.
/// </summary>
/// <remarks>
/// The decoder has a single KV cache. Each running beam is evaluated by rewinding the cache to the prefix it shares with the sequence
/// already in the cache and feeding the rest; beams are visited in token order, so neighbors share long prefixes and a step mostly costs one
/// decoder pass of one token per beam.
/// </remarks>
/// <param name="network">The decoder, positioned after the window's prompt.</param>
/// <param name="filter">The logit rules.</param>
/// <param name="endId">The end-of-text token.</param>
/// <param name="beams">The number of beams (at least 2).</param>
internal sealed class SpeechBeamSearch(SpeechEncoderDecoder network, SpeechLogitFilter filter, int endId, int beams)
{
    /// <summary>Exponent of the generated length that normalizes the scores of finished beams.</summary>
    private const double LengthPenalty = 1.0;

    private readonly int _promptPosition = network.Position;
    private int[] _cached = [];

    /// <summary>Searches from the logits after the prompt and returns the best finished sequence.</summary>
    /// <param name="first">The logits of the first generated token.</param>
    /// <param name="limit">The most tokens to generate (the end token included).</param>
    /// <param name="cancellationToken">Cancels the search between steps.</param>
    public BeamHypothesis Run(ReadOnlySpan<float> first, int limit, CancellationToken cancellationToken)
    {
        var running = new List<BeamHypothesis> { new([], 0, 0) };
        var finished = new List<BeamHypothesis>();
        var scores = new float[running.Count][];
        scores[0] = first.ToArray();
        for (int step = 0; step < limit; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (step > 0)
            {
                scores = Evaluate(running);
            }

            for (int b = 0; b < running.Count; b++)
            {
                LogProbabilities.Normalize(scores[b]);
                filter.Apply(scores[b], running[b].Tokens);
            }

            var candidates = TopCandidates(running, scores, 2 * beams);
            bool atLimit = step + 1 >= limit;
            var next = new List<BeamHypothesis>(beams);
            for (int rank = 0; rank < candidates.Count; rank++)
            {
                var candidate = candidates[rank];
                var parent = running[candidate.Beam];
                double logProbability = parent.LogProbability + scores[candidate.Beam][candidate.Token];
                bool stops = atLimit || candidate.Token == endId;
                if (stops)
                {
                    // Only the best beams-many continuations may finish; the others are spares that keep beams-many beams running.
                    if (rank < beams)
                    {
                        int[] tokens = candidate.Token == endId ? parent.Tokens : [.. parent.Tokens, candidate.Token];
                        double length = Math.Pow(parent.Tokens.Length + 1, LengthPenalty);
                        AddFinished(finished, new BeamHypothesis(tokens, candidate.Score / length, logProbability));
                    }
                }
                else if (next.Count < beams)
                {
                    next.Add(new BeamHypothesis([.. parent.Tokens, candidate.Token], candidate.Score, logProbability));
                }
            }

            if (next.Count == 0 || atLimit || !ImprovementPossible(next, finished, step + 1))
            {
                break;
            }

            running = next;
        }

        return finished.Count > 0 ? finished[0] : running[0];
    }

    /// <summary>The log-softmax inputs (raw logits) of every running beam.</summary>
    private float[][] Evaluate(List<BeamHypothesis> running)
    {
        var scores = new float[running.Count][];
        var order = Enumerable.Range(0, running.Count).OrderBy(b => running[b].Tokens, TokenOrder.Instance).ToArray();
        foreach (int b in order)
        {
            int[] tokens = running[b].Tokens;
            int shared = SharedPrefix(_cached, tokens);
            if (shared == tokens.Length)
            {
                // The logits after a full prefix are not kept, so the last token is fed again.
                shared--;
            }

            network.Rewind(_promptPosition + shared);
            scores[b] = network.Decode(tokens.AsSpan(shared)).ToArray();
            _cached = tokens;
        }

        return scores;
    }

    /// <summary>The best <paramref name="count"/> continuations of all running beams, best first.</summary>
    private static List<BeamCandidate> TopCandidates(List<BeamHypothesis> running, float[][] scores, int count)
    {
        var all = new List<BeamCandidate>(running.Count * count);
        for (int b = 0; b < running.Count; b++)
        {
            var best = new PriorityQueue<int, float>(count + 1);
            var row = scores[b];
            for (int token = 0; token < row.Length; token++)
            {
                if (float.IsNegativeInfinity(row[token]))
                {
                    continue;
                }

                best.Enqueue(token, row[token]);
                if (best.Count > count)
                {
                    best.Dequeue();
                }
            }

            while (best.TryDequeue(out int token, out float score))
            {
                all.Add(new BeamCandidate(b, token, running[b].Score + score));
            }
        }

        all.Sort((x, y) => y.Score.CompareTo(x.Score));
        return all.Count > count ? all.GetRange(0, count) : all;
    }

    /// <summary>Adds a finished beam, keeping the best beams-many, best first.</summary>
    private void AddFinished(List<BeamHypothesis> finished, BeamHypothesis hypothesis)
    {
        finished.Add(hypothesis);
        finished.Sort((x, y) => y.Score.CompareTo(x.Score));
        if (finished.Count > beams)
        {
            finished.RemoveAt(finished.Count - 1);
        }
    }

    /// <summary>
    /// False once the finished set is full and the best running beam, normalized by the current length, no longer beats its worst member
    /// (transformers' heuristic without early stopping).
    /// </summary>
    private bool ImprovementPossible(List<BeamHypothesis> running, List<BeamHypothesis> finished, int generated)
    {
        if (finished.Count < beams)
        {
            return true;
        }

        double best = running[0].Score / Math.Pow(generated, LengthPenalty);
        return best > finished[^1].Score;
    }

    private static int SharedPrefix(int[] a, int[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        int i = 0;
        while (i < n && a[i] == b[i])
        {
            i++;
        }

        return i;
    }
}
