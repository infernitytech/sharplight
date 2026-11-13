namespace Sharplight.Llm.Serving.Speculative;

/// <summary>
/// Prompt-lookup (n-gram) speculation without a draft model: finds the most recent earlier occurrence of the context's last
/// <c>n</c> tokens (longest <c>n</c> first) and proposes the tokens that followed it. Very effective when the output copies from
/// the prompt (code editing, retrieval answers, summaries with quotes, structured output); free when it finds nothing.
/// Stateless and cheap (a backward scan of the context).
/// </summary>
/// <param name="maxDraftTokens">Most tokens proposed per step.</param>
/// <param name="maxNgram">Longest suffix that is matched.</param>
/// <param name="minNgram">Shortest suffix that is matched (1 proposes after any repeated token; 2 is more precise).</param>
/// <param name="maxLookback">Tokens of context searched, counting back from the end (bounds the cost on long contexts).</param>
public sealed class NgramSpeculator(int maxDraftTokens = 4, int maxNgram = 4, int minNgram = 2, int maxLookback = 4096) : ISpeculator
{
    /// <inheritdoc/>
    public int MaxDraftTokens { get; } = Math.Max(1, maxDraftTokens);

    /// <inheritdoc/>
    public void Propose(IReadOnlyList<SpeculationRequest> requests)
    {
        foreach (var r in requests)
        {
            r.Count = Lookup(r.Context, r.Draft.AsSpan(0, Math.Min(r.MaxTokens, MaxDraftTokens)));
        }
    }

    /// <summary>Writes the continuation of the most recent earlier match of the context suffix; returns its length.</summary>
    internal int Lookup(IReadOnlyList<int> context, Span<int> draft)
    {
        int len = context.Count;
        if (draft.Length == 0)
        {
            return 0;
        }

        int start = Math.Max(0, len - maxLookback);
        for (int n = Math.Min(maxNgram, len - 1); n >= Math.Max(1, minNgram); n--)
        {
            // Candidate match ends at e (exclusive), the suffix occupies [len - n, len); e < len so the match is earlier.
            for (int e = len - 1; e - n >= start; e--)
            {
                if (!SuffixMatches(context, e, n))
                {
                    continue;
                }

                int count = Math.Min(draft.Length, len - e);
                for (int i = 0; i < count; i++)
                {
                    draft[i] = context[e + i];
                }

                return count;
            }
        }

        return 0;
    }

    private static bool SuffixMatches(IReadOnlyList<int> context, int end, int n)
    {
        int len = context.Count;
        for (int i = 1; i <= n; i++)
        {
            if (context[end - i] != context[len - i])
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc/>
    public void Release(int slot)
    {
    }

    /// <summary>Nothing to release.</summary>
    public void Dispose()
    {
    }
}
