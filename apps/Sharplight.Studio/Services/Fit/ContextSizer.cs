namespace Sharplight.Studio.Services.Fit;

/// <summary>
/// The context lengths a sizing tries (powers of two from <see cref="MinContext"/> up to the model's longest context, and that context
/// itself) and the search for the longest one that fits. A footprint grows with the context, so a binary search over the candidates
/// needs a few estimates only.
/// </summary>
internal static class ContextSizer
{
    /// <summary>The shortest context a sizing offers.</summary>
    public const int MinContext = 2048;

    /// <summary>The longest context tried when the model does not say what it was trained for.</summary>
    public const int DefaultMaxContext = 131072;

    /// <summary>The longest context tried for any model.</summary>
    public const int MaxContext = 1 << 20;

    /// <summary>The longest context to try for a model trained for <paramref name="trainedContext"/> tokens (null: unknown).</summary>
    /// <param name="trainedContext">The model's trained context, when known.</param>
    public static int Limit(int? trainedContext)
        => trainedContext is > 0 ? Math.Clamp(trainedContext.Value, 256, MaxContext) : DefaultMaxContext;

    /// <summary>The candidate contexts up to <paramref name="maxContext"/>, shortest first.</summary>
    /// <param name="maxContext">The longest context (included even when it is not a power of two).</param>
    /// <param name="include">A context to try as well (the configured one), when it is at most <paramref name="maxContext"/>.</param>
    public static IReadOnlyList<int> Candidates(int maxContext, int? include = null)
    {
        var candidates = new List<int>();
        for (int context = MinContext; context < maxContext && context > 0; context *= 2)
        {
            candidates.Add(context);
        }

        candidates.Add(Math.Max(256, maxContext));
        if (include is { } extra && extra >= 256 && extra <= maxContext && !candidates.Contains(extra))
        {
            candidates.Add(extra);
            candidates.Sort();
        }

        return candidates;
    }

    /// <summary>
    /// The longest of <paramref name="candidates"/> (shortest first) that <paramref name="fits"/>, assuming that a context fits when a
    /// longer one does; null when not even the shortest fits.
    /// </summary>
    /// <param name="candidates">Contexts, shortest first.</param>
    /// <param name="fits">Whether a context fits.</param>
    public static async Task<int?> LongestAsync(IReadOnlyList<int> candidates, Func<int, Task<bool>> fits)
    {
        int low = 0;
        int high = candidates.Count - 1;
        int? best = null;
        while (low <= high)
        {
            int middle = low + ((high - low) / 2);
            if (await fits(candidates[middle]).ConfigureAwait(false))
            {
                best = candidates[middle];
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }

        return best;
    }
}
