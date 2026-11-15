namespace Sharplight.Speech;

/// <summary>Lexicographic order of token sequences (sequences with a common prefix sort next to each other).</summary>
internal sealed class TokenOrder : IComparer<int[]>
{
    /// <summary>The instance.</summary>
    public static TokenOrder Instance { get; } = new();

    /// <inheritdoc/>
    public int Compare(int[]? x, int[]? y)
    {
        if (x is null || y is null)
        {
            return x is null ? (y is null ? 0 : -1) : 1;
        }

        return x.AsSpan().SequenceCompareTo(y);
    }
}
