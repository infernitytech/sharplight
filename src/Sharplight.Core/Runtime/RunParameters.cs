namespace Sharplight.Runtime;

/// <summary>Per-execution parameters.</summary>
public sealed class RunParameters
{
    /// <summary>
    /// Bindings of symbolic dimensions (e.g. T = number of tokens). Only atomic symbols are bound; expression dimensions
    /// (<c>12*N</c>) are evaluated from them. Data-dependent ops add the values of their symbols during a run.
    /// </summary>
    public Dictionary<string, long> Symbols { get; } = new(StringComparer.Ordinal);
    /// <summary>Runtime scalars (e.g. n_past).</summary>
    public Dictionary<string, long> Scalars { get; } = new(StringComparer.Ordinal);
}
