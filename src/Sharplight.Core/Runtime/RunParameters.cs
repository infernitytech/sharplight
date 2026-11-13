namespace Sharplight.Runtime;

/// <summary>Per-execution parameters.</summary>
public sealed class RunParameters
{
    /// <summary>Bindings of symbolic dimensions (e.g. T = number of tokens).</summary>
    public Dictionary<string, long> Symbols { get; } = new(StringComparer.Ordinal);
    /// <summary>Runtime scalars (e.g. n_past).</summary>
    public Dictionary<string, long> Scalars { get; } = new(StringComparer.Ordinal);
}
