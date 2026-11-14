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
    /// <summary>
    /// Names of graph outputs the caller does not read after this run (for example the full logits when tokens are sampled on the
    /// device). Providers may skip downloading them to host memory; reading such an output afterwards returns unspecified data.
    /// </summary>
    public HashSet<string> SkipOutputs { get; } = new(StringComparer.Ordinal);
    /// <summary>
    /// The strings that <see cref="Tensors.DataType.Str"/> handles of this run refer to. <see cref="CompiledModel.Run"/> clears it and adds
    /// the string inputs; string kernels add their results. Handles stay valid until the next run with these parameters.
    /// </summary>
    public StringHeap Strings { get; } = new();
}
