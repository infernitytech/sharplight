using Sharplight.Tensors;

namespace Sharplight.Models;

/// <summary>
/// Persistent per-sequence tensor (KV cache, recurrent/conv state). The outermost dimension of a
/// cache may be the symbolic context length <c>ctx</c>, bound when the model is compiled.
/// </summary>
public sealed class StateDefinition
{
    /// <summary>State name, unique within the model.</summary>
    public required string Name { get; init; }
    /// <summary>Element storage type.</summary>
    public required DataType Type { get; init; }
    /// <summary>Shape per sequence; the outermost dimension of a KV cache is the symbolic context length.</summary>
    public required Shape Shape { get; init; }
    /// <summary>Kind of state, used by runtimes to manage it (context shifting, snapshots).</summary>
    public StateKind Kind { get; init; } = StateKind.Other;
    /// <summary>
    /// Compressed KV cache: positions per cache row (1 = one row per position). A compressed cache holds one entry per window of this many
    /// positions (entry e summarizes positions [e · r, (e + 1) · r) and is written by the window's last position, see <c>CacheWrite</c> with
    /// attribute <c>compression</c>); its outermost dimension is <c>ceil(ctx / r)</c>. Multi-sequence graphs size and address it per slot
    /// (<see cref="Graph.MultiSequence"/>).
    /// </summary>
    public int Compression { get; init; } = 1;
    /// <summary>Placement group the state belongs to (same group as the layer using it).</summary>
    public string Group { get; init; } = "";
    /// <inheritdoc/>
    public override string ToString() => $"{Name} {Type}{Shape} ({Kind})";
}
