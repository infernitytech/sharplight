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
    /// <summary>Placement group the state belongs to (same group as the layer using it).</summary>
    public string Group { get; init; } = "";
    /// <inheritdoc/>
    public override string ToString() => $"{Name} {Type}{Shape} ({Kind})";
}
