using Sharplight.Tensors;

namespace Sharplight.Models;

/// <summary>
/// Persistent per-sequence tensor (KV cache, recurrent/conv state). The outermost dimension of a
/// cache may be the symbolic context length <c>ctx</c>, bound when the model is compiled.
/// </summary>
public sealed class StateDefinition
{
    public required string Name { get; init; }
    public required DataType Type { get; init; }
    public required Shape Shape { get; init; }
    public StateKind Kind { get; init; } = StateKind.Other;
    /// <summary>Placement group the state belongs to (same group as the layer using it).</summary>
    public string Group { get; init; } = "";
    public override string ToString() => $"{Name} {Type}{Shape} ({Kind})";
}
