namespace Sharplight.Onnx;

internal sealed class OnnxValueInfo
{
    public required string Name { get; init; }
    /// <summary>Whether the value is a tensor, sequence, optional or map.</summary>
    public OnnxValueKind Kind { get; init; } = OnnxValueKind.Tensor;
    /// <summary>Element type (<see cref="OnnxType"/>), 0 if the value is not a dense tensor.</summary>
    public int ElemType { get; init; }
    /// <summary>Null if the shape is not stated.</summary>
    public IReadOnlyList<OnnxDim>? Shape { get; init; }
    /// <summary>The element type of a sequence or optional (its <see cref="Name"/> is empty); null otherwise or when not stated.</summary>
    public OnnxValueInfo? Element { get; init; }
    /// <summary>The tensor type inside any nesting of sequences and optionals (this value itself for a tensor).</summary>
    public OnnxValueInfo? Innermost => Kind == OnnxValueKind.Tensor ? this : Element?.Innermost;
    public override string ToString() => Kind switch
    {
        OnnxValueKind.Sequence => $"{Name}: seq({Element})",
        OnnxValueKind.Optional => $"{Name}: optional({Element})",
        _ => $"{Name}: {OnnxType.Name(ElemType)}[{(Shape is null ? "?" : string.Join(", ", Shape))}]",
    };
}
