namespace Sharplight.Onnx;

internal sealed class OnnxValueInfo
{
    public required string Name { get; init; }
    /// <summary>Element type (<see cref="OnnxType"/>), 0 if the value is not a dense tensor.</summary>
    public int ElemType { get; init; }
    /// <summary>Null if the shape is not stated.</summary>
    public IReadOnlyList<OnnxDim>? Shape { get; init; }
    public override string ToString() => $"{Name}: {OnnxType.Name(ElemType)}[{(Shape is null ? "?" : string.Join(", ", Shape))}]";
}
