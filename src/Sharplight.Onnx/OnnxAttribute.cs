namespace Sharplight.Onnx;

internal sealed class OnnxAttribute
{
    public required string Name { get; init; }
    public OnnxAttrType Type { get; init; }
    public float F { get; init; }
    public long I { get; init; }
    public string? S { get; init; }
    public OnnxTensor? T { get; init; }
    public OnnxGraph? G { get; init; }
    public IReadOnlyList<float> Floats { get; init; } = [];
    public IReadOnlyList<long> Ints { get; init; } = [];
    public IReadOnlyList<string> Strings { get; init; } = [];
    public override string ToString() => $"{Name}:{Type}";
}
