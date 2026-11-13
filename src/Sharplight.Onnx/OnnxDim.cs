namespace Sharplight.Onnx;

/// <summary>A dimension of an ONNX value's shape: fixed value or symbolic parameter (both absent = unknown).</summary>
internal readonly record struct OnnxDim(long? Value, string? Param)
{
    public bool IsKnown => Value is > 0 && Param is null;
    public override string ToString() => Param ?? Value?.ToString() ?? "?";
}
