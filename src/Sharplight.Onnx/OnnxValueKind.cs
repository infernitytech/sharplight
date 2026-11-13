namespace Sharplight.Onnx;

/// <summary>The kind of an ONNX value type (TypeProto).</summary>
internal enum OnnxValueKind
{
    /// <summary>No type stated.</summary>
    Unknown,

    /// <summary>A dense tensor.</summary>
    Tensor,

    /// <summary>A sequence of values of one element type.</summary>
    Sequence,

    /// <summary>An optional value: an element or nothing.</summary>
    Optional,

    /// <summary>A map (not supported by the translator).</summary>
    Map,
}
