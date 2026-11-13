using Sharplight.Graph;

namespace Sharplight.Onnx;

/// <summary>An ONNX value during translation: either known at import time (<see cref="C"/>) or a dynamic IR value (<see cref="V"/>).</summary>
internal sealed class TranslatedTensor
{
    /// <summary>The IR value of a dynamic tensor; null for constants.</summary>
    public Value? V { get; init; }

    /// <summary>The compile-time constant; null for dynamic tensors.</summary>
    public OnnxConst? C { get; init; }

    /// <summary>
    /// ONNX element type of a dynamic tensor (<see cref="OnnxType"/>), 0 until the translator assigned one; constants carry theirs in
    /// <see cref="C"/>. The IR value may use a wider transport type (see OnnxTranslator.ElemTypeOf).
    /// </summary>
    public int ElemType { get; set; }

    /// <summary>The f32 view of a dynamic non-f32 tensor or of a constant, created on first use.</summary>
    public Value? CachedValue { get; set; }

    /// <summary>The i32 view of a dynamic non-i32 tensor (index or integer operand), created on first use.</summary>
    public Value? CachedIndex { get; set; }

    /// <summary>Logically rank-0 (an ONNX scalar graph input); physically a one-element vector.</summary>
    public bool Scalar { get; set; }

    /// <summary>
    /// Unit dims in front of a symbolic dim (ONNX shape <c>[1, ..., 1, N, ...]</c>): the IR value <see cref="V"/> has the shape without them,
    /// because only the outermost IR dim may be symbolic. Only ops that know about them accept such a tensor.
    /// </summary>
    public int LeadingUnits { get; set; }

    /// <summary>
    /// The IR value holds the transpose of the ONNX tensor: NonZero yields ONNX <c>[rank, count]</c>, stored as <c>[count, rank]</c> so that
    /// the data-dependent count is the outer dim. Only Transpose, Gather of a row, Squeeze, Shape, Cast and Identity accept it.
    /// </summary>
    public bool TransposedNonZero { get; set; }

    /// <summary>True when the value is known at import time.</summary>
    public bool IsConst => C is not null;

    /// <summary>Wraps a runtime value.</summary>
    public static TranslatedTensor Dyn(Value v) => new() { V = v };

    /// <summary>Wraps a compile-time constant.</summary>
    public static TranslatedTensor Const(OnnxConst c) => new() { C = c };
}
