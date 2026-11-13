using Sharplight.Models;

namespace Sharplight.Onnx;

/// <summary>Tensor payload: dims, element type and a byte source holding the little-endian dense data.</summary>
internal sealed class OnnxTensor
{
    public string Name { get; init; } = "";
    public int ElemType { get; init; }
    public long[] Dims { get; init; } = [];
    /// <summary>Dense little-endian element data (row-major); null for string tensors.</summary>
    public IWeightSource? Source { get; init; }
    public long Elements { get { long n = 1; foreach (var d in Dims) { n *= d; } return n; } }
    public override string ToString() => $"{Name} {OnnxType.Name(ElemType)}[{string.Join(", ", Dims)}]";
}
