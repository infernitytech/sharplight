using Sharplight.Models;

namespace Sharplight.Import.Onnx;

/// <summary>Tensor payload: dims, element type and a byte source holding the little-endian dense data.</summary>
internal sealed class OnnxTensor
{
    public string Name { get; init; } = "";
    public int ElemType { get; init; }
    public long[] Dims { get; init; } = [];
    /// <summary>Dense little-endian element data (row-major); null for string tensors.</summary>
    public IWeightSource? Source { get; init; }
    /// <summary>Elements of a string tensor (UTF-8 decoded, row-major); null for other types.</summary>
    public string[]? Strings { get; init; }
    public long Elements => Dims.Aggregate(1L, (product, dim) => product * dim);
    public override string ToString() => $"{Name} {OnnxType.Name(ElemType)}[{string.Join(", ", Dims)}]";
}
