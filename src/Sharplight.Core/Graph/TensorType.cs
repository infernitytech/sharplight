using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>Output type of an op as computed by shape inference.</summary>
public readonly record struct TensorType(DataType Type, Shape Shape);
