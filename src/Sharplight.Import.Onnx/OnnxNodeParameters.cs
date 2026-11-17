using Sharplight.Models;

namespace Sharplight.Import.Onnx;

/// <summary>
/// One node of an ONNX graph with the constant tensors among its inputs (initializers and the outputs of <c>Constant</c> nodes), so that an
/// importer can find the parameters of an export whose initializers carry generated names (<c>onnx::MatMul_1234</c>) by the node that reads them
/// (<c>/blocks.0/attn/query/MatMul</c>). See <see cref="OnnxImporter.NodeParameters"/>.
/// </summary>
public sealed class OnnxNodeParameters
{
    /// <summary>The node name (often the module path of the exported model).</summary>
    public required string Name { get; init; }

    /// <summary>The operator type (<c>MatMul</c>, <c>Conv</c>, ...).</summary>
    public required string OpType { get; init; }

    /// <summary>The node's inputs in order: the constant tensor of an input that has one, otherwise <see langword="null"/>.</summary>
    public required IReadOnlyList<WeightInfo?> Inputs { get; init; }

    /// <summary>The first constant input with more than one element, or <see langword="null"/>.</summary>
    public WeightInfo? FirstTensor => Inputs.FirstOrDefault(w => w is not null && w.Shape.Elements() > 1);
}
