namespace Sharplight.Onnx;

internal sealed class OnnxGraph
{
    public string Name { get; init; } = "";
    public IReadOnlyList<OnnxNode> Nodes { get; init; } = [];
    public IReadOnlyList<OnnxTensor> Initializers { get; init; } = [];
    public IReadOnlyList<OnnxValueInfo> Inputs { get; init; } = [];
    public IReadOnlyList<OnnxValueInfo> Outputs { get; init; } = [];
    public IReadOnlyList<OnnxValueInfo> ValueInfo { get; init; } = [];
    public string DocString { get; init; } = "";
}
