namespace Sharplight.Hub;

/// <summary>One ONNX model file of a repository and its tag (empty for the default <c>model.onnx</c>).</summary>
internal sealed record OnnxVariant(string Label, string Path);
