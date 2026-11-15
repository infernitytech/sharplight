namespace Sharplight.Import.Onnx;

/// <summary>One operator set a model imports (<c>ModelProto.opset_import</c>).</summary>
/// <param name="Domain">The operator domain; <c>ai.onnx</c> for the default (empty) domain.</param>
/// <param name="Version">The opset version.</param>
public readonly record struct OnnxOpset(string Domain, long Version);
