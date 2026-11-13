using Sharplight.Tensors;

namespace Sharplight.Convert;

/// <summary>Parsed <c>convert</c> options.</summary>
/// <param name="SourcePath">GGUF file, Hugging Face directory, ONNX file or package.</param>
/// <param name="DestinationPath">The package to write.</param>
/// <param name="Optimize">Run the device-independent graph optimization before saving.</param>
/// <param name="KvCacheType">KV cache element type baked into LLM graphs.</param>
/// <param name="ContextLength">Context length LLM graphs are built for (0 = the model's own).</param>
/// <param name="MmprojPath">Vision projector (GGUF mmproj) to build into the package, or null.</param>
/// <param name="InputShapes">Shapes of ONNX inputs with dynamic dimensions.</param>
internal sealed record ConvertSettings(
    string SourcePath,
    string DestinationPath,
    bool Optimize,
    DataType KvCacheType,
    int ContextLength,
    string? MmprojPath,
    IReadOnlyDictionary<string, long[]> InputShapes);
