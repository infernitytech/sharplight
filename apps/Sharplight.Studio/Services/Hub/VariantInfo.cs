namespace Sharplight.Studio.Services.Hub;

/// <summary>One pullable variant of a repository (a GGUF quantization, the safetensors checkpoint, an ONNX file).</summary>
/// <param name="Tag">The tag to pull it with; empty for a single-variant repository.</param>
/// <param name="Bytes">Download size.</param>
/// <param name="FileCount">Files downloaded.</param>
/// <param name="HasProjector">A vision projector is downloaded with it.</param>
/// <param name="Recommended">The variant pulled without a tag.</param>
internal sealed record VariantInfo(string Tag, long Bytes, int FileCount, bool HasProjector, bool Recommended);
