using Sharplight.Tensors;

namespace Sharplight.Import.WhisperCpp;

/// <summary>One tensor of a whisper.cpp model file.</summary>
/// <param name="Name">The whisper.cpp tensor name (e.g. <c>decoder.blocks.0.attn.query.weight</c>).</param>
/// <param name="Type">The storage type.</param>
/// <param name="Shape">The row-major shape (ggml's dimensions reversed).</param>
/// <param name="Offset">File offset of the data.</param>
/// <param name="Length">Data length in bytes.</param>
internal sealed record WhisperCppTensor(string Name, DataType Type, Shape Shape, long Offset, long Length);
