namespace Sharplight.Import.Gguf;

/// <summary>
/// One entry of a GGUF tensor table as <see cref="GgufHeaderProbe.ReadTensorTable"/> reads it from the start of a file: the tensor's
/// name, type id and dimensions, without its data.
/// </summary>
/// <param name="Name">The tensor's name as stored in the file (llama.cpp naming).</param>
/// <param name="GgmlType">The ggml type id; it may be unknown to this version (see <see cref="GgufSupport.GetTensorType"/>).</param>
/// <param name="Dimensions">The ggml dimensions, innermost first.</param>
public sealed record GgufTensorEntry(string Name, int GgmlType, IReadOnlyList<long> Dimensions);
