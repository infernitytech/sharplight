using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>How a checkpoint's weights are quantized and whether the importer reads that format (see <see cref="HfSupport.GetQuantizationSupport"/>).</summary>
/// <param name="Method">
/// The quantization method as the checkpoint names it (e.g. <c>fp8</c>, <c>gptq</c>, <c>compressed-tensors (int 4-bit)</c>), or null for
/// a checkpoint that is not quantized.
/// </param>
/// <param name="Level">
/// <see cref="SupportLevel.Supported"/> or <see cref="SupportLevel.Unsupported"/>; null when the method is not known to this version
/// (importing it then shows whether its tensors map).
/// </param>
public sealed record HfQuantizationSupport(string? Method, SupportLevel? Level);
