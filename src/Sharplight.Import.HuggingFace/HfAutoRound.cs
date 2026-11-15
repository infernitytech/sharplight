using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Intel AutoRound checkpoints (<c>quant_method</c> <c>auto-round</c> / <c>autoround</c> / <c>intel/auto-round</c>): the native
/// <c>auto_round</c> export stores integer weights in the GPTQ layout (<c>packing_format</c> <c>auto_round:auto_gptq</c>:
/// <see cref="HfGptqLayouts"/>, zero points stored minus one as GPTQ's original format does) or in the AWQ GEMM layout
/// (<c>auto_round:auto_awq</c>: <see cref="HfAwqLayouts"/>). The <c>auto_gptq</c> / <c>auto_awq</c> exports declare plain GPTQ /
/// AWQ and need nothing here. Other packings (ExLlamaV2, FP data types, activation-quantized MX / NV formats) are not read.
/// </summary>
internal static class HfAutoRound
{
    /// <summary>Whether <paramref name="method"/> is an AutoRound <c>quant_method</c>.</summary>
    public static bool IsAutoRound(string? method) => method?.ToLowerInvariant() is "auto-round" or "autoround" or "intel/auto-round";

    /// <summary>
    /// The method whose layout the weights have: <c>gptq</c> or <c>awq</c> for the supported integer packings, else null.
    /// </summary>
    /// <param name="settings">The <c>quantization_config</c>.</param>
    public static string? Layout(JsonElement settings)
    {
        string? dataType = HfPackedBits.String(settings, "data_type");
        if (dataType is not (null or "int"))
        {
            return null;
        }

        string packing = (HfPackedBits.String(settings, "packing_format") ?? HfPackedBits.String(settings, "backend") ?? "").ToLowerInvariant();
        if (packing.Contains("exllama", StringComparison.Ordinal) || packing.Contains("marlin", StringComparison.Ordinal))
        {
            return null;
        }

        return packing.Contains("awq", StringComparison.Ordinal) ? "awq" : packing.Contains("gptq", StringComparison.Ordinal) ? "gptq" : null;
    }
}
