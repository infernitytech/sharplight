using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// The per-module weight specs of an AMD Quark <c>quantization_config</c>: <c>global_quant_config.weight</c> unless a
/// <c>layer_quant_config</c> entry (module name or <c>*</c> / <c>?</c> pattern) overrides it, and the export's packing order.
/// </summary>
internal static class HfQuarkSpecs
{
    /// <summary>The weight spec that applies to the module <paramref name="stem"/> (without <c>.weight</c>), or null when its weight is not quantized.</summary>
    /// <param name="settings">The <c>quantization_config</c>.</param>
    /// <param name="stem">The module name as the checkpoint spells it.</param>
    public static JsonElement? WeightSpec(JsonElement settings, string stem)
    {
        if (settings.TryGetProperty("layer_quant_config", out var layers) && layers.ValueKind == JsonValueKind.Object)
        {
            foreach (var layer in layers.EnumerateObject())
            {
                if (Matches(layer.Name, stem))
                {
                    return Weight(layer.Value);
                }
            }
        }

        return settings.TryGetProperty("global_quant_config", out var global) ? Weight(global) : null;
    }

    /// <summary>Whether the int4 packing interleaves the columns of each word (<c>export.pack_method</c> <c>reorder</c>, the default).</summary>
    /// <param name="settings">The <c>quantization_config</c>.</param>
    public static bool Reorder(JsonElement settings)
        => !(settings.TryGetProperty("export", out var export) && HfPackedBits.String(export, "pack_method") == "order");

    private static JsonElement? Weight(JsonElement spec)
        => spec.ValueKind == JsonValueKind.Object && spec.TryGetProperty("weight", out var weight) && weight.ValueKind == JsonValueKind.Object ? weight : null;

    private static bool Matches(string pattern, string name)
    {
        if (!pattern.Contains('*') && !pattern.Contains('?'))
        {
            return pattern == name;
        }

        var regex = "^" + System.Text.RegularExpressions.Regex.Escape(pattern).Replace("\\*", ".*").Replace("\\?", ".") + "$";
        return System.Text.RegularExpressions.Regex.IsMatch(name, regex);
    }
}
