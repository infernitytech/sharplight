using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Hyper-parameter and weight conventions shared by several architecture mappings.</summary>
internal static class HfMappingHelpers
{
    /// <summary>The <c>layer_types</c> entry of layers that use a sliding attention window.</summary>
    private const string SlidingAttention = "sliding_attention";

    /// <summary>
    /// Adds <c>{prefix}context_length</c> from the first present of <paramref name="keys"/>
    /// (default <c>max_position_embeddings</c>).
    /// </summary>
    public static void AddContextLength(Dictionary<string, object> md, string prefix, HfConfig c, params string[] keys)
    {
        if (c.GetInt(keys.Length > 0 ? keys : ["max_position_embeddings"]) is { } contextLength)
        {
            md[prefix + "context_length"] = contextLength;
        }
    }

    /// <summary>Per-layer 0/1 flags: 1 for layers whose <c>layer_types</c> entry is <c>sliding_attention</c>.</summary>
    public static int[] SlidingFlags(string[] layerTypes)
        => layerTypes.Select(type => type == SlidingAttention ? 1 : 0).ToArray();

    /// <summary>A numeric property of an optional JSON object, or <paramref name="fallback"/> when the object or property is missing or not a number.</summary>
    public static double NumberOr(JsonElement? element, string key, double fallback)
        => OptionalNumber(element, key) ?? fallback;

    /// <summary>A numeric property of an optional JSON object, or null when the object or property is missing or not a number.</summary>
    public static double? OptionalNumber(JsonElement? element, string key)
        => element is { } e && e.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number ? value.GetDouble() : null;

    /// <summary>A lazily encoded F32 weight over <paramref name="values"/> (e.g. rope frequency factors).</summary>
    public static WeightInfo F32Weight(string name, float[] values)
        => WeightInfo.Derived(name, DataType.F32, new Shape(values.Length), () => HfTransforms.EncodeF32(values));
}
