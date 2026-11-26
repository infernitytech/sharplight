using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>aion_instruct</c> (Microsoft Aion 1.0, remote code): the Qwen3 layer (per-head q/k RMS norms, SwiGLU MLP, tied embeddings) with
/// sliding-window layers from <c>layer_types</c>. Each layer type has its own <c>rope_parameters</c>: the full-attention layers rotate with YaRN, the
/// sliding layers with plain RoPE on their own base, so the YaRN frequency factors and magnitude apply to the full layers only → canonical
/// <c>decoder</c>.
/// </summary>
internal sealed class AionInstructHfMapping : HfDecoderFamilyMapping
{
    private const string FullAttention = "full_attention";
    private const string SlidingAttention = "sliding_attention";

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["aion_instruct"];

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c) => ThetaOf(c, FullAttention);

    /// <inheritdoc/>
    public override IEnumerable<string> Validate(HfConfig config)
    {
        if (config.RopeParameters is { } parameters && parameters.TryGetProperty(SlidingAttention, out var sliding) && sliding.ValueKind == JsonValueKind.Object
            && sliding.TryGetProperty("rope_type", out var rope) && rope.GetString() is { } type && type != "default")
        {
            yield return $"aion_instruct: rope_type '{type}' of the sliding layers is not supported.";
        }
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        if (AddSlidingLayers(md, c) is not null)
        {
            md["decoder.rope.local_base"] = ThetaOf(c, SlidingAttention);
        }
    }

    private static float ThetaOf(HfConfig c, string layerType)
        => (float)(c.LayerTypeRopeDouble(layerType, "rope_theta") ?? c.RopeDouble("rope_theta", "rope_theta") ?? 10000.0);
}
