using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>chameleon</c> (Chameleon 7B / 34B, text generation): Llama layers whose q and k get a layer norm over each head with a gain
/// and bias per head ([heads, head_dim], canonical <c>attention.qk_norm_type</c> = "layer", epsilon 1e-5). With <c>swin_norm</c> the layer norms
/// move behind the attention and MLP outputs (<c>input_layernorm</c> becomes <c>attention.out_norm</c>, <c>post_attention_layernorm</c>
/// <c>mlp.out_norm</c>, no pre-norms). The image tokens of <c>vocabulary_map</c> (names starting with <c>IMGIMG</c>) are suppressed in the logits
/// as transformers does for text output; the VQ-GAN image tokenizer (<c>model.vqmodel.*</c>) is not imported.
/// </summary>
internal sealed class ChameleonHfMapping : HfDecoderFamilyMapping
{
    private const string ImageTokenPrefix = "IMGIMG";

    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = With(LlamaLayerTensors,
        ("self_attn.q_norm.bias", "attention.q_norm.bias"),
        ("self_attn.k_norm.bias", "attention.k_norm.bias"));

    // Swin-norm checkpoints: the layer norms follow the sub-layers instead of preceding them.
    private static readonly Dictionary<string, string> s_swinNorms = new(StringComparer.Ordinal)
    {
        ["input_layernorm.weight"] = "attention.out_norm.weight",
        ["post_attention_layernorm.weight"] = "mlp.out_norm.weight",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["chameleon"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override bool IsIgnored(string name) => base.IsIgnored(name) || name.StartsWith("model.vqmodel.", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (c.GetBool("swin_norm") != true || !s_swinNorms.TryGetValue(suffix, out var canonical))
        {
            return false;
        }

        collector.Add(NormWeight(w, layerPrefix + canonical));
        return true;
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.attention.qk_norm_type"] = "layer";
        md["decoder.attention.qk_norm_eps"] = 1e-5f;
        if (c.GetBool("swin_norm") == true)
        {
            md["decoder.pre_norm"] = false;
        }

        if (ImageTokenRanges(c) is { Length: > 0 } ranges)
        {
            md["decoder.logits.suppressed_ranges"] = ranges;
        }
    }

    /// <summary>The ids of the image tokens in <c>vocabulary_map</c> as sorted [start, end) pairs.</summary>
    private static int[] ImageTokenRanges(HfConfig c)
    {
        if (c.Get("vocabulary_map") is not { ValueKind: JsonValueKind.Object } map)
        {
            return [];
        }

        var ids = map.EnumerateObject()
            .Where(p => p.Name.StartsWith(ImageTokenPrefix, StringComparison.Ordinal) && p.Value.ValueKind == JsonValueKind.Number)
            .Select(p => p.Value.GetInt32())
            .Order()
            .ToArray();
        var ranges = new List<int>();
        for (int i = 0; i < ids.Length; i++)
        {
            int start = ids[i];
            while (i + 1 < ids.Length && ids[i + 1] <= ids[i] + 1)
            {
                i++;
            }

            ranges.Add(start);
            ranges.Add(ids[i] + 1);
        }

        return [.. ranges];
    }
}
