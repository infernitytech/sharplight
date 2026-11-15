using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face <c>neo_chat</c> (SenseNova-U1, <c>NEOChatModel</c>): the understanding branch is a Qwen3 language model under
/// <c>language_model.*</c> (config in <c>llm_config</c>) whose heads carry a temporal and a spatial half. Each half of a q/k head is RMS-normed
/// with its own weight (<c>q_norm</c> / <c>q_norm_hw</c>, joined into one canonical norm weight with <c>attention.qk_norm_groups</c> = 2), and
/// M-RoPE rotates the temporal half by the token position and the spatial quarters by the image row and column (base <c>rope_theta_hw</c>). Text
/// tokens sit at row and column 0, so for text the rotation is a partial rope over the first half of the head. The image-generation branch
/// (<c>*_mot_gen</c>, <c>fm_modules.*</c>) and the vision embeddings are not imported (text generation only).
/// </summary>
internal sealed partial class SenseNovaU1HfMapping : HfDecoderFamilyMapping
{
    private const string SpatialSuffix = "_hw.weight";

    private static readonly ConditionalWeakTable<HfWeightCollector, HfPartJoiner> s_norms = [];

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["language_model.model.embed_tokens.weight"] = "token_embedding.weight",
        ["language_model.model.norm.weight"] = "output_norm.weight",
        ["language_model.lm_head.weight"] = "lm_head.weight",
    };

    [GeneratedRegex(@"^language_model\.model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LanguageLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["neo_chat"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => LanguageLayerName();

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => name.StartsWith("vision_model.", StringComparison.Ordinal) || name.StartsWith("fm_modules.", StringComparison.Ordinal)
            || name.Contains("_mot_gen", StringComparison.Ordinal) || base.IsIgnored(name);

    /// <summary>The temporal half of the head rotates (text positions); the spatial quarters stay at position 0.</summary>
    protected override int RopeDimension(HfConfig c, int headDim) => headDim / 2;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c) => md["decoder.attention.qk_norm_groups"] = 2;

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        string? kind = suffix switch
        {
            "self_attn.q_norm.weight" or "self_attn.q_norm_hw.weight" => "q",
            "self_attn.k_norm.weight" or "self_attn.k_norm_hw.weight" => "k",
            _ => null,
        };
        if (kind is null)
        {
            return false;
        }

        int part = suffix.EndsWith(SpatialSuffix, StringComparison.Ordinal) ? 1 : 0;
        s_norms.GetValue(collector, _ => HfPartJoiner.RowShards()).Add($"{layerPrefix}attention.{kind}_norm.weight", part, w);
        return true;
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        if (s_norms.TryGetValue(collector, out var norms))
        {
            norms.JoinInto(collector, 2);
            s_norms.Remove(collector);
        }
    }
}
