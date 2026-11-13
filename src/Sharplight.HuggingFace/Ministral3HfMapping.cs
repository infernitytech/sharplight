using System.Text.RegularExpressions;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>ministral3</c> (Ministral 3 text) and the text model of <c>mistral3</c> (Ministral 3 / Mistral Small 3.x with
/// Pixtral vision; the vision tower is not imported): the Mistral layer with YaRN rope scaling and, when the rope parameters carry
/// <c>llama_4_scaling_beta</c>, Llama-4-style query temperature on every layer (<c>1 + β · ln(1 + ⌊pos / original context⌋)</c>).
/// </summary>
internal sealed partial class Ministral3HfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(LlamaGlobalTensors.Concat(
    [
        new("model.language_model.embed_tokens.weight", "token_embedding.weight"),
        new("model.language_model.norm.weight", "output_norm.weight"),
        new("language_model.model.embed_tokens.weight", "token_embedding.weight"),
        new("language_model.model.norm.weight", "output_norm.weight"),
        new("language_model.lm_head.weight", "lm_head.weight"),
    ]), StringComparer.Ordinal);

    [GeneratedRegex(@"^(?:model\.language_model\.|language_model\.model\.|model\.)layers\.(\d+)\.(.+)$")]
    private static partial Regex TextLayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["ministral3", "mistral3"];

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => TextLayerName();

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.Contains("vision_tower.", StringComparison.Ordinal) || name.Contains("multi_modal_projector.", StringComparison.Ordinal);

    /// <inheritdoc/>
    /// <exception cref="InvalidDataException">The temperature beta is set without an original context length.</exception>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        double beta = HfMappingHelpers.NumberOr(c.RopeParameters, "llama_4_scaling_beta", 0);
        if (beta == 0)
        {
            return;
        }

        double length = HfMappingHelpers.NumberOr(c.RopeParameters, "original_max_position_embeddings", 0);
        if (length <= 0)
        {
            throw new InvalidDataException("ministral3: llama_4_scaling_beta needs rope original_max_position_embeddings.");
        }

        md["decoder.attention.temperature_scale"] = (float)beta;
        md["decoder.attention.temperature_length"] = (float)length;
    }
}
