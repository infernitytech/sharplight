using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>vllm-decision2</c> (vllm-sr Decision 2.0): the Qwen3 decoder with the "endpoint" span scoring head (<c>decision.*</c> tensors →
/// <c>span_head.*</c>, its LayerNorm epsilon from <c>attention.layer_norm_epsilon</c>), run through the score_spans graph. The per-question score
/// biases (<c>decision.score_bias.*</c>) are calibration offsets the caller applies to the scores of score questions; they are not imported.
/// </summary>
internal sealed class VllmDecision2GgufMapping : GgufDecoderFamilyMapping
{
    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["vllm-decision2"];

    private static readonly Dictionary<string, string> s_head = new(StringComparer.Ordinal)
    {
        ["decision.option_norm.weight"] = "span_head.option_norm.weight",
        ["decision.option_norm.bias"] = "span_head.option_norm.bias",
        ["decision.query_norm.weight"] = "span_head.question_norm.weight",
        ["decision.query_norm.bias"] = "span_head.question_norm.bias",
        ["decision.key.weight"] = "span_head.key.weight",
        ["decision.query.weight"] = "span_head.query.weight",
        ["decision.option_mlp.weight"] = "span_head.option_mlp.weight",
        ["decision.option_mlp.bias"] = "span_head.option_mlp.bias",
        ["decision.query_mlp.weight"] = "span_head.question_mlp.weight",
        ["decision.scorer_out.weight"] = "span_head.scorer.weight",
    };

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? GlobalTensorOverrides => s_head;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        md[CanonicalKeys.SpanHeadType] = "endpoint";
        md[CanonicalKeys.SpanHeadNormEps] = g.Float("attention.layer_norm_epsilon", 1e-5f);
    }
}
