using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// The "endpoint" span scoring head (vllm-sr Decision 2.0): option i is read at the last row of its span, its question at the last row of the
/// question span. With o = LayerNorm(option row) and q = LayerNorm(question row), the score is
/// (key·o)·(query·q)/√D + scorer·gelu(option_mlp·o + b + question_mlp·q).
/// </summary>
/// <remarks>
/// Weights below <c>span_head.</c>: <c>option_norm</c> / <c>question_norm</c> (weight and bias), <c>key.weight</c> and <c>query.weight</c>
/// [D, H], <c>option_mlp.weight</c> [D, H] with <c>option_mlp.bias</c> [D], <c>question_mlp.weight</c> [D, H] and <c>scorer.weight</c> [1, D].
/// </remarks>
internal static class EndpointSpanHead
{
    /// <summary>The scores [O, 1] of the options.</summary>
    public static Value Build(GraphBuilder b, Value hidden, SpanScoringInputs spans, float eps)
    {
        using var _ = b.Group("span_head");
        var options = b.Emit1("SequencePool", [hidden, spans.OptionSpans], Attributes.From(("mode", "last")), "span_option_rows");        // [O, H]
        var questions = b.Emit1("SequencePool", [hidden, spans.QuestionSpans], Attributes.From(("mode", "last")), "span_question_rows");  // [Q, H]
        var o = SpanScoring.LayerNorm(b, options, "span_head.option_norm", eps);
        var q = SpanScoring.LayerNorm(b, b.Gather(questions, spans.OptionQuestions), "span_head.question_norm", eps);                    // [O, H]

        var key = b.Linear(o, "span_head.key.weight");
        var query = b.Linear(q, "span_head.query.weight");
        long width = b.Weight("span_head.key.weight").Shape[0].Fixed;
        var bilinear = b.Scale(b.Emit1("ReduceSum", [b.Mul(key, query)]), 1f / MathF.Sqrt(width), name: "span_bilinear");               // [O, 1]

        var mlp = b.Add(b.Linear(o, "span_head.option_mlp.weight", "span_head.option_mlp.bias"), b.Linear(q, "span_head.question_mlp.weight"));
        var nonlinear = b.Linear(b.Unary("Gelu", mlp), "span_head.scorer.weight", name: "span_nonlinear");                              // [O, 1]
        return b.Add(bilinear, nonlinear, "span_scores");
    }
}
