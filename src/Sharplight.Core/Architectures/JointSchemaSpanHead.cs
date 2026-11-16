using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// The "joint_schema" span scoring head (Cloudflare Clef): all questions of a prompt are decided jointly. With h = LayerNorm(hidden) and the
/// memory m = memory·h:
/// <list type="number">
/// <item>Every option starts as option_context·mean(h over the option) + option_lexical·mean(output embeddings of the option's tokens) +
/// option_question·mean(h over its question), then passes the routing layers (pre-norm cross-attention to the memory, GELU feed-forward).</item>
/// <item>Every question becomes a field: question·mean(h over the question) + LayerNorm(softmax-weighted summary of its own options) +
/// global·(last row of h) + the embedding of its kind; the fields pass the joint layers (pre-norm self-attention, cross-attention to the memory,
/// GELU feed-forward) and a final LayerNorm.</item>
/// <item>The score of an option is s0·prior + s2·(s1·cos(field, o) + residual), with o = LayerNorm(option), the prior the cosine of the
/// option's lexical vector and (question mean + last row), and the residual an MLP of [field, o, field·o, |field − o|].</item>
/// </list>
/// The attention of the layers is over every row of the run, so a run holds one sequence.
/// </summary>
/// <remarks>
/// Weights below <c>span_head.</c> (W the head width, H the hidden size): <c>hidden_norm</c> [H]; <c>memory</c>, <c>question</c>,
/// <c>option_question</c>, <c>global</c>, <c>option_context</c>, <c>option_lexical</c> <c>.weight</c> [W, H]; <c>type_embedding.weight</c> [3, W];
/// per routing layer i <c>routing.{i}.</c>: <c>query_norm</c>, <c>memory_norm</c>, <c>attention.{q,k,v,o}</c>, <c>ffn_norm</c>, <c>ffn_up</c>,
/// <c>ffn_down</c>; per joint layer i <c>layers.{i}.</c>: <c>self_norm</c>, <c>self_attention.{q,k,v,o}</c>, <c>cross_norm</c>,
/// <c>cross_attention.{q,k,v,o}</c>, <c>ffn_norm</c>, <c>ffn_up</c>, <c>ffn_down</c>; <c>option_summary_norm</c>, <c>field_norm</c>,
/// <c>option_norm</c>; <c>scorer</c> [W, 4W] and <c>scorer_out</c> [1, W]; <c>scales</c> [3] = (s0, s1, s2). Norms are LayerNorms and every
/// projection of the layers has a bias.
/// </remarks>
internal static class JointSchemaSpanHead
{
    private const string P = "span_head.";
    private const float CosineEps = 1e-24f;   // L2 normalization x / sqrt(Σx² + eps) ≈ torch's x / max(|x|, 1e-12)

    /// <summary>The scores [O, 1] of the options.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="hidden">Final hidden states [T, H].</param>
    /// <param name="tokens">Token ids [T].</param>
    /// <param name="sequenceSpans">Sequence spans [B, 2].</param>
    /// <param name="spans">The span tables.</param>
    /// <param name="s">The head settings.</param>
    public static Value Build(GraphBuilder b, Value hidden, Value tokens, Value sequenceSpans, SpanScoringInputs spans, JointSchemaSpanHeadSettings s)
    {
        using var _ = b.Group("span_head");
        var h = Norm(b, hidden, "hidden_norm", s);
        var memory = b.Linear(h, P + "memory.weight", name: "span_memory");                                                   // [T, W]
        long width = b.Weight(P + "memory.weight").Shape[0].Fixed;
        var global = b.Gather(Pool(b, h, sequenceSpans, "last"), spans.QuestionSequences, "span_global");                    // [Q, H]
        var question = Pool(b, h, spans.QuestionSpans, "mean");                                                                // [Q, H]
        var context = Pool(b, h, spans.OptionSpans, "mean");                                                                   // [O, H]
        var lexical = Pool(b, b.Gather(b.Weight(s.OutputEmbedding), tokens), spans.OptionSpans, "mean");                       // [O, H]

        var options = b.Add(b.Add(b.Linear(context, P + "option_context.weight"), b.Linear(lexical, P + "option_lexical.weight")),
            b.Gather(b.Linear(question, P + "option_question.weight"), spans.OptionQuestions), "span_option_queries");          // [O, W]
        for (int i = 0; i < s.RoutingLayers; i++)
        {
            string p = $"routing.{i}.";
            options = b.Add(options, Attention(b, Norm(b, options, p + "query_norm", s), Norm(b, memory, p + "memory_norm", s), p + "attention", s));
            options = b.Add(options, FeedForward(b, options, p, s), $"span_routing_{i}");
        }

        // Each question weights its own options (consecutive rows of the option table): a softmax per question.
        var baseFields = b.Linear(question, P + "question.weight");                                                            // [Q, W]
        var logits = b.Scale(b.Emit1("ReduceSum", [b.Mul(b.Gather(baseFields, spans.OptionQuestions), options)]), 1f / MathF.Sqrt(width));
        // The per-question maximum, repeated to 8 columns for the gather (row gathers move whole vectors), then back to one column.
        var maxima = Pool(b, logits, spans.QuestionOptions, "max");
        var peak = b.Slice(b.Gather(b.Concat(1, [.. Enumerable.Repeat(maxima, 8)]), spans.OptionQuestions), 1, 0, 1);
        var weights = b.Unary("Exp", b.Sub(logits, peak));                                                                     // [O, 1]
        var summary = b.Binary("Div", Pool(b, b.Mul(options, weights), spans.QuestionOptions, "sum"), Pool(b, weights, spans.QuestionOptions, "sum"));
        var fields = b.Add(b.Add(baseFields, Norm(b, summary, "option_summary_norm", s)),
            b.Add(b.Linear(global, P + "global.weight"), b.Gather(b.Weight(P + "type_embedding.weight"), spans.QuestionKinds)), "span_fields");
        for (int i = 0; i < s.Layers; i++)
        {
            string p = $"layers.{i}.";
            var x = Norm(b, fields, p + "self_norm", s);
            fields = b.Add(fields, Attention(b, x, x, p + "self_attention", s));
            fields = b.Add(fields, Attention(b, Norm(b, fields, p + "cross_norm", s), memory, p + "cross_attention", s));
            fields = b.Add(fields, FeedForward(b, fields, p, s), $"span_joint_{i}");
        }

        fields = Norm(b, fields, "field_norm", s);

        var anchor = b.Gather(b.L2Norm(b.Add(question, global), CosineEps), spans.OptionQuestions);                            // [O, H]
        var prior = b.Emit1("ReduceSum", [b.Mul(b.L2Norm(lexical, CosineEps), anchor)], name: "span_prior");                   // [O, 1]
        var o = Norm(b, options, "option_norm", s);
        var field = b.Gather(fields, spans.OptionQuestions);                                                                    // [O, W]
        var cosine = b.Emit1("ReduceSum", [b.Mul(b.L2Norm(field, CosineEps), b.L2Norm(o, CosineEps))], name: "span_cosine");
        var features = b.Concat(1, field, o, b.Mul(field, o), b.Unary("Abs", b.Sub(field, o)));                                 // [O, 4W]
        var residual = b.Linear(b.Unary("Gelu", b.Linear(features, P + "scorer.weight", P + "scorer.bias")), P + "scorer_out.weight", P + "scorer_out.bias",
            "span_residual");                                                                                                   // [O, 1]

        var scales = b.Weight(P + "scales");
        var joint = b.Add(b.Mul(cosine, b.Slice(scales, 0, 1, 1)), residual);
        return b.Add(b.Mul(prior, b.Slice(scales, 0, 0, 1)), b.Mul(joint, b.Slice(scales, 0, 2, 1)), "span_scores");
    }

    private static Value Pool(GraphBuilder b, Value x, Value spans, string mode) => b.Emit1("SequencePool", [x, spans], Attributes.From(("mode", mode)));

    private static Value Norm(GraphBuilder b, Value x, string name, JointSchemaSpanHeadSettings s) => SpanScoring.LayerNorm(b, x, P + name, s.Eps);

    /// <summary>Multi-head attention with biased projections of <paramref name="x"/> [N, W] over <paramref name="kv"/> [S, W], unmasked.</summary>
    private static Value Attention(GraphBuilder b, Value x, Value kv, string name, JointSchemaSpanHeadSettings s)
    {
        string p = P + name + ".";
        long width = b.Weight(p + "q.weight").Shape[0].Fixed;
        int d = (int)(width / s.Heads);
        var q = b.Reshape(b.Linear(x, p + "q.weight", p + "q.bias"), 0, s.Heads, d);
        var k = b.Reshape(b.Linear(kv, p + "k.weight", p + "k.bias"), 0, s.Heads, d);
        var v = b.Reshape(b.Linear(kv, p + "v.weight", p + "v.bias"), 0, s.Heads, d);
        var y = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(d)), ("causal", false)));
        return b.Linear(b.Reshape(y, 0, width), p + "o.weight", p + "o.bias");
    }

    private static Value FeedForward(GraphBuilder b, Value x, string p, JointSchemaSpanHeadSettings s)
        => b.Linear(b.Unary("Gelu", b.Linear(Norm(b, x, p + "ffn_norm", s), P + p + "ffn_up.weight", P + p + "ffn_up.bias")), P + p + "ffn_down.weight", P + p + "ffn_down.bias");
}
