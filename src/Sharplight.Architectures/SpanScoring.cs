using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The span scoring head of decision models (<see cref="WellKnown.ScoreSpansGraph"/>): a prompt holds questions and their options as spans of
/// the packed rows, and the head turns the backbone's final hidden states into one score per option. Shared by every adapter whose model has a
/// head below <c>span_head.</c>; the head type comes from the canonical key <c>span_head.type</c>:
/// <list type="bullet">
/// <item><c>endpoint</c> (<see cref="EndpointSpanHead"/>, vllm-sr Decision 2.0): the last row of each option against the last row of its
/// question.</item>
/// </list>
/// The LayerNorm epsilon of the head is <c>span_head.norm_eps</c> (default 1e-5).
/// </summary>
internal static class SpanScoring
{
    /// <summary>Name of the question table input [Q, 4].</summary>
    public const string QuestionSpansInput = "question_spans";
    /// <summary>Name of the option table input [O, 3].</summary>
    public const string OptionSpansInput = "option_spans";

    /// <summary>Whether the model has a span scoring head.</summary>
    public static bool IsPresent(ModelSource src) => src.Has(CanonicalKeys.SpanHeadType);

    /// <summary>The head type (<c>span_head.type</c>).</summary>
    public static string HeadType(ModelSource src) => src.Get<string>(CanonicalKeys.SpanHeadType);

    /// <summary>Declares the question [Q, 4] and option [O, 3] tables and splits their columns.</summary>
    public static SpanScoringInputs Declare(GraphBuilder b)
    {
        var questions = b.Input(QuestionSpansInput, InputRole.QuestionSpans, DataType.I32, new Shape(Dim.Sym(WellKnown.QuestionSymbol), 4));
        var options = b.Input(OptionSpansInput, InputRole.OptionSpans, DataType.I32, new Shape(Dim.Sym(WellKnown.OptionSymbol), 3));
        return new SpanScoringInputs(
            b.Contiguous(b.Slice(questions, 1, 0, 2)),
            b.Reshape(b.Contiguous(b.Slice(questions, 1, 2, 1)), 0),
            b.Reshape(b.Contiguous(b.Slice(questions, 1, 3, 1)), 0),
            b.Contiguous(b.Slice(options, 1, 0, 2)),
            b.Reshape(b.Contiguous(b.Slice(options, 1, 2, 1)), 0));
    }

    /// <summary>
    /// Emits <see cref="WellKnown.ScoresOutput"/> [O, 1] from the final hidden states <paramref name="hidden"/> [T, H] (after the output norm).
    /// </summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="src">The model source (head type and settings).</param>
    /// <param name="hidden">Final hidden states [T, H].</param>
    /// <param name="spans">The span tables.</param>
    /// <exception cref="NotSupportedException">The head type is unknown.</exception>
    public static void Output(GraphBuilder b, ModelSource src, Value hidden, SpanScoringInputs spans)
    {
        float eps = src.Get(CanonicalKeys.SpanHeadNormEps, 1e-5f);
        var scores = HeadType(src) switch
        {
            "endpoint" => EndpointSpanHead.Build(b, hidden, spans, eps),
            var t => throw new NotSupportedException($"Unknown span head type '{t}' (endpoint)."),
        };
        b.Output(WellKnown.ScoresOutput, scores);
    }

    /// <summary>LayerNorm with weight and bias <c>{name}.weight</c> / <c>{name}.bias</c>.</summary>
    internal static Value LayerNorm(GraphBuilder b, Value x, string name, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));
}
