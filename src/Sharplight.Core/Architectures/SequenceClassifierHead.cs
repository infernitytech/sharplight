using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The sequence classification head of a bidirectional encoder built on the decoder layers (ModernBERT cross-encoders): the normed hidden
/// rows pooled per sequence (<see cref="CanonicalKeys.ClassifierPooling"/>: "cls" or "mean", default "cls"), then, when the model has them, a
/// dense layer <c>classifier.pre.{weight,bias}</c> [H, H] with the activation <see cref="CanonicalKeys.ClassifierHeadActivation"/>
/// (gelu, gelu_tanh, relu, silu; default gelu) and a norm <c>classifier.pre_norm</c> of the decoder's norm kind, and last
/// <c>classifier.out.{weight,bias}</c> [labels, H], emitted as <see cref="WellKnown.LogitsOutput"/> [sequences, labels].
/// </summary>
internal static class SequenceClassifierHead
{
    /// <summary>The weight whose presence makes a model a sequence classifier.</summary>
    public const string OutWeight = "classifier.out.weight";

    private const string PreWeight = "classifier.pre.weight";
    private const string PreBias = "classifier.pre.bias";
    private const string PreNormWeight = "classifier.pre_norm.weight";
    private const string OutBias = "classifier.out.bias";

    /// <summary>Whether the model has a sequence classification head.</summary>
    /// <param name="src">The model source.</param>
    public static bool IsPresent(ModelSource src) => src.Weight(OutWeight) is not null;

    /// <summary>The number of labels the head scores (the rows of <see cref="OutWeight"/>).</summary>
    /// <param name="src">The model source, which has the head.</param>
    public static int LabelCount(ModelSource src) => (int)src.Weight(OutWeight)!.Shape[0].Fixed;

    /// <summary>Pools <paramref name="hidden"/> per sequence and emits the head's logits.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="src">The model source (pooling and activation keys).</param>
    /// <param name="hidden">The normed hidden rows [T, H] of the packed sequences.</param>
    /// <param name="sequenceSpans">The packed sequences' spans.</param>
    /// <param name="norm">Applies the decoder's norm with the named weight.</param>
    public static void Output(GraphBuilder b, ModelSource src, Value hidden, Value sequenceSpans, Func<Value, string, Value> norm)
    {
        string pooling = PackedSequences.NormalizePooling(src.Get<string?>(CanonicalKeys.ClassifierPooling, null)) ?? "cls";
        var x = PackedSequences.Pool(b, hidden, sequenceSpans, pooling, "classifier_rows");
        if (b.HasWeight(PreWeight))
        {
            x = TowerBlocks.Activation(b, b.Linear(x, PreWeight, PreBias), src.Get(CanonicalKeys.ClassifierHeadActivation, "gelu"));
            if (b.HasWeight(PreNormWeight))
            {
                x = norm(x, PreNormWeight);
            }
        }

        b.Output(WellKnown.LogitsOutput, b.Linear(x, OutWeight, OutBias, "logits"));
    }
}
