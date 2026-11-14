using Sharplight.Graph;

namespace Sharplight.Onnx;

// Loss operators: NegativeLogLikelihoodLoss and SoftmaxCrossEntropyLoss, composed from gathers, masks and reductions.
internal sealed partial class OnnxTranslator
{
    /// <summary>NegativeLogLikelihoodLoss: input [N, C, d...] (log-probabilities), target [N, d...], optional weight [C].</summary>
    private void NegativeLogLikelihoodLoss(OnnxNode n, TranslatedTensor?[] ins)
    {
        // Without class weights an ignore_index of -1 keeps every element in the mean (the reference's convention).
        long? ignore = n.Has("ignore_index") ? n.Int("ignore_index", 0) : null;
        bool countIgnored = Opt(ins, 2) is null && ignore == -1;
        SetDyn(n, 0, NllLoss(n, X(ins, 0), ins[1]!, Opt(ins, 2), ignore, countIgnored));
    }

    /// <summary>SoftmaxCrossEntropyLoss: NLL of LogSoftmax over the class axis; output #1 is the log-probabilities.</summary>
    private void SoftmaxCrossEntropyLoss(OnnxNode n, TranslatedTensor?[] ins)
    {
        var scores = X(ins, 0);
        var perm = Enumerable.Range(0, scores.Rank).ToArray();
        (perm[1], perm[^1]) = (perm[^1], perm[1]);
        var logProb = Permute(_b.Emit1("Softmax", [Permute(scores, perm)], Attributes.From(("log", true))), perm);
        long? ignore = n.Has("ignore_index") ? n.Int("ignore_index", 0) : null;
        SetDyn(n, 0, NllLoss(n, logProb, ins[1]!, Opt(ins, 2), ignore, false));
        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, logProb);
        }
    }

    /// <summary>
    /// loss[n, d...] = -weight[t] · input[n, t, d...] with t = target[n, d...]; elements whose target is ignore_index count zero. Reductions:
    /// none, sum, or mean = sum / sum of the element weights (the element count when nothing is weighted or ignored).
    /// </summary>
    private Value NllLoss(OnnxNode n, Value input, TranslatedTensor target, TranslatedTensor? weight, long? ignore, bool countIgnored)
    {
        if (input.Rank < 2)
        {
            throw new InvalidDataException($"{n.OpType} input must be at least [N, C]");
        }

        string reduction = n.Str("reduction", "mean");
        var labels = Idx(target);
        Value? valid = null;
        if (ignore is { } ignored)
        {
            var labelValues = AsValue(target);
            valid = _b.Unary("Not", _b.Binary("Equal", labelValues, ConstValue(OnnxConst.Scalar(ignored), Nm(n) + ".ignore")));
            labels = _b.Emit1("Cast", [Emit("Where", [valid, labelValues, ConstValue(OnnxConst.Scalar(0f), Nm(n) + ".zero")])], Attributes.From(("type", "i32")));
        }

        // [N, d...] → [N, 1, d...]: one class index per element.
        long[] withClassAxis = [.. Fixed(labels).Take(1), 1, .. Fixed(labels).Skip(1)];
        var picked = Emit("GatherElements", [input, Rs(labels, withClassAxis)], Attributes.From(("axis", 1L)));
        var loss = _b.Scale(Rs(picked, Fixed(labels)), -1f);
        Value? elementWeights = null;
        if (weight is not null)
        {
            elementWeights = Emit("GatherAxis", [AsValue(weight), labels], Attributes.From(("axis", 0L)));
            if (valid is not null)
            {
                elementWeights = _b.Mul(elementWeights, valid);
            }
        }
        else if (valid is not null)
        {
            elementWeights = valid;
        }

        if (elementWeights is not null)
        {
            loss = _b.Mul(loss, elementWeights);
        }

        long[] all = [.. Enumerable.Range(0, loss.Rank).Select(a => (long)a)];
        Value Total(Value v, string mode) => Emit("Reduce", [v], Attributes.From(("mode", mode), ("axes", all), ("keepdims", false)));
        return reduction switch
        {
            "none" => loss,
            "sum" => Total(loss, "sum"),
            "mean" when elementWeights is not null && !countIgnored => _b.Binary("Div", Total(loss, "sum"), Total(elementWeights, "sum")),
            "mean" => Total(loss, "mean"),
            _ => throw new NotSupportedException($"{n.OpType} reduction '{reduction}'"),
        };
    }
}
