namespace Sharplight.Onnx;

// Zero-size tensors: a tensor with a zero extent carries no data, so it is known at import time (see EmptyInputDims). Foldable consumers
// fold (reductions over an empty set give their identity); elementwise consumers that do not fold produce an empty result of the same shape.
internal sealed partial class OnnxTranslator
{
    // Ops whose output has the shape of their first input (besides the unary elementwise ops), with the output type when it is fixed.
    private static readonly Dictionary<string, int> s_shapePreserving = new()
    {
        ["Trilu"] = 0, ["Identity"] = 0, ["Softmax"] = 0, ["LogSoftmax"] = 0, ["Hardmax"] = 0, ["CumSum"] = 0, ["CumProd"] = 0,
        ["Clip"] = 0, ["Shrink"] = 0, ["Celu"] = 0, ["Elu"] = 0, ["Selu"] = 0, ["LeakyRelu"] = 0, ["ThresholdedRelu"] = 0,
        ["HardSigmoid"] = 0, ["Mish"] = 0, ["Gelu"] = 0, ["Not"] = OnnxType.Bool, ["IsNaN"] = OnnxType.Bool, ["IsInf"] = OnnxType.Bool,
        ["RegexFullMatch"] = OnnxType.Bool,
    };

    /// <summary>
    /// Translates a node of constant inputs whose first input is empty and whose output has its shape (elementwise and shape-preserving
    /// ops that do not fold) to an empty constant; returns false for every other node.
    /// </summary>
    private bool TryEmptyResult(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (ins.Length == 0 || ins[0] is not { C: { Count: 0 } empty } || n.Outputs.Count == 0 || n.Outputs[0] == "")
        {
            return false;
        }

        int elemType;
        if (n.OpType == "Cast")
        {
            elemType = (int)n.Int("to", OnnxType.Float);
        }
        else if (s_shapePreserving.TryGetValue(n.OpType, out int fixedType))
        {
            elemType = fixedType != 0 ? fixedType : empty.ElemType;
        }
        else if (s_unary.ContainsKey(n.OpType))
        {
            elemType = empty.ElemType;
        }
        else
        {
            return false;
        }

        _env[n.Outputs[0]] = TranslatedTensor.Const(EmptyConst(elemType, empty.Dims, n.Outputs[0]));
        return true;
    }
}
