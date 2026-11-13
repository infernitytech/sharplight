using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Tensor operators without a better home: Det, EyeLike.
internal sealed partial class OnnxTranslator
{
    /// <summary>EyeLike of a runtime tensor: only its (fixed) shape matters, so the result is the constant identity pattern.</summary>
    private void EyeLike(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = ins[0]!.V!;
        if (Sym(x))
        {
            throw new NotSupportedException("EyeLike of a tensor with a symbolic dimension");
        }

        int elemType = x.Type == DataType.I32 ? OnnxType.Int32 : OnnxType.Float;
        var shapeOnly = OnnxConst.FromInts(elemType, Fixed(x), new long[x.Shape.Elements()]);
        _env[n.Outputs[0]] = TranslatedTensor.Const(OnnxFold.Eval(n, [shapeOnly])[0]);
    }

    /// <summary>Det: determinants of [..., M, M] (the IR Det op; CPU).</summary>
    private void Det(OnnxNode n, TranslatedTensor?[] ins)
    {
        SetDyn(n, 0, Emit("Det", [X(ins, 0)], null, Nm(n)));
    }
}
