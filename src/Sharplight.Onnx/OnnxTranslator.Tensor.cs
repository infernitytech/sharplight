using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Tensor operators without a better home: Det, EyeLike, TensorScatter.
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

    /// <summary>
    /// TensorScatter (KV-cache write): present[b, ..., s, ...] = update[b, ..., s - w[b], ...] where 0 ≤ s - w[b] &lt; S_update
    /// (s - w[b] taken modulo the cache length in circular mode), else past[b, ..., s, ...]. Composed from a GatherElements of
    /// the update at clamped relative positions and a Where.
    /// </summary>
    private void TensorScatter(OnnxNode n, TranslatedTensor?[] ins)
    {
        var cache = X(ins, 0);
        var update = X(ins, 1);
        int rank = cache.Rank;
        int axis = NormalizeAxis(n.Int("axis", -2), rank);
        if (axis == 0)
        {
            throw new NotSupportedException("TensorScatter along the batch axis");
        }

        long length = cache.Shape[axis].Fixed;
        long updateLength = update.Shape[axis].Fixed;
        string name = Nm(n);
        var gridDims = Enumerable.Repeat(1L, rank).ToArray();
        gridDims[axis] = length;
        var positions = ConstValue(OnnxConst.FromFloats(OnnxType.Float, gridDims, Enumerable.Range(0, (int)length).Select(s => (double)s).ToArray(), name + ".positions"), name + ".positions");
        var relative = positions;
        if (ins.Length > 2 && ins[2] is not null)
        {
            var writes = AsValue(ins[2]!);
            var start = writes.Type == DataType.F32 ? writes : Emit("Cast", [writes], Attributes.From(("type", "f32")));
            var batchDims = Enumerable.Repeat(1L, rank).ToArray();
            batchDims[0] = Sym(cache) ? 0 : cache.Shape[0].Fixed;
            relative = _b.Binary("Sub", positions, Rs(start, batchDims));
        }

        if (n.Str("mode", "linear") == "circular")
        {
            relative = Emit("Mod", [_b.Scale(relative, 1f, length), ConstValue(OnnxConst.Scalar(length), name + ".length")], Attributes.From(("fmod", true)));
        }

        var valid = _b.Mul(_b.Binary("GreaterEqual", relative, ConstValue(OnnxConst.Scalar(0f), name + ".zero")),
            _b.Binary("Less", relative, ConstValue(OnnxConst.Scalar(updateLength), name + ".update_length")));
        var fullDims = cache.Shape.Dims.Select(d => d.IsSymbolic ? 1L : d.Fixed).ToArray();
        fullDims[0] = 1;
        var spread = _b.Add(Emit("Clip", [relative], Attributes.From(("min", 0f), ("max", (float)(updateLength - 1)))),
            ConstValue(OnnxConst.FromFloats(OnnxType.Float, fullDims, new double[fullDims.Aggregate(1L, (a, b) => a * b)], name + ".spread"), name + ".spread"));
        if (Sym(cache) && !Sym(spread))
        {
            throw new NotSupportedException("TensorScatter without write_indices over a symbolic batch");
        }

        if (!Sym(cache) && cache.Shape[0].Fixed > 1 && spread.Shape[0].Fixed == 1)
        {
            spread = _b.Add(spread, ConstValue(OnnxConst.FromFloats(OnnxType.Float, [cache.Shape[0].Fixed, .. Enumerable.Repeat(1L, rank - 1)], new double[cache.Shape[0].Fixed], name + ".batch"), name + ".batch"));
        }

        var indices = Emit("Cast", [spread], Attributes.From(("type", "i32")));
        var gathered = Emit("GatherElements", [update, indices], Attributes.From(("axis", (long)axis)));
        SetDyn(n, 0, Emit("Where", [valid, gathered, cache], null, name));
    }

    /// <summary>Det: determinants of [..., M, M] (the IR Det op; CPU).</summary>
    private void Det(OnnxNode n, TranslatedTensor?[] ins)
    {
        SetDyn(n, 0, Emit("Det", [X(ins, 0)], null, Nm(n)));
    }
}
