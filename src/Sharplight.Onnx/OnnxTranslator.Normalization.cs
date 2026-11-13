using Sharplight.Graph;

namespace Sharplight.Onnx;

// Softmax and normalization operators.
internal sealed partial class OnnxTranslator
{
    // ================================ normalization / misc =================================================
    private void Softmax(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int rank = x.Rank;
        int axis = NormalizeAxis(n.Int("axis", Opset >= 13 ? -1 : 1), rank);
        Value y;
        bool lg = n.OpType == "LogSoftmax";
        Value Sm(Value v) => lg ? _b.Emit1("Softmax", [v], Attributes.From(("log", true))) : _b.Emit1("Softmax", [v]);
        if (axis == rank - 1)
        {
            y = Sm(x);
        }
        else if (Opset >= 13)
        {
            if (axis == 0 && Sym(x))
            {
                throw new NotSupportedException("Softmax over the symbolic batch axis");
            }

            var perm = Enumerable.Range(0, rank).ToArray();
            (perm[axis], perm[rank - 1]) = (perm[rank - 1], perm[axis]);
            y = Permute(Sm(Permute(x, perm)), perm);
        }
        else
        {
            // Pre-13: coerce to 2-D at 'axis' (softmax over all trailing dims).
            if (axis == 0 && Sym(x))
            {
                throw new NotSupportedException("Softmax over the symbolic batch axis");
            }

            bool sym = Sym(x);
            var dims = x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();
            long inner = 1;
            for (int i = axis; i < rank; i++)
            {
                inner *= dims[i];
            }

            long[] flat = [.. dims[..axis], inner];
            var s = Sm(Rs(x, flat));
            y = Rs(s, dims);
        }
        SetDyn(n, 0, y);
        _ = Nm(n);
    }

    private void LayerNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        CheckUnusedOutputs(n, 1);
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", -1), x.Rank);
        if (axis == 0 && Sym(x))
        {
            throw new NotSupportedException("LayerNormalization over the batch axis");
        }

        var sc = ConstIn(ins, 1, "LayerNormalization scale");
        var bi = OptConst(ins, 2, "LayerNormalization bias");
        long d = 1;
        for (int i = axis; i < x.Rank; i++)
        {
            d *= x.Shape[i].Fixed;
        }

        var xin = x;
        bool flatten = axis != x.Rank - 1;
        if (flatten)
        {
            long[] flat = [.. x.Shape.Dims.Take(axis).Select(q => q.IsSymbolic ? 0L : q.Fixed), d];
            xin = Rs(x, flat);
        }
        var ws = new List<Value> { xin, ConstValue(sc.Reshaped([d]), Nm(n) + ".scale") };
        if (bi is not null)
        {
            ws.Add(ConstValue(bi.Reshaped([d]), Nm(n) + ".bias"));
        }

        var y = Emit("LayerNorm", [.. ws], Attributes.From(("eps", n.Float("epsilon", 1e-5f))), flatten ? null : Nm(n));
        if (flatten)
        {
            y = Emit("Reshape", [y], Attributes.From(("shape", x.Shape.Dims.Select(q => q.IsSymbolic ? 0L : q.Fixed).ToArray())), Nm(n));
        }

        SetDyn(n, 0, y);
    }

    private void InstanceNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var sc = ConstIn(ins, 1, "InstanceNormalization scale");
        var bi = ConstIn(ins, 2, "InstanceNormalization bias");
        SetDyn(n, 0, Emit("GroupNorm", [x, ConstValue(sc, Nm(n) + ".scale"), ConstValue(bi, Nm(n) + ".bias")],
            Attributes.From(("groups", x.Shape[1].Fixed), ("eps", n.Float("epsilon", 1e-5f))), Nm(n)));
    }

    private void GroupNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long g = n.Int("num_groups", 1);
        long c = x.Shape[1].Fixed;
        var sc = ConstIn(ins, 1, "GroupNormalization scale");
        var bi = ConstIn(ins, 2, "GroupNormalization bias");
        if (sc.Count == g && g != c)
        {
            sc = Repeat(sc, c / g);
            bi = Repeat(bi, c / g);
        }
        SetDyn(n, 0, Emit("GroupNorm", [x, ConstValue(sc, Nm(n) + ".scale"), ConstValue(bi, Nm(n) + ".bias")], Attributes.From(("groups", g), ("eps", n.Float("epsilon", 1e-5f))), Nm(n)));
    }

    private static OnnxConst Repeat(OnnxConst c, long r)
    {
        var f = c.Floats();
        return OnnxConst.FromFloats(OnnxType.Float, [f.Length * r], f.SelectMany(v => Enumerable.Repeat(v, (int)r)).ToArray(), c.Name + ".rep");
    }

    private void RmsNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", -1), x.Rank);
        if (axis != x.Rank - 1)
        {
            throw new NotSupportedException("RMS normalization over more than the last axis");
        }

        CheckUnusedOutputs(n, 1);
        SetDyn(n, 0, Emit("RmsNorm", [x, ConstValue(ConstIn(ins, 1, "scale"), Nm(n) + ".scale")], Attributes.From(("eps", n.Float("epsilon", 1e-5f))), Nm(n)));
    }

    private void LocalResponseNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long size = n.Int("size", 1);
        float alpha = n.Float("alpha", 1e-4f);
        float beta = n.Float("beta", 0.75f);
        float bias = n.Float("bias", 1f);
        var sq = _b.Mul(x, x);
        long pb = (size - 1) / 2;
        long pe = size - 1 - pb;
        var pads = new long[2 * x.Rank];
        pads[1] = pb;
        pads[x.Rank + 1] = pe;
        var padded = Emit("Pad", [sq], Attributes.From(("pads", pads), ("mode", "constant"), ("value", 0f)));
        long c = x.Shape[1].Fixed;
        Value sum = _b.Slice(padded, 1, 0, c);
        for (long i = 1; i < size; i++)
        {
            sum = _b.Add(sum, _b.Slice(padded, 1, i, c));
        }

        var denom = _b.Binary("Pow", _b.Scale(sum, alpha / size, bias), ConstValue(OnnxConst.Scalar(beta), "lrn_beta"));
        SetDyn(n, 0, Emit("Div", [x, denom], null, Nm(n)));
    }

    private void LpNormalization(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long axis = NormalizeAxis(n.Int("axis", -1), x.Rank);
        long p = n.Int("p", 2);
        var norm = Emit("Reduce", [x], Attributes.From(("mode", p == 1 ? "l1" : "l2"), ("axes", L(axis)), ("keepdims", true)));
        SetDyn(n, 0, Emit("Div", [x, norm], null, Nm(n)));
    }

    private void MeanVarianceNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        var axes = n.Ints("axes") ?? [0, 2, 3];
        var mean = Emit("Reduce", [x], Attributes.From(("mode", "mean"), ("axes", axes), ("keepdims", true)));
        var centered = _b.Sub(x, mean);
        var sumSquares = Emit("Reduce", [centered], Attributes.From(("mode", "sumsquare"), ("axes", axes), ("keepdims", true)));
        long count = axes.Aggregate(1L, (product, axis) => product * x.Shape[NormalizeAxis(axis, x.Rank)].Fixed);
        var stdDev = _b.Unary("Sqrt", _b.Scale(sumSquares, 1f / count, 1e-9f));
        SetDyn(n, 0, Emit("Div", [centered, stdDev], null, Nm(n)));
    }
}
