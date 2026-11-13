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

    /// <summary>
    /// LayerNormalization over the axes from <c>axis</c> on (flattened into one row for the IR LayerNorm), with constant or runtime scale and
    /// bias of the normalized shape (or broadcast to it), and the optional Mean / InvStdDev outputs (keepdims shape).
    /// </summary>
    private void LayerNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", -1), x.Rank);
        float eps = n.Float("epsilon", 1e-5f);
        var (rows, d) = NormalizedRows(x, axis, "LayerNormalization");
        var scale = RowParameter(ins, 1, d, Nm(n) + ".scale");
        var bias = RowParameter(ins, 2, d, Nm(n) + ".bias");
        var args = new List<Value> { rows };
        if (scale.Row is not null)
        {
            args.Add(scale.Row);
            if (bias.Row is not null)
            {
                args.Add(bias.Row);
            }
        }

        bool plain = rows == x && scale.Broadcast is null && bias.Broadcast is null && (bias.Row is null || scale.Row is not null);
        var y = Emit("LayerNorm", [.. args], Attributes.From(("eps", eps)), plain ? Nm(n) : null);
        if (!plain)
        {
            y = rows == x ? y : Rs(y, Fixed(x));
            if (scale.Broadcast is not null)
            {
                y = _b.Mul(y, scale.Broadcast);
            }

            if (bias.Broadcast is not null || (bias.Row is not null && scale.Row is null))
            {
                y = _b.Add(y, bias.Broadcast ?? bias.Row!);
            }
        }

        SetDyn(n, 0, y);
        bool wantsMean = n.Outputs.Count > 1 && n.Outputs[1] != "";
        bool wantsInvStdDev = n.Outputs.Count > 2 && n.Outputs[2] != "";
        if (!wantsMean && !wantsInvStdDev)
        {
            return;
        }

        long[] statShape = [.. Fixed(x).Take(axis), .. Enumerable.Repeat(1L, x.Rank - axis)];
        var mean = Emit("Reduce", [rows], Attributes.From(("mode", "mean"), ("axes", L(-1)), ("keepdims", true)));
        if (wantsMean)
        {
            SetDyn(n, 1, Rs(mean, statShape));
        }

        if (wantsInvStdDev)
        {
            var sumSquares = Emit("Reduce", [_b.Sub(rows, mean)], Attributes.From(("mode", "sumsquare"), ("axes", L(-1)), ("keepdims", true)));
            SetDyn(n, 2, Rs(_b.Unary("Rsqrt", _b.Scale(sumSquares, 1f / d, eps)), statShape));
        }
    }

    /// <summary>x viewed as rows over the normalized axes [axis, rank): [outer dims..., D].</summary>
    private (Value Rows, long D) NormalizedRows(Value x, int axis, string op)
    {
        if (axis == 0 && Sym(x))
        {
            throw new NotSupportedException($"{op} over the batch axis");
        }

        long d = 1;
        for (int i = axis; i < x.Rank; i++)
        {
            d *= x.Shape[i].Fixed;
        }

        // A single row keeps a leading unit axis: providers treat the outer axis of a row op as the row count.
        return (axis == x.Rank - 1 ? x : Rs(x, axis == 0 ? [1, d] : [.. Fixed(x).Take(axis), d]), d);
    }

    /// <summary>
    /// A per-element normalization parameter (scale or bias): a [D] row when it covers the normalized shape, otherwise the value to
    /// broadcast against the output (a parameter of fewer elements, broadcast numpy-style); both null when absent.
    /// </summary>
    private (Value? Row, Value? Broadcast) RowParameter(TranslatedTensor?[] ins, int index, long d, string name)
    {
        if (Opt(ins, index) is not { } t)
        {
            return (null, null);
        }

        if (t.C is { } c)
        {
            return c.Count == d ? (ConstValue(c.Reshaped([d]), name), null) : (null, ConstValue(c, name));
        }

        var v = AsValue(t);
        return !Sym(v) && v.Shape.Elements() == d ? (Rs(v, d), null) : (null, v);
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

    /// <summary>RMSNormalization over the axes from <c>axis</c> on, with a constant or runtime scale.</summary>
    private void RmsNorm(OnnxNode n, TranslatedTensor?[] ins)
    {
        CheckUnusedOutputs(n, 1);
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", -1), x.Rank);
        var (rows, d) = NormalizedRows(x, axis, "RMSNormalization");
        var scale = RowParameter(ins, 1, d, Nm(n) + ".scale");
        var attributes = Attributes.From(("eps", n.Float("epsilon", 1e-5f)));
        bool plain = rows == x && scale.Broadcast is null;
        var y = Emit("RmsNorm", scale.Row is null ? [rows] : [rows, scale.Row], attributes, plain ? Nm(n) : null);
        y = rows == x ? y : Rs(y, Fixed(x));
        if (scale.Broadcast is not null)
        {
            y = _b.Mul(y, scale.Broadcast);
        }

        SetDyn(n, 0, y);
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
        // A zero norm means a zero slice: divide by 1 there so the result is 0 instead of NaN.
        var one = ConstValue(OnnxConst.Scalar(1f), Nm(n) + ".one");
        var safeNorm = Emit("Where", [_b.Binary("Equal", norm, ConstValue(OnnxConst.Scalar(0f), Nm(n) + ".zero")), one, norm]);
        SetDyn(n, 0, Emit("Div", [x, safeNorm], null, Nm(n)));
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
