using System.Buffers.Binary;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Broader operator coverage: indexing, normalization variants, einsum, quantization, recurrent nets, contrib ops.
internal sealed partial class OnnxTranslator
{
    private Value Idx(Tn t)
    {
        if (t.V is { } v)
        {
            return v.Type == DataType.I32 ? v : _b.Emit1("Cast", [v], Attributes.From(("type", "i32")));
        }

        var c = t.C!;
        var dims = c.Rank == 0 ? [1L] : c.Dims;
        var ints = c.Ints();
        var bytes = new byte[ints.Length * 4];
        for (int i = 0; i < ints.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), (int)ints[i]);
        }

        return _b.Constant(c.Name != "" ? c.Name : "idx", DataType.I32, new Shape(dims.Select(d => (Dim)d).ToArray()), bytes);
    }

    private static long[] Fixed(Value x) => x.Shape.Dims.Select(d => d.IsSymbolic ? 0L : d.Fixed).ToArray();

    private Value Act(string? name, Value v, float[]? p)
    {
        switch (name)
        {
            case null: case "": case "Identity": return v;
            case "Relu": return _b.Unary("Relu", v);
            case "Sigmoid": return _b.Unary("Sigmoid", v);
            case "Tanh": return _b.Unary("Tanh", v);
            case "LeakyRelu": return Emit("LeakyRelu", [v], Attributes.From(("alpha", p is { Length: > 0 } ? p[0] : 0.01f)));
            case "Clip": return Emit("Clip", [v], Attributes.From(("min", p![0]), ("max", p[1])));
            case "HardSigmoid": return Emit("HardSigmoid", [v], Attributes.From(("alpha", p is { Length: > 0 } ? p[0] : 0.2f), ("beta", p is { Length: > 1 } ? p[1] : 0.5f)));
            default: throw new NotSupportedException($"activation '{name}'");
        }
    }

    private bool More(OnnxNode n, Tn?[] ins)
    {
        if (QuantExtra(n, ins))
        {
            return true;
        }

        if (n.Domain == "com.microsoft") { Contrib(n, ins); return true; }
        string name = Nm(n);
        switch (n.OpType)
        {
            case "Where": SetDyn(n, 0, Emit("Where", [X(ins, 0), X(ins, 1), X(ins, 2)], null, name)); return true;
            case "ArgMax":
            case "ArgMin":
                {
                    var x = X(ins, 0);
                    int axis = (int)n.Int("axis", 0); if (axis < 0)
                    {
                        axis += x.Rank;
                    }

                    SetDyn(n, 0, Emit("ArgReduce", [x], Attributes.From(("mode", n.OpType == "ArgMax" ? "max" : "min"), ("axis", (long)axis),
                    ("keepdims", n.Int("keepdims", 1) != 0), ("select_last_index", n.Int("select_last_index", 0) != 0)), name));
                    return true;
                }
            case "TopK":
                {
                    var x = X(ins, 0);
                    long k = n.Has("k") ? n.Int("k", 1) : ConstIn(ins, 1, "TopK k").ScalarInt();
                    var outs = _b.Emit("TopK", [x], Attributes.From(("k", k), ("axis", n.Int("axis", -1)), ("largest", n.Int("largest", 1) != 0), ("sorted", n.Int("sorted", 1) != 0)), name);
                    SetDyn(n, 0, outs[0]); if (n.Outputs.Count > 1 && n.Outputs[1] != "")
                    {
                        SetDyn(n, 1, outs[1]);
                    }

                    return true;
                }
            case "Hardmax": Hardmax(n, ins); return true;
            case "GroupNormalization": GroupNorm(n, ins); return true;
            case "LpNormalization":
                {
                    var x = X(ins, 0); long axis = n.Int("axis", -1); if (axis < 0)
                    {
                        axis += x.Rank;
                    }

                    long p = n.Int("p", 2);
                    var norm = Emit("Reduce", [x], Attributes.From(("mode", p == 1 ? "l1" : "l2"), ("axes", L(axis)), ("keepdims", true)));
                    SetDyn(n, 0, Emit("Div", [x, norm], null, name));
                    return true;
                }
            case "RMSNormalization": case "SimplifiedLayerNormalization": RmsNorm(n, ins); return true;
            case "LRN": LRN(n, ins); return true;
            case "MeanVarianceNormalization":
                {
                    var x = X(ins, 0);
                    var axes = n.Ints("axes") ?? [0, 2, 3];
                    var mean = Emit("Reduce", [x], Attributes.From(("mode", "mean"), ("axes", axes), ("keepdims", true)));
                    var xc = _b.Sub(x, mean);
                    var v = Emit("Reduce", [xc], Attributes.From(("mode", "sumsquare"), ("axes", axes), ("keepdims", true)));
                    long cnt = axes.Aggregate(1L, (a, ax) => a * x.Shape[(int)(ax < 0 ? ax + x.Rank : ax)].Fixed);
                    var sd = _b.Unary("Sqrt", _b.Scale(v, 1f / cnt, 1e-9f));
                    SetDyn(n, 0, Emit("Div", [xc, sd], null, name));
                    return true;
                }
            case "ConvTranspose": ConvTranspose(n, ins); return true;
            case "LpPool": LpPool(n, ins); return true;
            case "Einsum": Einsum(n, ins); return true;
            case "Tile":
                SetDyn(n, 0, Emit("Tile", [X(ins, 0)], Attributes.From(("repeats", ConstIn(ins, 1, "Tile repeats").Ints())), name)); return true;
            case "GatherElements":
                SetDyn(n, 0, Emit("GatherElements", [X(ins, 0), Idx(ins[1]!)], Attributes.From(("axis", n.Int("axis", 0))), name)); return true;
            case "GatherND":
                SetDyn(n, 0, Emit("GatherND", [X(ins, 0), Idx(ins[1]!)], Attributes.From(("batch_dims", n.Int("batch_dims", 0))), name)); return true;
            case "ScatterElements":
            case "Scatter":
                SetDyn(n, 0, Emit("ScatterElements", [X(ins, 0), Idx(ins[1]!), X(ins, 2)], Attributes.From(("axis", n.Int("axis", 0)), ("reduction", n.Str("reduction", "none"))), name)); return true;
            case "ScatterND":
                SetDyn(n, 0, Emit("ScatterND", [X(ins, 0), Idx(ins[1]!), X(ins, 2)], Attributes.From(("reduction", n.Str("reduction", "none"))), name)); return true;
            case "CumSum":
                SetDyn(n, 0, Emit("CumSum", [X(ins, 0)], Attributes.From(("axis", ConstIn(ins, 1, "CumSum axis").ScalarInt()), ("exclusive", n.Int("exclusive", 0) != 0), ("reverse", n.Int("reverse", 0) != 0)), name)); return true;
            case "DepthToSpace": case "SpaceToDepth": DepthSpace(n, ins); return true;
            case "Trilu":
                {
                    long k = ins.Length > 1 && ins[1] is not null ? ConstIn(ins, 1, "Trilu k").ScalarInt() : 0;
                    SetDyn(n, 0, Emit("Trilu", [X(ins, 0)], Attributes.From(("upper", n.Int("upper", 1) != 0), ("k", k)), name)); return true;
                }
            case "OneHot":
                {
                    long depth = ConstIn(ins, 1, "OneHot depth").ScalarInt();
                    var vals = ConstIn(ins, 2, "OneHot values").Floats();
                    SetDyn(n, 0, Emit("OneHot", [Idx(ins[0]!)], Attributes.From(("depth", depth), ("axis", n.Int("axis", -1)), ("on", (float)vals[1]), ("off", (float)vals[0])), name)); return true;
                }
            case "GridSample":
                {
                    string mode = n.Str("mode", "bilinear"); mode = mode == "bilinear" ? "linear" : mode == "bicubic" ? "cubic" : mode;
                    SetDyn(n, 0, Emit("GridSample", [X(ins, 0), X(ins, 1)], Attributes.From(("mode", mode), ("padding", n.Str("padding_mode", "zeros")), ("align_corners", n.Int("align_corners", 0) != 0)), name)); return true;
                }
            case "CastLike":
                {
                    var like = ins[1]!;
                    int to = like.IsConst ? like.C!.ElemType : OnnxType.Float;
                    var fake = new OnnxNode { OpType = "Cast", Inputs = n.Inputs.Take(1).ToList(), Outputs = n.Outputs, Attributes = new Dictionary<string, OnnxAttribute> { ["to"] = new() { Name = "to", Type = OnnxAttrType.Int, I = to } } };
                    CastDyn(fake, ins);
                    return true;
                }
            case "QuantizeLinear": case "DequantizeLinear": QuantDyn(n, ins); return true;
            case "MatMulInteger": case "ConvInteger": case "QLinearMatMul": case "QLinearConv": QuantOps(n, ins); return true;
            case "If": IfOp(n, ins); return true;
            case "LSTM": case "GRU": case "RNN": Rnn(n, ins); return true;
            case "NonZero":
            case "EyeLike":
                throw new NotSupportedException($"{n.OpType} of a runtime-dependent tensor has a data-dependent shape");
        }
        return false;
    }

    private void CastDyn(OnnxNode n, Tn?[] ins)
    {
        int to = (int)n.Int("to", OnnxType.Float);
        var x = ins[0]!;
        Value v = x.V ?? AsValue(x);
        if (to == OnnxType.Bool)
        {
            v = Emit("NotEqual", [AsValue(x), _b.Constant("zero", DataType.F32, new Shape(1), new byte[4])], null, Nm(n));
        }
        else if (OnnxType.IsFloat(to)) { v = x.V is { Type.Kind: DataTypeKind.Int } ? AsValue(x) : v; }
        else
        {
            if (v.Type == DataType.F32)
            {
                v = Emit("Cast", [v], Attributes.From(("type", "i32")), Nm(n));
            }
        }
        _env[n.Outputs[0]] = Tn.Dyn(v);
    }

    // ---- normalization ---------------------------------------------------------------------------------
    private void Hardmax(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        int axis = (int)n.Int("axis", Opset >= 13 ? -1 : 1); if (axis < 0)
        {
            axis += x.Rank;
        }

        var perm = Enumerable.Range(0, x.Rank).ToArray(); (perm[axis], perm[^1]) = (perm[^1], perm[axis]);
        var xp = Permute(x, perm);
        var idx = Emit("ArgReduce", [xp], Attributes.From(("mode", "max"), ("axis", (long)(x.Rank - 1)), ("keepdims", false)));
        var oh = Emit("OneHot", [idx], Attributes.From(("depth", xp.Shape[-1].Fixed), ("axis", -1L), ("on", 1f), ("off", 0f)));
        SetDyn(n, 0, Permute(oh, perm));
    }

    private void GroupNorm(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        long g = n.Int("num_groups", 1), c = x.Shape[1].Fixed;
        var sc = ConstIn(ins, 1, "GroupNormalization scale"); var bi = ConstIn(ins, 2, "GroupNormalization bias");
        if (sc.Count == g && g != c) { sc = Repeat(sc, c / g); bi = Repeat(bi, c / g); }
        SetDyn(n, 0, Emit("GroupNorm", [x, ConstValue(sc, Nm(n) + ".scale"), ConstValue(bi, Nm(n) + ".bias")], Attributes.From(("groups", g), ("eps", n.Float("epsilon", 1e-5f))), Nm(n)));
    }

    private static OnnxConst Repeat(OnnxConst c, long r)
    {
        var f = c.Floats(); return OnnxConst.FromFloats(OnnxType.Float, [f.Length * r], f.SelectMany(v => Enumerable.Repeat(v, (int)r)).ToArray(), c.Name + ".rep");
    }

    private void RmsNorm(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        int axis = (int)n.Int("axis", -1); if (axis < 0)
        {
            axis += x.Rank;
        }

        if (axis != x.Rank - 1)
        {
            throw new NotSupportedException("RMS normalization over more than the last axis");
        }

        CheckUnusedOutputs(n, 1);
        SetDyn(n, 0, Emit("RmsNorm", [x, ConstValue(ConstIn(ins, 1, "scale"), Nm(n) + ".scale")], Attributes.From(("eps", n.Float("epsilon", 1e-5f))), Nm(n)));
    }

    private void LRN(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        long size = n.Int("size", 1); float alpha = n.Float("alpha", 1e-4f), beta = n.Float("beta", 0.75f), bias = n.Float("bias", 1f);
        var sq = _b.Mul(x, x);
        long pb = (size - 1) / 2, pe = size - 1 - pb;
        var pads = new long[2 * x.Rank]; pads[1] = pb; pads[x.Rank + 1] = pe;
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

    private void LpPool(OnnxNode n, Tn?[] ins)
    {
        long p = n.Int("p", 2);
        if (p is not (1 or 2))
        {
            throw new NotSupportedException("LpPool with p other than 1 or 2");
        }

        var x = X(ins, 0);
        var kernel = n.Ints("kernel_shape")!;
        var strides = n.Ints("strides") ?? [1, 1]; var pads = n.Ints("pads") ?? [0, 0, 0, 0];
        var inp = p == 1 ? _b.Unary("Abs", x) : _b.Mul(x, x);
        var avg = Emit("Pool", [inp], Attributes.From(("mode", "avg"), ("kernel", kernel), ("strides", strides), ("pads", pads), ("dilations", L(1, 1)), ("ceil_mode", n.Int("ceil_mode", 0) != 0), ("count_include_pad", true)));
        var s = _b.Scale(avg, kernel[0] * kernel[1]);
        SetDyn(n, 0, p == 1 ? s : Emit("Sqrt", [s], null, Nm(n)));
    }

    // ---- conv transpose ----------------------------------------------------------------------------------
    private void ConvTranspose(OnnxNode n, Tn?[] ins)
    {
        var x0 = X(ins, 0);
        var wC = ConstIn(ins, 1, "ConvTranspose weight");
        var bC = ins.Length > 2 && ins[2] is not null ? ConstIn(ins, 2, "ConvTranspose bias") : null;
        bool oneD = x0.Rank == 3;
        var x = To4D(x0);
        long[] wd = oneD ? [wC.Dims[0], wC.Dims[1], 1, wC.Dims[2]] : wC.Dims;
        long[] Ext(long[]? v, long def) => v is null ? [def, def] : oneD ? [1, v[0]] : v;
        var strides = Ext(n.Ints("strides"), 1); var dil = Ext(n.Ints("dilations"), 1); var outPad = Ext(n.Ints("output_padding"), 0);
        long groups = n.Int("group", 1);
        long[] pads = n.Ints("pads") is { } p ? (oneD ? [0, p[0], 0, p[1]] : p) : [0, 0, 0, 0];
        string ap = n.Str("auto_pad", "NOTSET");
        long[]? os = n.Ints("output_shape");
        if (os is not null || ap is "SAME_UPPER" or "SAME_LOWER")
        {
            long[] want = os is not null ? (oneD ? [1, os[0]] : os) : [x.Shape[2].Fixed * strides[0], x.Shape[3].Fixed * strides[1]];
            pads = new long[4];
            for (int i = 0; i < 2; i++)
            {
                long total = strides[i] * (x.Shape[2 + i].Fixed - 1) + outPad[i] + (wd[2 + i] - 1) * dil[i] + 1 - want[i];
                bool upper = ap == "SAME_UPPER" || (os is not null && ap != "SAME_LOWER");
                long small = total / 2, big = total - small;
                pads[i] = upper ? small : big; pads[i + 2] = upper ? big : small;
            }
        }
        var w = WeightValue(wC.Reshaped(wd), false, Nm(n) + ".weight");
        var inputs = new List<Value> { x, w };
        if (bC is not null)
        {
            inputs.Add(ConstValue(bC, Nm(n) + ".bias"));
        }

        var y = Emit("ConvTranspose", [.. inputs], Attributes.From(("strides", strides), ("pads", pads), ("dilations", dil), ("groups", groups), ("output_padding", outPad)), oneD ? null : Nm(n));
        Heavy();
        if (oneD)
        {
            y = Emit("Reshape", [y], Attributes.From(("shape", L(0, y.Shape[1].Fixed, y.Shape[3].Fixed))), Nm(n));
        }

        SetDyn(n, 0, y);
    }

    private void DepthSpace(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        long b = n.Int("blocksize", 1);
        long c = x.Shape[1].Fixed, h = x.Shape[2].Fixed, w = x.Shape[3].Fixed;
        Value y;
        if (n.OpType == "DepthToSpace")
        {
            bool crd = n.Str("mode", "DCR") == "CRD";
            long oc = c / (b * b);
            y = crd ? Rs(x, 0, oc, b, b, h, w) : Rs(x, 0, b, b, oc, h, w);
            y = _b.Permute(y, crd ? [0, 1, 4, 2, 5, 3] : [0, 3, 4, 1, 5, 2]);
            y = Emit("Reshape", [y], Attributes.From(("shape", L(0, oc, h * b, w * b))), Nm(n));
        }
        else
        {
            y = Rs(x, 0, c, h / b, b, w / b, b);
            y = _b.Permute(y, [0, 3, 5, 1, 2, 4]);
            y = Emit("Reshape", [y], Attributes.From(("shape", L(0, c * b * b, h / b, w / b))), Nm(n));
        }
        SetDyn(n, 0, y);
    }

    // ---- Einsum --------------------------------------------------------------------------------------------
    private sealed class Lt(Value v, List<char> labels) { public Value V = v; public List<char> L = labels; }

    private void Einsum(OnnxNode n, Tn?[] ins)
    {
        string eq = n.Str("equation", "").Replace(" ", "");
        if (eq.Contains("..."))
        {
            throw new NotSupportedException("Einsum with ellipsis");
        }

        var parts = eq.Split("->");
        var terms = parts[0].Split(',');
        string rhs = parts.Length > 1 ? parts[1] : new string(string.Concat(terms).GroupBy(c => c).Where(g => g.Count() == 1).Select(g => g.Key).OrderBy(c => c).ToArray());
        if (terms.Length is < 1 or > 2)
        {
            throw new NotSupportedException("Einsum with more than two operands");
        }

        var ops = terms.Select((t, i) => new Lt(X(ins, i), [.. t])).ToList();
        foreach (var o in ops)
        {
            if (o.L.Count != o.V.Rank || o.L.Distinct().Count() != o.L.Count)
            {
                throw new NotSupportedException("Einsum with repeated labels (traces/diagonals)");
            }
        }

        var outL = rhs.ToList();
        // sum out labels that appear in only one operand and not in the output
        for (int i = 0; i < ops.Count; i++)
        {
            var other = ops.Count == 2 ? ops[1 - i].L : [];
            var drop = ops[i].L.Where(c => !outL.Contains(c) && !other.Contains(c)).ToList();
            if (drop.Count > 0)
            {
                var axes = drop.Select(c => (long)ops[i].L.IndexOf(c)).ToArray();
                ops[i] = new Lt(Emit("Reduce", [ops[i].V], Attributes.From(("mode", "sum"), ("axes", axes), ("keepdims", false))), ops[i].L.Where(c => !drop.Contains(c)).ToList());
            }
        }
        Lt result;
        if (ops.Count == 1)
        {
            result = ops[0];
        }
        else
        {
            var a = ops[0]; var b = ops[1];
            var batch = a.L.Where(c => b.L.Contains(c) && outL.Contains(c)).ToList();
            var contr = a.L.Where(c => b.L.Contains(c) && !outL.Contains(c)).ToList();
            var aOnly = a.L.Where(c => !b.L.Contains(c)).ToList();
            var bOnly = b.L.Where(c => !a.L.Contains(c)).ToList();
            Value ap = PermuteLabels(a, [.. batch, .. aOnly, .. contr]);
            Value bp = PermuteLabels(b, [.. batch, .. contr, .. bOnly]);
            var dimOf = new Dictionary<char, Dim>();
            foreach (var o in new[] { a, b })
            {
                for (int i = 0; i < o.L.Count; i++)
                {
                    dimOf[o.L[i]] = o.V.Shape[i];
                }
            }
            // A keeps its free dims separate (so a symbolic batch can stay in front); B gets matching size-1 dims to broadcast over them.
            var aGroups = new List<List<char>>(batch.Select(c => new List<char> { c }));
            if (aOnly.Count == 0)
            {
                aGroups.Add([]);
            }
            else
            {
                aGroups.AddRange(aOnly.Select(c => new List<char> { c }));
            }

            aGroups.Add(contr);
            var bGroups = new List<List<char>>(batch.Select(c => new List<char> { c }));
            for (int i = 1; i < aOnly.Count; i++)
            {
                bGroups.Add([]);
            }

            bGroups.Add(contr); bGroups.Add(bOnly);
            var ap2 = ReshapeGroups(ap, aGroups, dimOf);
            var bp2 = ReshapeGroups(bp, bGroups, dimOf);
            var y = Emit("BatchMatMul", [ap2, bp2]);
            Heavy();
            var yl = new List<List<char>>([.. batch.Select(c => new List<char> { c }), .. aOnly.Select(c => new List<char> { c }), .. bOnly.Select(c => new List<char> { c })]);
            result = new Lt(ReshapeToLabels(y, [.. batch, .. aOnly, .. bOnly], dimOf), [.. batch, .. aOnly, .. bOnly]);
        }
        var final = PermuteLabels(result, outL);
        _env[n.Outputs[0]] = Tn.Dyn(ReferenceEquals(final, result.V) ? Emit("Reshape", [final], Attributes.From(("shape", Fixed(final))), Nm(n)) : final);
    }

    private Value PermuteLabels(Lt t, List<char> order)
    {
        var perm = order.Select(c => t.L.IndexOf(c)).ToArray();
        return Permute(t.V, perm);
    }

    private Value ReshapeGroups(Value x, List<List<char>> groups, Dictionary<char, Dim> dimOf)
    {
        var shape = new List<long>();
        for (int gi = 0; gi < groups.Count; gi++)
        {
            long prod = 1; bool sym = false;
            foreach (var c in groups[gi])
            {
                if (dimOf[c].IsSymbolic)
                {
                    sym = true;
                }
                else
                {
                    prod *= dimOf[c].Fixed;
                }
            }
            if (sym)
            {
                if (shape.Count != 0 || groups[gi].Count != 1)
                {
                    throw new NotSupportedException("Einsum: the batch dimension would be merged or moved");
                }

                shape.Add(0);
            }
            else
            {
                shape.Add(prod);
            }
        }
        return Emit("Reshape", [x], Attributes.From(("shape", shape.ToArray())));
    }

    private Value ReshapeToLabels(Value y, List<char> labels, Dictionary<char, Dim> dimOf)
    {
        var shape = new List<long>();
        for (int i = 0; i < labels.Count; i++)
        {
            if (dimOf[labels[i]].IsSymbolic) { if (i != 0) { throw new NotSupportedException("Einsum: batch dimension in a non-leading position"); } shape.Add(0); }
            else
            {
                shape.Add(dimOf[labels[i]].Fixed);
            }
        }
        return Emit("Reshape", [y], Attributes.From(("shape", shape.ToArray())));
    }

    // ---- quantization ----------------------------------------------------------------------------------------
    /// <summary>Per-axis constant (scale/zero point) shaped for broadcasting against a tensor of rank <paramref name="rank"/>.</summary>
    private OnnxConst Bcast(OnnxConst c, int rank, int axis)
    {
        if (c.Count == 1)
        {
            return c.Reshaped([1]);
        }

        var d = Enumerable.Repeat(1L, rank).ToArray(); d[axis < 0 ? axis + rank : axis] = c.Count;
        return c.Reshaped(d);
    }

    private void QuantDyn(OnnxNode n, Tn?[] ins)
    {
        var x = X(ins, 0);
        var sc = ConstIn(ins, 1, "scale");
        var zp = OptConst(ins, 2, "zero point");
        int axis = (int)n.Int("axis", 1);
        var scV = ConstValue(Bcast(sc.CastTo(OnnxType.Float), x.Rank, axis), Nm(n) + ".scale");
        Value zpV = zp is null ? null! : ConstValue(Bcast(zp.CastTo(OnnxType.Float), x.Rank, axis), Nm(n) + ".zp");
        if (n.OpType == "DequantizeLinear")
        {
            var v = zp is null ? x : _b.Sub(x, zpV);
            SetDyn(n, 0, Emit("Mul", [v, scV], null, Nm(n)));
            return;
        }
        int et = zp?.ElemType ?? OnnxType.UInt8;
        var q = Emit("Round", [_b.Binary("Div", x, scV)]);
        if (zp is not null)
        {
            q = _b.Add(q, zpV);
        }

        var (lo, hi) = OnnxFold.Range(et);
        SetDyn(n, 0, Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), Nm(n)));
    }

    /// <summary>Dequantizes a constant (per-tensor/per-axis zero point and scale) to f32.</summary>
    private static OnnxConst DequantConst(OnnxConst x, OnnxConst scale, OnnxConst? zp, int axis = 0) => OnnxFold.Dequantize(x, scale, zp, axis);

    private Value ShiftZp(Tn a, Tn? zpT, string nm)
    {
        if (zpT is { IsConst: false })
        {
            return _b.Sub(AsValue(a), zpT.V!);   // runtime zero point (DynamicQuantizeLinear)
        }

        return ShiftZp(a, zpT?.C, nm);
    }

    private Value ShiftZp(Tn a, OnnxConst? zp, string nm)
    {
        var v = AsValue(a);
        if (zp is null || zp.Floats().All(z => z == 0))
        {
            return v;
        }

        return _b.Sub(v, ConstValue(zp.Count == 1 ? zp.CastTo(OnnxType.Float).Reshaped([1]) : throw new NotSupportedException("per-row activation zero points"), nm + ".azp"));
    }

    private void QuantOps(OnnxNode n, Tn?[] ins)
    {
        string name = Nm(n);
        switch (n.OpType)
        {
            case "MatMulInteger":
                {
                    var a = ShiftZp(ins[0]!, Opt(ins, 2), name);
                    var bC = ConstIn(ins, 1, "MatMulInteger B");
                    var bz = OptConst(ins, 3, "b zero point");
                    var bf = DequantConst(bC, OnnxConst.Scalar(1f), bz, bC.Rank - 1);   // per-column zero points when bz is a vector
                    SetDyn(n, 0, MatMulWeightOrBatch(a, bf, name));
                    break;
                }
            case "QLinearMatMul":
                {
                    var aScale = ConstIn(ins, 1, "a_scale").ScalarF32();
                    var a = _b.Scale(ShiftZp(ins[0]!, OptConst(ins, 2, "a_zp"), name), aScale);
                    var bf = DequantConst(ConstIn(ins, 3, "B"), ConstIn(ins, 4, "b_scale"), OptConst(ins, 5, "b_zp"), 1);
                    var y = MatMulWeightOrBatch(a, bf, name + ".mm");
                    SetDyn(n, 0, Requant(y, ConstIn(ins, 6, "y_scale").ScalarF32(), OptConst(ins, 7, "y_zp"), name));
                    break;
                }
            case "ConvInteger":
            case "QLinearConv":
                {
                    bool q = n.OpType == "QLinearConv";
                    Tn xin = ins[0]!;
                    float aScale = q ? ConstIn(ins, 1, "x_scale").ScalarF32() : 1f;
                    var xzp = q ? OptConst(ins, 2, "x_zp") : null;
                    var xv = ShiftZp(xin, q ? xzp : null, name);
                    if (!q)
                    {
                        xv = ShiftZp(xin, Opt(ins, 2), name);
                    }

                    if (q)
                    {
                        xv = _b.Scale(xv, aScale);
                    }

                    var wC = ConstIn(ins, q ? 3 : 1, "W");
                    var wscale = q ? ConstIn(ins, 4, "w_scale") : OnnxConst.Scalar(1f);
                    var wzp = OptConst(ins, q ? 5 : 3, "w_zp");
                    var wf = DequantConst(wC, wscale, wzp, 0);
                    OnnxConst? bias = null;
                    if (q && ins.Length > 8 && ins[8] is not null)
                    {
                        bias = OnnxConst.FromFloats(OnnxType.Float, [wf.Dims[0]], ConstIn(ins, 8, "B").Floats().Select((v, i) => v * aScale * wscale.Floats()[wscale.Count == 1 ? 0 : i]).ToArray());
                    }

                    var tmpIns = new Tn?[] { Tn.Dyn(xv), Tn.Const(wf.Reshaped(wf.Dims, name + ".w")), bias is null ? null : Tn.Const(bias) };
                    var conv = new OnnxNode { OpType = "Conv", Name = n.Name, Inputs = ["x", "w", bias is null ? "" : "b"], Outputs = [name + ".conv"], Attributes = n.Attributes };
                    Conv(conv, tmpIns);
                    var y = _env[name + ".conv"].V!;
                    SetDyn(n, 0, q ? Requant(y, ConstIn(ins, 6, "y_scale").ScalarF32(), OptConst(ins, 7, "y_zp"), name) : y);
                    break;
                }
        }
    }

    private Value MatMulWeightOrBatch(Value a, OnnxConst bf, string name)
    {
        Value y;
        if (bf.Rank == 2 && a.Rank >= 2)
        {
            y = Emit("MatMul", [a, WeightValue(bf.Transposed([1, 0]), false, name + ".weight")], null, name);
        }
        else
        {
            y = Emit("BatchMatMul", [a, ConstValue(bf, name + ".b")], null, name);
        }

        Heavy();
        return y;
    }

    private Value Requant(Value y, float yScale, OnnxConst? yzp, string name)
    {
        var q = Emit("Round", [_b.Scale(y, 1f / yScale)]);
        int et = yzp?.ElemType ?? OnnxType.UInt8;
        if (yzp is not null)
        {
            q = _b.Scale(q, 1f, yzp.ScalarF32());
        }

        var (lo, hi) = OnnxFold.Range(et);
        return Emit("Clip", [q], Attributes.From(("min", (float)lo), ("max", (float)hi)), name);
    }

    // ---- If --------------------------------------------------------------------------------------------------------
    private void IfOp(OnnxNode n, Tn?[] ins)
    {
        var cond = ConstIn(ins, 0, "If condition");
        var g = (cond.Floats()[0] != 0 ? n.Attributes["then_branch"] : n.Attributes["else_branch"]).G!;
        foreach (var t in g.Initializers)
        {
            if (t.Source is not null)
            {
                _env[t.Name] = Tn.Const(OnnxConst.FromTensor(t));
            }
        }

        foreach (var sub in g.Nodes)
        {
            if (!IsSupportedNode(sub))
            {
                throw new NotSupportedException($"If branch uses unsupported operator {sub.OpType}");
            }

            HandleNode(sub);
        }
        for (int i = 0; i < n.Outputs.Count; i++)
        {
            if (n.Outputs[i] != "")
            {
                _env[n.Outputs[i]] = Get(g.Outputs[i].Name);
            }
        }
    }

    /// <summary>Concat of any number of values (the IR op takes at most 64).</summary>
    private Value ConcatMany(int axis, Value[] vals)
    {
        var cur = vals;
        while (cur.Length > 64)
        {
            cur = [.. cur.Chunk(64).Select(c => c.Length == 1 ? c[0] : _b.Concat(axis, c))];
        }

        return cur.Length == 1 ? cur[0] : _b.Concat(axis, cur);
    }

    // ---- RNN family ---------------------------------------------------------------------------------------------------
    private static OnnxConst Slice0(OnnxConst c, long i)
    {
        long per = c.Count / c.Dims[0]; int es = OnnxType.ElementSize(c.ElemType);
        return OnnxConst.Lazy(c.ElemType, c.Dims[1..], () => c.Bytes().AsSpan((int)(i * per * es), (int)(per * es)).ToArray(), c.Name + $"[{i}]");
    }

    private void Rnn(OnnxNode n, Tn?[] ins)
    {
        string kind = n.OpType;
        if (n.Has("activations") && n.Attributes["activations"].Strings.Count > 0)
        {
            var def = kind == "LSTM" ? new[] { "Sigmoid", "Tanh", "Tanh" } : kind == "GRU" ? new[] { "Sigmoid", "Tanh" } : new[] { "Tanh" };
            var got = n.Attributes["activations"].Strings;
            if (!got.Take(def.Length).SequenceEqual(def, StringComparer.OrdinalIgnoreCase) && got.Count <= 2 * def.Length)
            {
                throw new NotSupportedException("custom RNN activations");
            }
        }
        if (n.Has("clip"))
        {
            throw new NotSupportedException("RNN cell clipping");
        }

        if (kind == "LSTM" && ins.Length > 7 && ins[7] is not null)
        {
            throw new NotSupportedException("LSTM peepholes");
        }

        if (ins.Length > 4 && ins[4] is not null)
        {
            throw new NotSupportedException("RNN sequence_lens");
        }

        long hs = n.Int("hidden_size", 0); int layout = (int)n.Int("layout", 0);
        bool lbr = n.Int("linear_before_reset", 0) != 0;
        string dir = n.Str("direction", "forward");
        var X0 = X(ins, 0);
        long seq = layout == 0 ? X0.Shape[0].Fixed : X0.Shape[1].Fixed;   // a symbolic sequence axis is rejected here
        var W = ConstIn(ins, 1, "W"); var R = ConstIn(ins, 2, "R"); var Bc = OptConst(ins, 3, "B");
        int D = (int)W.Dims[0]; int gates = kind == "LSTM" ? 4 : kind == "GRU" ? 3 : 1;
        if (hs == 0)
        {
            hs = R.Dims[2];
        }

        var Ys = new Value?[seq][];
        var lastH = new Value[D]; var lastC = new Value?[D];
        long bsz = layout == 0 ? X0.Shape[1].Fixed : 0;
        for (int d = 0; d < D; d++)
        {
            bool rev = dir == "reverse" || (dir == "bidirectional" && d == 1);
            var Wd = WeightValue(Slice0(W, d), false, Nm(n) + $".W{d}");
            var Rd = WeightValue(Slice0(R, d), false, Nm(n) + $".R{d}");
            Value? bW = null, bR = null;
            if (Bc is not null)
            {
                var bd = Slice0(Bc, d).F32();
                bW = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [gates * hs], bd.Take((int)(gates * hs)).Select(v => (double)v).ToArray(), Nm(n) + $".bW{d}"), null);
                bR = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [gates * hs], bd.Skip((int)(gates * hs)).Take((int)(gates * hs)).Select(v => (double)v).ToArray(), Nm(n) + $".bR{d}"), null);
            }
            Value? h = null, c = null;
            if (ins.Length > 5 && ins[5] is { } h0)
            {
                h = h0.IsConst ? ConstValue(Slice0(h0.C!, d), null) : layout == 0 ? Rs(_b.Slice(h0.V!, 0, d, 1), bsz, hs) : Rs(_b.Slice(h0.V!, 1, d, 1), 0, hs);
            }
            if (kind == "LSTM" && ins.Length > 6 && ins[6] is { } c0)
            {
                c = c0.IsConst ? ConstValue(Slice0(c0.C!, d), null) : layout == 0 ? Rs(_b.Slice(c0.V!, 0, d, 1), bsz, hs) : Rs(_b.Slice(c0.V!, 1, d, 1), 0, hs);
            }
            for (int s = 0; s < seq; s++)
            {
                int t = rev ? (int)seq - 1 - s : s;
                Value xt = layout == 0 ? Rs(_b.Slice(X0, 0, t, 1), bsz, X0.Shape[2].Fixed) : Rs(_b.Slice(X0, 1, t, 1), 0, X0.Shape[2].Fixed);
                Value Lin(Value inp, Value w, Value? bias) { var m = _b.MatMul(inp, w); Heavy(); return bias is null ? m : _b.Add(m, bias); }
                Value Part(Value v, int g) => _b.Slice(v, 1, g * hs, hs);
                var xw = Lin(xt, Wd, bW);
                if (kind == "RNN")
                {
                    var pre = h is null ? (bR is null ? xw : _b.Add(xw, bR)) : _b.Add(xw, Lin(h, Rd, bR));
                    h = _b.Unary("Tanh", pre);
                }
                else if (kind == "LSTM")
                {
                    var g = h is null ? (bR is null ? xw : _b.Add(xw, bR)) : _b.Add(xw, Lin(h, Rd, bR));
                    var i = _b.Sigmoid(Part(g, 0)); var o = _b.Sigmoid(Part(g, 1)); var f = _b.Sigmoid(Part(g, 2)); var cc = _b.Unary("Tanh", Part(g, 3));
                    c = c is null ? _b.Mul(i, cc) : _b.Add(_b.Mul(f, c), _b.Mul(i, cc));
                    h = _b.Mul(o, _b.Unary("Tanh", c));
                }
                else
                {
                    Value? hr = h is null ? null : Lin(h, Rd, null);   // h·Rᵀ for z,r,h gates
                    Value zr(int g)
                    {
                        var v = Part(xw, g);
                        if (hr is not null)
                        {
                            v = _b.Add(v, Part(hr, g));
                        }

                        if (bR is not null)
                        {
                            v = _b.Add(v, _b.Slice(bR, 0, g * hs, hs));
                        }

                        return v;
                    }
                    var z = _b.Sigmoid(zr(0)); var r = _b.Sigmoid(zr(1));
                    Value cand;
                    if (lbr)
                    {
                        var rh = h is null ? null : Part(hr!, 2);
                        if (bR is not null) { var rb = _b.Slice(bR, 0, 2 * hs, hs); rh = rh is null ? rb : _b.Add(rh, rb); }
                        cand = Part(xw, 2); if (rh is not null)
                        {
                            cand = _b.Add(cand, _b.Mul(r, rh));
                        }
                    }
                    else
                    {
                        cand = Part(xw, 2);
                        if (h is not null)
                        {
                            cand = _b.Add(cand, Lin(_b.Mul(r, h), SliceRows(Rd, 2, hs, Nm(n) + $".Rh{d}", R, d), null));
                        }

                        if (bR is not null)
                        {
                            cand = _b.Add(cand, _b.Slice(bR, 0, 2 * hs, hs));
                        }
                    }
                    cand = _b.Unary("Tanh", cand);
                    h = h is null ? _b.Mul(_b.Scale(z, -1f, 1f), cand) : _b.Add(_b.Mul(_b.Scale(z, -1f, 1f), cand), _b.Mul(z, h));
                }
                (Ys[t] ??= new Value?[D])[d] = h;
            }
            lastH[d] = h!; lastC[d] = c;
        }
        if (n.Outputs.Count > 0 && n.Outputs[0] != "")
        {
            // Y: layout 0 [seq, D, B, H]; layout 1 [B, seq, D, H]
            var perTime = new List<Value>();
            for (int t = 0; t < seq; t++)
            {
                var hs_ = Ys[t].Select(v => layout == 0 ? Rs(v!, 1, 1, bsz, hs) : Rs(v!, 0, 1, 1, hs)).ToArray();
                perTime.Add(D == 1 ? hs_[0] : _b.Concat(layout == 0 ? 1 : 2, hs_));
            }
            SetDyn(n, 0, perTime.Count == 1 ? perTime[0] : ConcatMany(layout == 0 ? 0 : 1, perTime.ToArray()));
        }
        Value Stack(Value[] v) => D == 1 ? (layout == 0 ? Rs(v[0], 1, bsz, hs) : Rs(v[0], 0, 1, hs))
            : ConcatMany(layout == 0 ? 0 : 1, v.Select(x => layout == 0 ? Rs(x, 1, bsz, hs) : Rs(x, 0, 1, hs)).ToArray());
        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, Stack(lastH));
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, Stack(lastC.Select(c => c ?? throw new NotSupportedException("Y_c requested for a non-LSTM")).ToArray()));
        }
    }

    /// <summary>Rows [g*H, (g+1)*H) of a recurrent weight [G*H, H] as a new weight.</summary>
    private Value SliceRows(Value full, int g, long hs, string name, OnnxConst R, int d)
    {
        var sub = Slice0(R, d);
        long cols = sub.Dims[1], es = OnnxType.ElementSize(sub.ElemType);
        var rows = OnnxConst.Lazy(sub.ElemType, [hs, cols], () => sub.Bytes().AsSpan((int)(g * hs * cols * es), (int)(hs * cols * es)).ToArray(), name);
        return WeightValue(rows, false, name);
    }

    // ---- com.microsoft contrib ops -------------------------------------------------------------------------------------
    private void Contrib(OnnxNode n, Tn?[] ins)
    {
        string name = Nm(n);
        switch (n.OpType)
        {
            case "Gelu": SetDyn(n, 0, Emit("Gelu", [X(ins, 0)], null, name)); break;
            case "BiasGelu": SetDyn(n, 0, Emit("Gelu", [_b.Add(X(ins, 0), X(ins, 1))], null, name)); break;
            case "FastGelu":
                {
                    var x = X(ins, 0); if (ins.Length > 1 && ins[1] is not null)
                    {
                        x = _b.Add(x, X(ins, 1));
                    }

                    SetDyn(n, 0, Emit("GeluTanh", [x], null, name)); break;
                }
            case "QuickGelu":
                {
                    var x = X(ins, 0);
                    SetDyn(n, 0, Emit("Mul", [x, _b.Sigmoid(_b.Scale(x, n.Float("alpha", 1.702f)))], null, name)); break;
                }
            case "BiasAdd":
                {
                    var y = _b.Add(X(ins, 0), X(ins, 1)); if (ins.Length > 2 && ins[2] is not null)
                    {
                        y = _b.Add(y, X(ins, 2));
                    }

                    SetDyn(n, 0, y); break;
                }
            case "FusedConv":
                {
                    var conv = new OnnxNode { OpType = "Conv", Name = n.Name, Inputs = n.Inputs.Take(3).ToList(), Outputs = [name + ".conv"], Attributes = n.Attributes };
                    Conv(conv, ins.Take(3).ToArray());
                    var y = _env[name + ".conv"].V!;
                    if (ins.Length > 3 && ins[3] is not null)
                    {
                        y = _b.Add(y, X(ins, 3));
                    }

                    y = Act(n.Has("activation") ? n.Str("activation", "") : null, y, n.Floats("activation_params"));
                    SetDyn(n, 0, y); break;
                }
            case "FusedMatMul":
                SetDyn(n, 0, Emit("BatchMatMul", [X(ins, 0), X(ins, 1)], Attributes.From(("trans_a", n.Int("transA", 0) != 0), ("trans_b", n.Int("transB", 0) != 0), ("alpha", n.Float("alpha", 1f))), name));
                Heavy(); break;
            case "SkipLayerNormalization":
            case "SkipSimplifiedLayerNormalization":
                {
                    var sum = _b.Add(X(ins, 0), X(ins, 1));
                    bool simple = n.OpType == "SkipSimplifiedLayerNormalization";
                    int biasIdx = simple ? 3 : 4;
                    if (ins.Length > biasIdx && ins[biasIdx] is not null)
                    {
                        sum = _b.Add(sum, X(ins, biasIdx));
                    }

                    Value y;
                    float eps = n.Float("epsilon", 1e-12f);
                    if (simple)
                    {
                        y = Emit("RmsNorm", [sum, X(ins, 2)], Attributes.From(("eps", eps)), name);
                    }
                    else
                    {
                        var args = new List<Value> { sum, X(ins, 2) };
                        if (ins.Length > 3 && ins[3] is not null)
                        {
                            args.Add(X(ins, 3));
                        }

                        y = Emit("LayerNorm", [.. args], Attributes.From(("eps", eps)), name);
                    }
                    SetDyn(n, 0, y);
                    if (n.Outputs.Count > 3 && n.Outputs[3] != "")
                    {
                        SetDyn(n, 3, sum);
                    }

                    break;
                }
            case "SimplifiedLayerNormalization": RmsNorm(n, ins); break;
            case "MatMulNBits": MatMulNBits(n, ins); break;
            case "Attention": ContribAttention(n, ins); break;
            case "MultiHeadAttention":
                {
                    if (ins.Length > 4 && ins.Skip(4).Any(t => t is not null))
                    {
                        throw new NotSupportedException("MultiHeadAttention with mask/past/bias inputs beyond bias");
                    }

                    var q = X(ins, 0); var k = X(ins, 1); var v = X(ins, 2);
                    if (ins.Length > 3 && ins[3] is not null)
                    {
                        long h = q.Shape[2].Fixed; var bias = X(ins, 3);
                        q = _b.Add(q, _b.Slice(bias, 0, 0, h)); k = _b.Add(k, _b.Slice(bias, 0, h, h)); v = _b.Add(v, _b.Slice(bias, 0, 2 * h, h));
                    }
                    SetDyn(n, 0, AttentionCore(q, k, v, (int)n.Int("num_heads", 1), n.Has("scale") ? n.Float("scale", 0) : 0f, false, name)); break;
                }
            default: throw new NotSupportedException(n.OpType);
        }
    }

    private Value AttentionCore(Value q, Value k, Value v, int heads, float scale, bool causal, string name)
    {
        long hidden = q.Shape[2].Fixed, dh = hidden / heads, S = q.Shape[1].Fixed, Sk = k.Shape[1].Fixed, vh = v.Shape[2].Fixed / heads;
        if (scale == 0)
        {
            scale = 1f / MathF.Sqrt(dh);
        }

        Value Heads(Value x, long d, long len) => _b.Permute(Rs(x, 0, len, heads, d), 0, 2, 1, 3);
        var qh = Heads(q, dh, S); var kh = Heads(k, dh, Sk); var vv = Heads(v, vh, Sk);
        var scores = Emit("BatchMatMul", [qh, kh], Attributes.From(("trans_b", true), ("alpha", scale)));
        Heavy();
        if (causal)
        {
            var mask = new byte[S * Sk * 4];
            for (long i = 0; i < S; i++)
            {
                for (long j = 0; j < Sk; j++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(mask.AsSpan((int)((i * Sk + j) * 4), 4), j > i + (Sk - S) ? -1e9f : 0f);
                }
            }

            scores = _b.Add(scores, _b.Constant(name + ".causal", DataType.F32, new Shape(S, Sk), mask));
        }
        var p = _b.Emit1("Softmax", [scores]);
        var o = Emit("BatchMatMul", [p, vv]);
        Heavy();
        return Emit("Reshape", [_b.Permute(o, 0, 2, 1, 3)], Attributes.From(("shape", L(0, S, heads * vh))), name);
    }

    private void ContribAttention(OnnxNode n, Tn?[] ins)
    {
        if ((ins.Length > 3 && ins[3] is not null) || (ins.Length > 4 && ins[4] is not null) || (ins.Length > 5 && ins[5] is not null))
        {
            throw new NotSupportedException("Attention with mask_index, past or relative position bias");
        }

        if (n.Has("qkv_hidden_sizes"))
        {
            throw new NotSupportedException("Attention with unequal qkv_hidden_sizes");
        }

        var x = X(ins, 0);
        var w = ConstIn(ins, 1, "Attention weights");
        var qkv = _b.MatMul(x, WeightValue(w.Transposed([1, 0]), true, Nm(n) + ".qkv"));
        Heavy();
        if (ins.Length > 2 && ins[2] is not null)
        {
            qkv = _b.Add(qkv, X(ins, 2));
        }

        long h = qkv.Shape[2].Fixed / 3;
        SetDyn(n, 0, AttentionCore(_b.Slice(qkv, 2, 0, h), _b.Slice(qkv, 2, h, h), _b.Slice(qkv, 2, 2 * h, h), (int)n.Int("num_heads", 1), n.Has("scale") ? n.Float("scale", 0) : 0f, n.Int("unidirectional", 0) != 0, Nm(n)));
    }

    private void MatMulNBits(OnnxNode n, Tn?[] ins)
    {
        long K = n.Int("K", 0), N = n.Int("N", 0), bits = n.Int("bits", 4), bs = n.Int("block_size", 32);
        if (bits is not (4 or 8))
        {
            throw new NotSupportedException($"MatMulNBits with {bits} bits");
        }

        if (ins.Length > 4 && ins[4] is not null)
        {
            throw new NotSupportedException("MatMulNBits with g_idx");
        }

        var B = ConstIn(ins, 1, "quantized weight"); var scales = ConstIn(ins, 2, "scales"); var zps = OptConst(ins, 3, "zero points");
        long nb = (K + bs - 1) / bs;
        var wc = OnnxConst.Lazy(OnnxType.Float, [N, K], () =>
        {
            var q = B.Bytes(); var sc = scales.Floats(); var zb = zps?.Bytes();
            var res = new float[N * K];
            long blobPerBlock = bs * bits / 8;
            long zpRow = (nb * bits + 7) / 8;
            for (long r = 0; r < N; r++)
            {
                for (long k = 0; k < K; k++)
                {
                    long blk = k / bs, j = k % bs;
                    int v = bits == 4 ? (q[(r * nb + blk) * blobPerBlock + j / 2] >> (int)(4 * (j & 1))) & 0xF : q[(r * nb + blk) * blobPerBlock + j];
                    int z = bits == 4 ? 8 : 128;
                    if (zb is not null)
                    {
                        z = bits == 4 ? (zb[r * zpRow + blk / 2] >> (int)(4 * (blk & 1))) & 0xF : zb[r * nb + blk];
                    }

                    res[r * K + k] = (float)((v - z) * sc[r * nb + blk]);
                }
            }

            var bytes = new byte[res.Length * 4]; Buffer.BlockCopy(res, 0, bytes, 0, bytes.Length); return bytes;
        }, Nm(n) + ".dequant");
        var y = _b.MatMul(X(ins, 0), WeightValue(wc, false, Nm(n) + ".weight"));
        Heavy();
        if (ins.Length > 5 && ins[5] is not null)
        {
            y = _b.Add(y, X(ins, 5));
        }

        SetDyn(n, 0, y);
    }
}
