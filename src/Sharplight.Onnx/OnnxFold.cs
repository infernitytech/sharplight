namespace Sharplight.Onnx;

/// <summary>Host-side evaluation of constant subgraphs (shape computations, weight preprocessing).</summary>
internal static class OnnxFold
{
    /// <summary>Ops that can be evaluated when all their inputs are constants.</summary>
    public static readonly HashSet<string> Ops =
    [
        "Constant", "Identity", "Dropout", "Cast", "Shape", "Size", "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Sum", "Neg", "Abs", "Sqrt", "Exp", "Log",
        "Reciprocal", "Floor", "Ceil", "Relu", "Sigmoid", "Tanh", "Erf", "Not", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or",
        "Where", "Gather", "Unsqueeze", "Squeeze", "Concat", "Slice", "Reshape", "Flatten", "Transpose", "Expand", "Range", "ConstantOfShape", "Split",
        "ReduceProd", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "Clip",
        "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Round", "Sign", "Softplus", "Softsign", "Mod", "IsNaN", "IsInf",
        "Tile", "NonZero", "EyeLike", "QuantizeLinear", "DequantizeLinear", "CastLike", "ArgMax", "ArgMin",
    ];

    private static readonly Dictionary<string, Func<double, double>> s_math = new()
    {
        ["Sin"] = Math.Sin,
        ["Cos"] = Math.Cos,
        ["Tan"] = Math.Tan,
        ["Asin"] = Math.Asin,
        ["Acos"] = Math.Acos,
        ["Atan"] = Math.Atan,
        ["Sinh"] = Math.Sinh,
        ["Cosh"] = Math.Cosh,
        ["Asinh"] = Math.Asinh,
        ["Acosh"] = Math.Acosh,
        ["Atanh"] = Math.Atanh,
        ["Round"] = v => Math.Round(v, MidpointRounding.ToEven),
        ["Sign"] = v => Math.Sign(v),
        ["Softplus"] = v => Math.Log(1 + Math.Exp(v)),
        ["Softsign"] = v => v / (1 + Math.Abs(v)),
    };

    private static bool IsSym(long v) => v == OnnxConst.SymBatch;

    public static IReadOnlyList<OnnxConst> Eval(OnnxNode n, IReadOnlyList<OnnxConst?> ins)
    {
        OnnxConst X(int i) => i < ins.Count && ins[i] is { } c ? c : throw new InvalidDataException($"{n.OpType} '{n.Name}': missing input {i}.");
        OnnxConst? Opt(int i) => i < ins.Count ? ins[i] : null;
        if (s_math.TryGetValue(n.OpType, out var mf))
        {
            return [UnF(X(0), mf)];
        }

        switch (n.OpType)
        {
            case "IsNaN":
                return [OnnxConst.FromInts(OnnxType.Bool, X(0).Dims, X(0).Floats().Select(v => double.IsNaN(v) ? 1L : 0L).ToArray())];
            case "IsInf":
                return [OnnxConst.FromInts(OnnxType.Bool, X(0).Dims, X(0).Floats().Select(v => double.IsInfinity(v) ? 1L : 0L).ToArray())];
            case "Mod":
                {
                    bool fm = n.Int("fmod", 0) != 0;
                    return [Bin(X(0), X(1), (a, b) => fm ? a % b : a - Math.Floor(a / b) * b,
                    (a, b) => b == 0 ? throw new DivideByZeroException() : fm ? a % b : ((a % b) + b) % b)];
                }
            case "Tile":
                {
                    var x = X(0);
                    var rep = X(1).Ints();
                    var dims = x.Dims.Select((d, i) => d * rep[i]).ToArray();
                    long total = 1;
                    foreach (var d in dims)
                    {
                        total *= d;
                    }

                    var map = new long[total];
                    var idx = new long[x.Rank];
                    var st = new long[x.Rank];
                    long q = 1;
                    for (int d = x.Rank - 1; d >= 0; d--) { st[d] = q; q *= x.Dims[d]; }
                    for (long o = 0; o < total; o++)
                    {
                        long p = 0;
                        for (int d = 0; d < x.Rank; d++)
                        {
                            p += (idx[d] % x.Dims[d]) * st[d];
                        }

                        map[o] = p;
                        for (int d = x.Rank - 1; d >= 0; d--)
                        {
                            if (++idx[d] < dims[d]) { break; } idx[d] = 0;
                        }
                    }
                    return [x.IsFloat ? OnnxConst.FromFloats(x.ElemType, dims, map.Select(p => x.Floats()[p]).ToArray()) : OnnxConst.FromInts(x.ElemType, dims, map.Select(p => x.Ints()[p]).ToArray())];
                }
            case "NonZero":
                {
                    var x = X(0);
                    var f = x.Floats();
                    var nz = Enumerable.Range(0, f.Length).Where(i => f[i] != 0).ToArray();
                    var res = new long[x.Rank * nz.Length];
                    for (int k = 0; k < nz.Length; k++)
                    {
                        long rem = nz[k];
                        for (int d = x.Rank - 1; d >= 0; d--)
                        {
                            res[d * nz.Length + k] = rem % x.Dims[d];
                            rem /= x.Dims[d];
                        }
                    }
                    return [OnnxConst.FromInts(OnnxType.Int64, [x.Rank, nz.Length], res)];
                }
            case "EyeLike":
                {
                    var x = X(0);
                    long k = n.Int("k", 0);
                    int et = n.Has("dtype") ? (int)n.Int("dtype", OnnxType.Float) : x.ElemType;
                    var v = new double[x.Count];
                    for (long r = 0; r < x.Dims[0]; r++)
                    {
                        long c = r + k;
                        if (c >= 0 && c < x.Dims[1])
                        {
                            v[r * x.Dims[1] + c] = 1;
                        }
                    }
                    return [OnnxConst.FromFloats(OnnxType.Float, x.Dims, v).CastTo(et)];
                }
            case "CastLike":
                return [X(0).CastTo(X(1).ElemType)];
            case "ArgMax":
            case "ArgMin":
                {
                    var x = X(0);
                    int axis = (int)n.Int("axis", 0);
                    if (axis < 0)
                    {
                        axis += x.Rank;
                    }

                    bool keep = n.Int("keepdims", 1) != 0;
                    bool last = n.Int("select_last_index", 0) != 0;
                    long outer = 1;
                    long inner = 1;
                    for (int i = 0; i < axis; i++)
                    {
                        outer *= x.Dims[i];
                    }

                    for (int i = axis + 1; i < x.Rank; i++)
                    {
                        inner *= x.Dims[i];
                    }

                    long len = x.Dims[axis];
                    var f = x.Floats();
                    var r = new long[outer * inner];
                    for (long o = 0; o < outer; o++)
                    {
                        for (long i = 0; i < inner; i++)
                        {
                            long best = 0;
                            for (long j = 1; j < len; j++)
                            {
                                double a = f[(o * len + j) * inner + i];
                                double b = f[(o * len + best) * inner + i];
                                bool better = n.OpType == "ArgMax" ? (last ? a >= b : a > b) : (last ? a <= b : a < b);
                                if (better)
                                {
                                    best = j;
                                }
                            }
                            r[o * inner + i] = best;
                        }
                    }

                    var dims = x.Dims.ToList();
                    if (keep)
                    {
                        dims[axis] = 1;
                    }
                    else
                    {
                        dims.RemoveAt(axis);
                    }

                    return [OnnxConst.FromInts(OnnxType.Int64, [.. dims], r)];
                }
            case "DequantizeLinear":
                return [Dequantize(X(0), X(1), Opt(2), (int)n.Int("axis", 1))];
            case "QuantizeLinear":
                {
                    var x = X(0);
                    var sc = X(1);
                    var zp = Opt(2);
                    int et = zp?.ElemType ?? OnnxType.UInt8;
                    int axis = (int)n.Int("axis", 1);
                    if (axis < 0)
                    {
                        axis += x.Rank;
                    }

                    var (lo, hi) = Range(et);
                    var f = x.Floats();
                    var s = sc.Floats();
                    var z = zp?.Floats();
                    long inner = 1;
                    for (int i = axis + 1; i < x.Rank; i++)
                    {
                        inner *= x.Dims[i];
                    }

                    long ax = x.Rank == 0 ? 1 : x.Dims[axis];
                    var r = new long[f.Length];
                    for (long i = 0; i < f.Length; i++)
                    {
                        long c = s.Length == 1 ? 0 : (i / inner) % ax;
                        double q = Math.Round((float)f[i] / (float)s[c], MidpointRounding.ToEven) + (z is null ? 0 : z[z.Length == 1 ? 0 : c]);
                        r[i] = (long)Math.Clamp(q, lo, hi);
                    }
                    return [OnnxConst.FromInts(et, x.Dims, r)];
                }
            case "Constant":
                return [ConstantValue(n)];
            case "Identity":
                return [X(0)];
            case "Dropout":
                return [X(0)];
            case "Cast":
                return [X(0).CastTo((int)n.Int("to", OnnxType.Float))];
            case "Shape":
                {
                    var d = X(0).Dims;
                    long s = n.Int("start", 0);
                    long e = n.Int("end", d.Length);
                    if (s < 0)
                    {
                        s += d.Length;
                    }

                    if (e < 0)
                    {
                        e += d.Length;
                    }

                    s = Math.Clamp(s, 0, d.Length);
                    e = Math.Clamp(e, s, d.Length);
                    return [OnnxConst.FromInts(OnnxType.Int64, [e - s], d[(int)s..(int)e])];
                }
            case "Size":
                return [OnnxConst.ScalarInt(X(0).Count)];
            case "Add":
                return [Bin(X(0), X(1), (a, b) => a + b, (a, b) => a + b)];
            case "Sub":
                return [Bin(X(0), X(1), (a, b) => a - b, (a, b) => a - b)];
            case "Mul":
                return [Bin(X(0), X(1), (a, b) => a * b, (a, b) => a * b)];
            case "Div":
                return [Bin(X(0), X(1), (a, b) => a / b, (a, b) => b == 0 ? throw new DivideByZeroException($"Integer division by zero in constant '{n.Name}'.") : a / b)];
            case "Pow":
                return [Bin(X(0), X(1), Math.Pow, (a, b) => (long)Math.Pow(a, b))];
            case "Max":
            case "Min":
            case "Sum":
                {
                    Func<double, double, double> f = n.OpType == "Max" ? Math.Max : n.OpType == "Min" ? Math.Min : (a, b) => a + b;
                    Func<long, long, long> g = n.OpType == "Max" ? Math.Max : n.OpType == "Min" ? Math.Min : (a, b) => a + b;
                    var r = X(0);
                    for (int i = 1; i < ins.Count; i++)
                    {
                        r = Bin(r, X(i), f, g);
                    }

                    return [r];
                }
            case "Equal":
                return [Cmp(X(0), X(1), (a, b) => a == b)];
            case "Less":
                return [Cmp(X(0), X(1), (a, b) => a < b)];
            case "Greater":
                return [Cmp(X(0), X(1), (a, b) => a > b)];
            case "LessOrEqual":
                return [Cmp(X(0), X(1), (a, b) => a <= b)];
            case "GreaterOrEqual":
                return [Cmp(X(0), X(1), (a, b) => a >= b)];
            case "And":
                return [Cmp(X(0), X(1), (a, b) => a != 0 && b != 0)];
            case "Or":
                return [Cmp(X(0), X(1), (a, b) => a != 0 || b != 0)];
            case "Not":
                return [OnnxConst.FromInts(OnnxType.Bool, X(0).Dims, X(0).Ints().Select(v => v == 0 ? 1L : 0L).ToArray())];
            case "Neg":
                return [Un(X(0), v => -v, v => -v)];
            case "Abs":
                return [Un(X(0), Math.Abs, Math.Abs)];
            case "Sqrt":
                return [UnF(X(0), Math.Sqrt)];
            case "Exp":
                return [UnF(X(0), Math.Exp)];
            case "Log":
                return [UnF(X(0), Math.Log)];
            case "Reciprocal":
                return [UnF(X(0), v => 1.0 / v)];
            case "Floor":
                return [UnF(X(0), Math.Floor)];
            case "Ceil":
                return [UnF(X(0), Math.Ceiling)];
            case "Relu":
                return [Un(X(0), v => Math.Max(v, 0), v => Math.Max(v, 0))];
            case "Sigmoid":
                return [UnF(X(0), v => 1 / (1 + Math.Exp(-v)))];
            case "Tanh":
                return [UnF(X(0), Math.Tanh)];
            case "Erf":
                return [UnF(X(0), Erf)];
            case "Clip":
                {
                    double lo = n.Has("min") ? n.Float("min", 0) : Opt(1) is { } l ? l.Floats()[0] : double.NegativeInfinity;
                    double hi = n.Has("max") ? n.Float("max", 0) : Opt(2) is { } h ? h.Floats()[0] : double.PositiveInfinity;
                    return [Un(X(0), v => Math.Min(Math.Max(v, lo), hi), v => (long)Math.Min(Math.Max(v, lo), hi))];
                }
            case "Where":
                {
                    var c = X(0);
                    var a = X(1);
                    var b = X(2);
                    var abDims = Broadcast(a.Dims, b.Dims);
                    bool abFloat = a.IsFloat || b.IsFloat;
                    int abType = a.IsFloat ? a.ElemType : b.IsFloat ? b.ElemType : a.ElemType;
                    var dims = Broadcast(c.Dims, abDims);
                    var cb = ExpandTo(c, dims);
                    var ae = ExpandTo(a, dims);
                    var be = ExpandTo(b, dims);
                    var cv = cb.Ints();
                    if (abFloat)
                    {
                        var av = ae.Floats();
                        var bv = be.Floats();
                        return [OnnxConst.FromFloats(abType, dims, cv.Select((m, i) => m != 0 ? av[i] : bv[i]).ToArray())];
                    }
                    var ai = ae.Ints();
                    var bi = be.Ints();
                    return [OnnxConst.FromInts(abType, dims, cv.Select((m, i) => m != 0 ? ai[i] : bi[i]).ToArray())];
                }
            case "Gather":
                return [Gather(X(0), X(1), (int)n.Int("axis", 0))];
            case "Unsqueeze":
                {
                    var axes = n.Has("axes") ? n.Ints("axes")! : X(1).Ints();
                    return [X(0).Reshaped(UnsqueezeDims(X(0).Dims, axes))];
                }
            case "Squeeze":
                {
                    var axes = n.Has("axes") ? n.Ints("axes") : Opt(1)?.Ints();
                    return [X(0).Reshaped(SqueezeDims(X(0).Dims, axes))];
                }
            case "Concat":
                {
                    int axis = (int)n.Int("axis", 0);
                    return [Concat(ins.Select((c, i) => c ?? throw new InvalidDataException("Concat: missing input")).ToList(), axis)];
                }
            case "Slice":
                return [Slice(n, ins)];
            case "Reshape":
                {
                    var x = X(0);
                    var req = X(1).Ints();
                    bool allowZero = n.Int("allowzero", 0) != 0;
                    var dims = new long[req.Length];
                    long known = 1;
                    int infer = -1;
                    for (int i = 0; i < req.Length; i++)
                    {
                        if (req[i] == 0 && !allowZero)
                        {
                            dims[i] = x.Dims[i];
                        }
                        else if (req[i] == -1)
                        {
                            infer = i;
                            continue;
                        }
                        else
                        {
                            dims[i] = req[i];
                        }

                        known *= dims[i];
                    }
                    if (infer >= 0)
                    {
                        dims[infer] = known == 0 ? 0 : x.Count / known;
                    }

                    return [x.Reshaped(dims)];
                }
            case "Flatten":
                {
                    var x = X(0);
                    int axis = (int)n.Int("axis", 1);
                    if (axis < 0)
                    {
                        axis += x.Rank;
                    }

                    long a = 1;
                    long b = 1;
                    for (int i = 0; i < x.Rank; i++)
                    {
                        if (i < axis)
                        {
                            a *= x.Dims[i];
                        }
                        else
                        {
                            b *= x.Dims[i];
                        }
                    }

                    return [x.Reshaped([a, b])];
                }
            case "Transpose":
                {
                    var x = X(0);
                    var perm = n.Ints("perm")?.Select(p => (int)p).ToArray() ?? Enumerable.Range(0, x.Rank).Reverse().ToArray();
                    return [x.Transposed(perm)];
                }
            case "Expand":
                return [ExpandTo(X(0), Broadcast(X(0).Dims, X(1).Ints().ToArray()))];
            case "Range":
                {
                    var s = X(0);
                    var l = X(1);
                    var d = X(2);
                    if (s.IsFloat || l.IsFloat || d.IsFloat)
                    {
                        double a = s.Floats()[0];
                        double b = l.Floats()[0];
                        double st = d.Floats()[0];
                        long cnt = Math.Max(0, (long)Math.Ceiling((b - a) / st));
                        return [OnnxConst.FromFloats(s.ElemType, [cnt], Enumerable.Range(0, (int)cnt).Select(i => a + i * st).ToArray())];
                    }
                    else
                    {
                        long a = s.Ints()[0];
                        long b = l.Ints()[0];
                        long st = d.Ints()[0];
                        if (IsSym(a) || IsSym(b) || IsSym(st))
                        {
                            throw new NotSupportedException("Range over the symbolic batch size is not supported.");
                        }

                        long cnt = Math.Max(0, (b - a + st - (st > 0 ? 1 : -1)) / st);
                        return [OnnxConst.FromInts(s.ElemType, [cnt], Enumerable.Range(0, (int)cnt).Select(i => a + i * st).ToArray())];
                    }
                }
            case "ConstantOfShape":
                {
                    var dims = X(0).Ints();
                    if (dims.Any(IsSym))
                    {
                        throw new NotSupportedException($"ConstantOfShape '{n.Name}' depends on the symbolic batch size.");
                    }

                    var v = n.Tensor("value") is { } t ? OnnxConst.FromTensor(t) : OnnxConst.Scalar(0);
                    long cnt = 1;
                    foreach (var d in dims)
                    {
                        cnt *= d;
                    }

                    return [v.IsFloat ? OnnxConst.FromFloats(v.ElemType, dims, Enumerable.Repeat(v.Floats()[0], (int)cnt).ToArray())
                                  : OnnxConst.FromInts(v.ElemType, dims, Enumerable.Repeat(v.Ints()[0], (int)cnt).ToArray())];
                }
            case "Split":
                return Split(n, X(0), Opt(1));
            case "ReduceProd":
            case "ReduceSum":
            case "ReduceMean":
            case "ReduceMax":
            case "ReduceMin":
                return [Reduce(n, X(0), Opt(1))];
        }
        throw new NotSupportedException($"Constant folding of {n.OpType} is not implemented.");
    }

    public static (double Lo, double Hi) Range(int et) => et switch
    {
        OnnxType.Int8 => (-128, 127),
        OnnxType.UInt8 => (0, 255),
        OnnxType.Int16 => (-32768, 32767),
        OnnxType.UInt16 => (0, 65535),
        OnnxType.Int32 => (int.MinValue, int.MaxValue),
        _ => (double.MinValue, double.MaxValue),
    };

    /// <summary>(x - zero_point) * scale with per-tensor or per-axis scale.</summary>
    public static OnnxConst Dequantize(OnnxConst x, OnnxConst sc, OnnxConst? zp, int axis)
    {
        if (axis < 0)
        {
            axis += x.Rank;
        }

        var f = x.Floats();
        var s = sc.Floats();
        var z = zp?.Floats();
        long inner = 1;
        for (int i = axis + 1; i < x.Rank; i++)
        {
            inner *= x.Dims[i];
        }

        long ax = x.Rank == 0 ? 1 : x.Dims[axis];
        var r = new double[f.Length];
        for (long i = 0; i < f.Length; i++)
        {
            long c = (i / inner) % ax;
            r[i] = (float)((f[i] - (z is null ? 0 : z[z.Length == 1 ? 0 : c])) * s[s.Length == 1 ? 0 : c]);
        }
        return OnnxConst.FromFloats(sc.IsFloat ? sc.ElemType : OnnxType.Float, x.Dims, r);
    }

    public static OnnxConst ConstantValue(OnnxNode n)
    {
        if (n.Tensor("value") is { } t)
        {
            return OnnxConst.FromTensor(t);
        }

        if (n.Has("value_float"))
        {
            return OnnxConst.Scalar(n.Float("value_float", 0));
        }

        if (n.Has("value_floats"))
        {
            var f = n.Floats("value_floats")!;
            return OnnxConst.FromFloats(OnnxType.Float, [f.Length], f.Select(v => (double)v).ToArray());
        }
        if (n.Has("value_int"))
        {
            return OnnxConst.ScalarInt(n.Int("value_int", 0));
        }

        if (n.Has("value_ints"))
        {
            var i = n.Ints("value_ints")!;
            return OnnxConst.FromInts(OnnxType.Int64, [i.Length], i);
        }
        throw new NotSupportedException($"Constant node '{n.Name}' has no supported value attribute.");
    }

    // ---- helpers --------------------------------------------------------------------------------
    public static long[] Broadcast(long[] a, long[] b)
    {
        int r = Math.Max(a.Length, b.Length);
        var o = new long[r];
        for (int k = 0; k < r; k++)
        {
            long da = k - (r - a.Length) >= 0 ? a[k - (r - a.Length)] : 1;
            long db = k - (r - b.Length) >= 0 ? b[k - (r - b.Length)] : 1;
            if (IsSym(db))
            {
                db = da; // Expand target holding the symbolic batch: keep source extent logic elsewhere
            }

            o[k] = da == db ? da : da == 1 ? db : db == 1 ? da : throw new ArgumentException($"Shapes [{string.Join(",", a)}] and [{string.Join(",", b)}] do not broadcast.");
        }
        return o;
    }

    private static long[] StridesFor(long[] dims, long[] outDims)
    {
        int r = outDims.Length;
        var st = new long[r];
        long s = 1;
        for (int d = dims.Length - 1; d >= 0; d--)
        {
            int od = d + (r - dims.Length);
            st[od] = dims[d] == 1 && outDims[od] != 1 ? 0 : s;
            s *= dims[d];
        }
        return st;
    }

    /// <summary>Calls <paramref name="f"/>(outIndex, aIndex, bIndex) for every element of the broadcast result.</summary>
    private static void ForEachBroadcast(long[] da, long[] db, long[] outDims, Action<long, long, long> f)
    {
        int r = outDims.Length;
        var sa = StridesFor(da, outDims);
        var sb = StridesFor(db, outDims);
        long total = 1;
        foreach (var d in outDims)
        {
            total *= d;
        }

        var idx = new long[r];
        long pa = 0;
        long pb = 0;
        for (long o = 0; o < total; o++)
        {
            f(o, pa, pb);
            for (int d = r - 1; d >= 0; d--)
            {
                idx[d]++;
                pa += sa[d];
                pb += sb[d];
                if (idx[d] < outDims[d])
                {
                    break;
                }

                pa -= sa[d] * outDims[d];
                pb -= sb[d] * outDims[d];
                idx[d] = 0;
            }
        }
    }

    private static OnnxConst ExpandTo(OnnxConst x, long[] dims)
    {
        if (x.Dims.SequenceEqual(dims))
        {
            return x;
        }

        long total = 1;
        foreach (var d in dims)
        {
            total *= d;
        }

        if (x.IsFloat)
        {
            var src = x.Floats();
            var dst = new double[total];
            ForEachBroadcast(x.Dims, x.Dims, dims, (o, a, _) => dst[o] = src[a]);
            return OnnxConst.FromFloats(x.ElemType, dims, dst);
        }
        else
        {
            var src = x.Ints();
            var dst = new long[total];
            ForEachBroadcast(x.Dims, x.Dims, dims, (o, a, _) => dst[o] = src[a]);
            return OnnxConst.FromInts(x.ElemType, dims, dst);
        }
    }

    private static OnnxConst Bin(OnnxConst a, OnnxConst b, Func<double, double, double> ff, Func<long, long, long> fi)
    {
        var dims = Broadcast(a.Dims, b.Dims);
        long total = 1;
        foreach (var d in dims)
        {
            total *= d;
        }

        if (a.IsFloat || b.IsFloat)
        {
            var av = a.Floats();
            var bv = b.Floats();
            var r = new double[total];
            ForEachBroadcast(a.Dims, b.Dims, dims, (o, i, j) => r[o] = ff(av[i], bv[j]));
            int et = a.IsFloat ? a.ElemType : b.ElemType;
            if (et == OnnxType.Float)
            {
                for (long i = 0; i < total; i++)
                {
                    r[i] = (float)r[i];
                }
            }

            return OnnxConst.FromFloats(et, dims, r);
        }
        else
        {
            var av = a.Ints();
            var bv = b.Ints();
            var r = new long[total];
            ForEachBroadcast(a.Dims, b.Dims, dims, (o, i, j) =>
            {
                if (IsSym(av[i]) || IsSym(bv[j]))
                {
                    throw new NotSupportedException("Arithmetic on the symbolic batch size is not supported; the model computes with a dynamic batch dimension.");
                }

                r[o] = fi(av[i], bv[j]);
            });
            return OnnxConst.FromInts(a.ElemType, dims, r);
        }
    }

    private static OnnxConst Cmp(OnnxConst a, OnnxConst b, Func<double, double, bool> f)
    {
        var dims = Broadcast(a.Dims, b.Dims);
        long total = 1;
        foreach (var d in dims)
        {
            total *= d;
        }

        var av = a.Floats();
        var bv = b.Floats();
        var r = new long[total];
        ForEachBroadcast(a.Dims, b.Dims, dims, (o, i, j) => r[o] = f(av[i], bv[j]) ? 1 : 0);
        return OnnxConst.FromInts(OnnxType.Bool, dims, r);
    }

    private static OnnxConst Un(OnnxConst x, Func<double, double> ff, Func<long, long> fi)
        => x.IsFloat ? OnnxConst.FromFloats(x.ElemType, x.Dims, x.Floats().Select(ff).ToArray())
                     : OnnxConst.FromInts(x.ElemType, x.Dims, x.Ints().Select(fi).ToArray());

    private static OnnxConst UnF(OnnxConst x, Func<double, double> f)
    {
        int et = x.IsFloat ? x.ElemType : OnnxType.Float;
        return OnnxConst.FromFloats(et, x.Dims, x.Floats().Select(v => et == OnnxType.Float ? (double)(float)f(v) : f(v)).ToArray());
    }

    public static double Erf(double x)
    {
        // Abramowitz-Stegun 7.1.26 is too coarse; use a series/continued-fraction hybrid.
        if (double.IsNaN(x))
        {
            return x;
        }

        double ax = Math.Abs(x);
        double r;
        if (ax < 2.5)
        {
            double sum = ax;
            double term = ax;
            int n = 0;
            while (Math.Abs(term) > 1e-17 * Math.Abs(sum) && n < 200)
            {
                n++;
                term *= -ax * ax / n;
                sum += term / (2 * n + 1);
            }
            r = 2 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double t = 0;
            for (int k = 60; k >= 1; k--)
            {
                t = k / 2.0 / (ax + t);
            }

            r = 1 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / (ax + t);
        }
        return x < 0 ? -r : r;
    }

    public static long[] UnsqueezeDims(long[] dims, long[] axes)
    {
        int outRank = dims.Length + axes.Length;
        var set = axes.Select(a => (int)(a < 0 ? a + outRank : a)).ToHashSet();
        var o = new long[outRank];
        int j = 0;
        for (int i = 0; i < outRank; i++)
        {
            o[i] = set.Contains(i) ? 1 : dims[j++];
        }

        return o;
    }

    public static long[] SqueezeDims(long[] dims, long[]? axes)
    {
        var set = axes is null ? null : axes.Select(a => (int)(a < 0 ? a + dims.Length : a)).ToHashSet();
        var o = new List<long>();
        for (int i = 0; i < dims.Length; i++)
        {
            if (set is null ? dims[i] != 1 : !set.Contains(i))
            {
                o.Add(dims[i]);
            }
        }

        return [.. o];
    }

    private static OnnxConst Gather(OnnxConst data, OnnxConst indices, int axis)
    {
        if (axis < 0)
        {
            axis += data.Rank;
        }

        long outer = 1;
        long inner = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= data.Dims[i];
        }

        for (int i = axis + 1; i < data.Rank; i++)
        {
            inner *= data.Dims[i];
        }

        long dimA = data.Dims[axis];
        var idx = indices.Ints().Select(v => v < 0 ? v + dimA : v).ToArray();
        var dims = data.Dims[..axis].Concat(indices.Dims).Concat(data.Dims[(axis + 1)..]).ToArray();
        long total = outer * idx.Length * inner;
        if (data.IsFloat)
        {
            var src = data.Floats();
            var dst = new double[total];
            long p = 0;
            for (long o = 0; o < outer; o++)
            {
                foreach (var ix in idx)
                {
                    CheckIdx(ix, dimA);
                    Array.Copy(src, (o * dimA + ix) * inner, dst, p, inner);
                    p += inner;
                }
            }

            return OnnxConst.FromFloats(data.ElemType, dims, dst);
        }
        else
        {
            var src = data.Ints();
            var dst = new long[total];
            long p = 0;
            for (long o = 0; o < outer; o++)
            {
                foreach (var ix in idx)
                {
                    CheckIdx(ix, dimA);
                    Array.Copy(src, (o * dimA + ix) * inner, dst, p, inner);
                    p += inner;
                }
            }

            return OnnxConst.FromInts(data.ElemType, dims, dst);
        }
    }

    private static void CheckIdx(long ix, long dim)
    {
        if (ix < 0 || ix >= dim)
        {
            throw new IndexOutOfRangeException($"Gather index {ix} out of range [0, {dim}).");
        }
    }

    private static OnnxConst Concat(List<OnnxConst> xs, int axis)
    {
        int rank = xs[0].Rank;
        if (axis < 0)
        {
            axis += rank;
        }

        long outer = 1;
        for (int i = 0; i < axis; i++)
        {
            outer *= xs[0].Dims[i];
        }

        var dims = (long[])xs[0].Dims.Clone();
        dims[axis] = xs.Sum(x => x.Dims[axis]);
        bool fl = xs.Any(x => x.IsFloat);
        int et = fl ? xs.First(x => x.IsFloat).ElemType : xs[0].ElemType;
        long total = 1;
        foreach (var d in dims)
        {
            total *= d;
        }

        if (fl)
        {
            var dst = new double[total];
            long p = 0;
            var arrs = xs.Select(x => x.Floats()).ToList();
            var chunk = xs.Select(x => x.Count / Math.Max(outer, 1)).ToList();
            for (long o = 0; o < outer; o++)
            {
                for (int k = 0; k < xs.Count; k++)
                {
                    Array.Copy(arrs[k], o * chunk[k], dst, p, chunk[k]);
                    p += chunk[k];
                }
            }

            return OnnxConst.FromFloats(et, dims, dst);
        }
        else
        {
            var dst = new long[total];
            long p = 0;
            var arrs = xs.Select(x => x.Ints()).ToList();
            var chunk = xs.Select(x => x.Count / Math.Max(outer, 1)).ToList();
            for (long o = 0; o < outer; o++)
            {
                for (int k = 0; k < xs.Count; k++)
                {
                    Array.Copy(arrs[k], o * chunk[k], dst, p, chunk[k]);
                    p += chunk[k];
                }
            }

            return OnnxConst.FromInts(et, dims, dst);
        }
    }

    /// <summary>Normalized (start, length) of an ONNX slice range on a dimension of extent <paramref name="dim"/>.</summary>
    public static (long Start, long Length) SliceRange(long start, long end, long dim)
    {
        if (start < 0)
        {
            start += dim;
        }

        if (end < 0)
        {
            end += dim;
        }

        start = Math.Clamp(start, 0, dim);
        end = Math.Clamp(end, 0, dim);
        return (start, Math.Max(0, end - start));
    }

    private static OnnxConst Slice(OnnxNode n, IReadOnlyList<OnnxConst?> ins)
    {
        var x = ins[0]!;
        long[] starts;
        long[] ends;
        long[]? axes, steps = null;
        if (n.Has("starts"))
        {
            starts = n.Ints("starts")!;
            ends = n.Ints("ends")!;
            axes = n.Ints("axes");
        }
        else
        {
            starts = ins[1]!.Ints();
            ends = ins[2]!.Ints();
            axes = ins.Count > 3 ? ins[3]?.Ints() : null;
            steps = ins.Count > 4 ? ins[4]?.Ints() : null;
        }
        axes ??= Enumerable.Range(0, starts.Length).Select(i => (long)i).ToArray();
        var pos = new long[x.Rank][];   // per axis: the source indices that are kept
        for (int d = 0; d < x.Rank; d++)
        {
            pos[d] = Enumerable.Range(0, (int)x.Dims[d]).Select(v => (long)v).ToArray();
        }

        for (int i = 0; i < axes.Length; i++)
        {
            int ax = (int)(axes[i] < 0 ? axes[i] + x.Rank : axes[i]);
            long step = steps is null ? 1 : steps[i];
            long dim = x.Dims[ax];
            if (step == 0)
            {
                throw new InvalidDataException($"Slice '{n.Name}': step 0");
            }

            long s0 = starts[i];
            long e0 = ends[i];
            if (s0 < 0)
            {
                s0 += dim;
            }

            if (e0 < 0)
            {
                e0 += dim;
            }

            var list = new List<long>();
            if (step > 0)
            {
                s0 = Math.Clamp(s0, 0, dim);
                e0 = Math.Clamp(e0, 0, dim);
                for (long k = s0; k < e0; k += step)
                {
                    list.Add(k);
                }
            }
            else
            {
                s0 = Math.Clamp(s0, 0, dim - 1);
                e0 = Math.Clamp(e0, -1, dim - 1);
                for (long k = s0; k > e0; k += step)
                {
                    list.Add(k);
                }
            }
            pos[ax] = [.. list];
        }
        var len = pos.Select(p => (long)p.Length).ToArray();
        long total = 1;
        foreach (var d in len)
        {
            total *= d;
        }

        var srcStr = new long[x.Rank];
        long st = 1;
        for (int d = x.Rank - 1; d >= 0; d--)
        {
            srcStr[d] = st;
            st *= x.Dims[d];
        }
        var map = new long[total];
        var idx = new long[x.Rank];
        for (long o = 0; o < total; o++)
        {
            long off = 0;
            for (int d = 0; d < x.Rank; d++)
            {
                off += pos[d][idx[d]] * srcStr[d];
            }

            map[o] = off;
            for (int d = x.Rank - 1; d >= 0; d--)
            {
                if (++idx[d] < len[d]) { break; } idx[d] = 0;
            }
        }
        return x.IsFloat ? OnnxConst.FromFloats(x.ElemType, len, map.Select(p => x.Floats()[p]).ToArray())
                         : OnnxConst.FromInts(x.ElemType, len, map.Select(p => x.Ints()[p]).ToArray());
    }

    private static IReadOnlyList<OnnxConst> Split(OnnxNode n, OnnxConst x, OnnxConst? splitIn)
    {
        int axis = (int)n.Int("axis", 0);
        if (axis < 0)
        {
            axis += x.Rank;
        }

        var sizes = n.Has("split") ? n.Ints("split")! : splitIn?.Ints();
        int outs = n.Outputs.Count;
        sizes ??= Enumerable.Repeat(x.Dims[axis] / outs, outs).ToArray();
        var result = new List<OnnxConst>();
        long start = 0;
        foreach (var sz in sizes)
        {
            var node = new OnnxNode
            {
                OpType = "Slice",
                Inputs = [],
                Outputs = [],
                Attributes = new Dictionary<string, OnnxAttribute>
                {
                    ["starts"] = new() { Name = "starts", Type = OnnxAttrType.Ints, Ints = [start] },
                    ["ends"] = new() { Name = "ends", Type = OnnxAttrType.Ints, Ints = [start + sz] },
                    ["axes"] = new() { Name = "axes", Type = OnnxAttrType.Ints, Ints = [axis] },
                }
            };
            result.Add(Slice(node, [x]));
            start += sz;
        }
        return result;
    }

    private static OnnxConst Reduce(OnnxNode n, OnnxConst x, OnnxConst? axesIn)
    {
        var axesL = n.Has("axes") ? n.Ints("axes") : axesIn?.Ints();
        var axes = (axesL ?? Enumerable.Range(0, x.Rank).Select(i => (long)i).ToArray()).Select(a => (int)(a < 0 ? a + x.Rank : a)).ToHashSet();
        bool keep = n.Int("keepdims", 1) != 0;
        var outDims = new List<long>();
        var keptDims = new List<long>();
        for (int d = 0; d < x.Rank; d++)
        {
            if (!axes.Contains(d))
            {
                outDims.Add(x.Dims[d]);
                keptDims.Add(x.Dims[d]);
            }
            else if (keep)
            {
                outDims.Add(1);
            }
        }
        var vals = x.Floats();
        long outCount = 1;
        foreach (var d in keptDims)
        {
            outCount *= d;
        }

        var acc = new double[outCount];
        var cnt = new long[outCount];
        string op = n.OpType;
        Array.Fill(acc, op switch { "ReduceProd" => 1.0, "ReduceMax" => double.NegativeInfinity, "ReduceMin" => double.PositiveInfinity, _ => 0.0 });
        var idx = new long[x.Rank];
        for (long i = 0; i < x.Count; i++)
        {
            long o = 0;
            for (int d = 0; d < x.Rank; d++)
            {
                if (!axes.Contains(d))
                {
                    o = o * x.Dims[d] + idx[d];
                }
            }

            double v = vals[i];
            acc[o] = op switch { "ReduceProd" => acc[o] * v, "ReduceMax" => Math.Max(acc[o], v), "ReduceMin" => Math.Min(acc[o], v), _ => acc[o] + v };
            cnt[o]++;
            for (int d = x.Rank - 1; d >= 0; d--)
            {
                if (++idx[d] < x.Dims[d]) { break; } idx[d] = 0;
            }
        }
        if (op == "ReduceMean")
        {
            for (long i = 0; i < outCount; i++)
            {
                acc[i] /= Math.Max(cnt[i], 1);
            }
        }

        return x.IsFloat ? OnnxConst.FromFloats(x.ElemType, [.. outDims], acc) : OnnxConst.FromInts(x.ElemType, [.. outDims], acc.Select(v => (long)v).ToArray());
    }
}
