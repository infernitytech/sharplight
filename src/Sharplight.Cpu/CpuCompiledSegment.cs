using System.Collections;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// A graph segment compiled for the CPU: every value is a (buffer, offset, dims, strides) view, activation
/// buffers are packed into one arena by liveness, weights are used in place (zero-copy when memory mapped).
/// </summary>
public sealed unsafe class CpuCompiledSegment : ICompiledSegment
{
    readonly GraphSegment _segment;
    readonly CompileOptions _options;
    readonly Dictionary<Value, TRef> _refs = [];
    readonly Dictionary<Value, TRef> _inputs = [];
    readonly Dictionary<Value, TRef> _f32Cache = [];
    readonly Dictionary<WeightInfo, Buf> _weightBufs = [];
    readonly List<(CpuOp Op, TRef[] Touches)> _steps = [];
    readonly List<Buf> _bufs = [];
    readonly List<TRef> _allRefs = [];
    readonly List<nint> _owned = [];
    readonly List<(Buf State, long Bytes)> _stateBufs = [];
    readonly Dictionary<string, long> _symBound = new(StringComparer.Ordinal);
    byte* _arena;
    long _arenaBytes, _ownedBytes, _weightBytes;
    RunParameters? _last;
    bool _disposed;

    public GraphSegment Segment => _segment;
    public long DeviceMemoryBytes => _arenaBytes + _ownedBytes + _weightBytes;
    /// <summary>Number of executable steps (view ops are free and not counted).</summary>
    public int StepCount => _steps.Count;
    /// <summary>Bytes of the shared activation arena.</summary>
    public long ArenaBytes => _arenaBytes;
    /// <summary>Optional per-op timing (ticks accumulate per op name) for profiling.</summary>
    public Dictionary<string, (long Ticks, long Calls)>? Profile { get; set; }

    internal CpuCompiledSegment(GraphSegment segment, CompileOptions options, CpuDevice device)
    {
        _segment = segment;
        _options = options;
        try
        {
            foreach (var v in segment.Inputs) Ref(v);
            foreach (var n in segment.Nodes) Emit(n, device);
            foreach (var o in segment.Outputs)
                if (_refs.TryGetValue(o, out var r)) r.Buf.Pinned = true;
            AllocateArena();
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    // ============================== leaves ====================================================
    long Bound(string sym)
    {
        long b = sym == WellKnown.ContextSymbol ? _options.ContextLength
               : sym == WellKnown.TokenSymbol ? _options.MaxTokens
               : _options.SymbolBounds.TryGetValue(sym, out var v) ? v : _options.MaxTokens;
        if (!_symBound.TryGetValue(sym, out var old) || old < b) _symBound[sym] = b;
        return b;
    }

    long[] DimsOf(Shape s)
    {
        var d = new long[s.Rank];
        for (int i = 0; i < d.Length; i++) d[i] = s.Dims[i].IsSymbolic ? Bound(s.Dims[i].Symbol!) : s.Dims[i].Fixed;
        return d;
    }

    TRef Register(TRef r) { _allRefs.Add(r); return r; }

    Buf NewArenaBuf(long bytes, string name)
    {
        var b = new Buf { Bytes = (Math.Max(bytes, 1) + 63) & ~63L, InArena = true, Name = name };
        _bufs.Add(b);
        return b;
    }

    TRef NewDense(Value v)
    {
        if (v.Type.IsQuantized) throw new NotSupportedException($"Quantized activation {v}.");
        var dims = DimsOf(v.Shape);
        long elems = 1; foreach (var d in dims) elems *= d;
        return Register(new TRef
        {
            Type = v.Type, Dims = dims, Strides = TRef.RowMajor(dims), Sym = v.Shape.Symbol,
            Buf = NewArenaBuf(elems * v.Type.BlockBytes, v.Name),
        });
    }

    TRef NewDense(DataType type, long[] dims, string? sym, string name)
    {
        long elems = 1; foreach (var d in dims) elems *= d;
        return Register(new TRef
        {
            Type = type, Dims = dims, Strides = TRef.RowMajor(dims), Sym = sym,
            Buf = NewArenaBuf(elems * type.BlockBytes, name),
        });
    }

    Buf LoadWeight(WeightInfo w)
    {
        if (_weightBufs.TryGetValue(w, out var cached)) return cached;
        long bytes = w.ByteSize;
        var src = w.Source;
        var buf = new Buf { Bytes = bytes, Name = w.Name };
        nint ptr = src.TryGetPointer();
        if (ptr != 0) buf.Ptr = (byte*)ptr;
        else
        {
            buf.Ptr = Mem.Alloc(bytes, false);
            _owned.Add((nint)buf.Ptr);
            _ownedBytes += bytes;
            const long chunk = 1L << 30;
            for (long off = 0; off < bytes; off += chunk)
                src.Read(off, new Span<byte>(buf.Ptr + off, (int)Math.Min(chunk, bytes - off)));
        }
        _weightBytes += bytes;
        _weightBufs[w] = buf;
        return buf;
    }

    /// <summary>The value as seen by an op: weights/constants keep their raw storage type.</summary>
    TRef Ref(Value v)
    {
        if (_refs.TryGetValue(v, out var r)) return r;
        switch (v.Kind)
        {
            case ValueKind.Weight:
            {
                var dims = DimsOf(v.Shape);
                r = Register(new TRef { Type = v.Type, Dims = dims, Strides = TRef.RowMajor(dims), Buf = LoadWeight(v.Weight!) });
                break;
            }
            case ValueKind.Constant:
            {
                var data = v.ConstantData!;
                var buf = new Buf { Bytes = data.Length, Ptr = Mem.Alloc(data.Length, false), Name = v.Name };
                _owned.Add((nint)buf.Ptr); _ownedBytes += data.Length;
                data.CopyTo(new Span<byte>(buf.Ptr, data.Length));
                var dims = DimsOf(v.Shape);
                r = Register(new TRef { Type = v.Type, Dims = dims, Strides = TRef.RowMajor(dims), Buf = buf });
                break;
            }
            case ValueKind.State:
            {
                var def = v.State!;
                var dims = DimsOf(def.Shape);
                long elems = 1; foreach (var d in dims) elems *= d;
                long bytes = def.Type.ByteSize(elems);
                var buf = new Buf { Bytes = bytes, Ptr = Mem.Alloc(bytes, true), Name = def.Name };
                _owned.Add((nint)buf.Ptr); _ownedBytes += bytes;
                _stateBufs.Add((buf, bytes));
                r = Register(new TRef { Type = def.Type, Dims = dims, Strides = TRef.RowMajor(dims), Buf = buf });
                break;
            }
            default:
                // Input, or Intermediate produced outside this segment: dense buffer filled by SetInput.
                if (v.Kind == ValueKind.Intermediate && v.Producer is { } prod && _segment.Nodes.Contains(prod))
                    throw new InvalidOperationException($"Value {v} used before it is produced.");
                r = NewDense(v);
                r.Buf.Pinned = true;
                _inputs[v] = r;
                break;
        }
        _refs[v] = r;
        return r;
    }

    /// <summary>Like <see cref="Ref"/> but weights/constants are converted to dense f32.</summary>
    TRef RefF32(Value v)
    {
        var r = Ref(v);
        if (v.Kind is not (ValueKind.Weight or ValueKind.Constant) || r.Type == DataType.F32 || r.Type.Kind is DataTypeKind.Int or DataTypeKind.Bool) return r;
        if (_f32Cache.TryGetValue(v, out var c)) return c;
        var fmt = Quant.FormatOf(r.Type) ?? throw new NotSupportedException($"Weight type {r.Type} cannot be converted to f32.");
        long elems = r.MaxElements;
        var buf = new Buf { Bytes = elems * 4, Ptr = Mem.Alloc(elems * 4, false), Name = v.Name + ".f32" };
        _owned.Add((nint)buf.Ptr); _ownedBytes += elems * 4;
        Quant.Dequantize(fmt, r.Buf.Ptr, elems, (float*)buf.Ptr);
        c = Register(new TRef { Type = DataType.F32, Dims = r.Dims, Strides = TRef.RowMajor(r.Dims), Buf = buf });
        _f32Cache[v] = c;
        return c;
    }

    // ============================== steps =====================================================
    void AddStep(CpuOp op, params TRef[] touches)
    {
        int i = _steps.Count;
        _steps.Add((op, touches));
        foreach (var t in touches)
        {
            if (!t.Buf.InArena) continue;
            if (i < t.Buf.First) t.Buf.First = i;
            if (i > t.Buf.Last) t.Buf.Last = i;
        }
    }

    TRef Materialize(TRef x)
    {
        var y = NewDense(x.Type, (long[])x.Dims.Clone(), x.Sym, "materialized");
        AddStep(new CopyOp(x, y), x, y);
        return y;
    }

    TRef RowContig(TRef x) => x.LastContiguous ? x : Materialize(x);
    TRef Dense(TRef x) => x.IsDense ? x : Materialize(x);

    TRef View(TRef src, long[] dims, long[] strides, long offset, string? sym) => Register(new TRef
    {
        Type = src.Type, Dims = dims, Strides = strides, Sym = sym, Buf = src.Buf, Offset = src.Offset + offset,
    });

    static bool TryReshapeStrides(long[] od, long[] os, long[] nd, out long[] ns)
    {
        ns = new long[nd.Length];
        var oldD = new List<long>(); var oldS = new List<long>();
        for (int i = 0; i < od.Length; i++) if (od[i] != 1) { oldD.Add(od[i]); oldS.Add(os[i]); }
        int oldnd = oldD.Count;
        int oi = 0, oj = 1, ni = 0, nj = 1;
        while (ni < nd.Length && oi < oldnd)
        {
            long np = nd[ni], op = oldD[oi];
            while (np != op)
            {
                if (np < op) { if (nj >= nd.Length) return false; np *= nd[nj++]; }
                else { if (oj >= oldnd) return false; op *= oldD[oj++]; }
            }
            for (int ok = oi; ok < oj - 1; ok++)
                if (oldS[ok] != oldD[ok + 1] * oldS[ok + 1]) return false;
            ns[nj - 1] = oldS[oj - 1];
            for (int nk = nj - 1; nk > ni; nk--) ns[nk - 1] = ns[nk] * nd[nk];
            ni = nj++; oi = oj++;
        }
        long last = ni >= 1 ? ns[ni - 1] : 1;
        for (int nk = ni; nk < nd.Length; nk++) ns[nk] = last;
        return true;
    }

    TRef ReshapeView(TRef s, Value outV)
    {
        var nd = DimsOf(outV.Shape);
        string? sym = outV.Shape.Symbol;
        if (s.IsDense) return View(s, nd, TRef.RowMajor(nd), 0, sym);
        long[] ns;
        if (s.Sym is not null && sym is not null)
        {
            if (TryReshapeStrides(s.Dims[1..], s.Strides[1..], nd[1..], out var inner))
            {
                ns = new long[nd.Length]; ns[0] = s.Strides[0]; inner.CopyTo(ns, 1);
                return View(s, nd, ns, 0, sym);
            }
        }
        else if (s.Sym is null && sym is null && TryReshapeStrides(s.Dims, s.Strides, nd, out ns))
            return View(s, nd, ns, 0, sym);
        var m = Materialize(s);
        return View(m, nd, TRef.RowMajor(nd), 0, sym);
    }

    static int Norm(long axis, int rank) => (int)(axis < 0 ? axis + rank : axis);

    static int[] Ints(Attributes a, string key, params int[] dflt)
    {
        var r = IntArray(a, key);
        return r.Length > 0 ? r : dflt;
    }

    static int[] IntArray(Attributes a, string key)
    {
        if (!a.TryGetValue(key, out var o)) return [];
        return o switch
        {
            int[] i => i,
            long[] l => l.Select(x => (int)x).ToArray(),
            IEnumerable e => e.Cast<object>().Select(x => System.Convert.ToInt32(x)).ToArray(),
            _ => [],
        };
    }

    static readonly Dictionary<string, UnaryKind> s_unary = Enum.GetValues<UnaryKind>().Where(k => k is not (UnaryKind.Scale or UnaryKind.Clip)).ToDictionary(k => k.ToString(), StringComparer.Ordinal);

    void Emit(Node n, CpuDevice device)
    {
        string op = n.Op.Name;
        var a = n.Attributes;
        var ins = n.Inputs;
        if (n.Op.Category != OpCategory.View)
        {
            var sup = device.Supports(n);
            if (!sup.IsSupported) throw new NotSupportedException($"CPU provider cannot run {op} (node {n.Id}): {sup.Reason}");
        }
        switch (op)
        {
            case "Reshape": _refs[n.Output] = ReshapeView(Ref(ins[0]), n.Output); return;
            case "Slice":
            {
                var s = Ref(ins[0]);
                int axis = a.Get<int>("axis"); if (axis < 0) axis += s.Rank;
                long start = a.Get<long>("start");
                var dims = (long[])s.Dims.Clone(); dims[axis] = a.Get<long>("length");
                _refs[n.Output] = View(s, dims, s.Strides, start * s.Strides[axis], s.Sym);
                return;
            }
            case "Permute":
            {
                var s = Ref(ins[0]);
                var perm = a.Get<int[]>("perm");
                var dims = perm.Select(p => s.Dims[p]).ToArray();
                var strides = perm.Select(p => s.Strides[p]).ToArray();
                _refs[n.Output] = View(s, dims, strides, 0, perm[0] == 0 ? s.Sym : null);
                return;
            }
        }

        if (s_unary.TryGetValue(op, out var uk))
        {
            var x = RowContig(RefF32(ins[0])); var y = NewDense(n.Output);
            UnaryOp uop = uk switch
            {
                UnaryKind.HardSigmoid => new UnaryOp(uk, x, y, a.Get("alpha", 0.2f), a.Get("beta", 0.5f)),
                UnaryKind.Elu or UnaryKind.Celu or UnaryKind.ThresholdedRelu => new UnaryOp(uk, x, y, a.Get("alpha", 1f)),
                UnaryKind.Selu => new UnaryOp(uk, x, y, a.Get("alpha", 1.67326319217681884765625f), a.Get("gamma", 1.05070102214813232421875f)),
                UnaryKind.LeakyRelu => new UnaryOp(uk, x, y, a.Get("alpha", 0.01f)),
                _ => new UnaryOp(uk, x, y),
            };
            AddStep(uop, x, y);
            _refs[n.Output] = y; return;
        }
        switch (op)
        {
            case "Clip":
            {
                var x = RowContig(RefF32(ins[0])); var y = NewDense(n.Output);
                AddStep(new UnaryOp(UnaryKind.Clip, x, y, a.Get("min", float.NegativeInfinity), a.Get("max", float.PositiveInfinity)), x, y);
                _refs[n.Output] = y; return;
            }
            case "Conv":
            {
                var x = Dense(RefF32(ins[0])); var w = RefF32(ins[1]);
                TRef? bias = ins.Length > 2 ? Dense(RefF32(ins[2])) : null;
                var y = NewDense(n.Output);
                var conv = ConvOp.Create(x, (float*)(w.Buf.Ptr + w.Offset * 4), (int)w.Dims[2], (int)w.Dims[3], bias, y, (int)a.Get("groups", 1L),
                    Ints(a, "strides", 1, 1), Ints(a, "pads", 0, 0, 0, 0), Ints(a, "dilations", 1, 1));
                AddStep(conv, bias is null ? [x, y] : [x, bias, y]); _refs[n.Output] = y; return;
            }
            case "Pool":
            {
                var x = Dense(RefF32(ins[0])); var y = NewDense(n.Output);
                AddStep(new PoolOp(x, y, a.Get("mode", "max") == "max", Ints(a, "kernel"), Ints(a, "strides", 1, 1), Ints(a, "pads", 0, 0, 0, 0),
                    Ints(a, "dilations", 1, 1), a.Get("count_include_pad", false)), x, y);
                _refs[n.Output] = y; return;
            }
            case "Reduce":
            {
                var x = RefF32(ins[0]); var y = NewDense(n.Output);
                var red = new bool[x.Rank];
                foreach (var ax in a.Get<long[]>("axes")) red[ax < 0 ? ax + x.Rank : ax] = true;
                var kind = a.Get("mode", "sum") switch { "mean" => ReduceKind.Mean, "max" => ReduceKind.Max, "min" => ReduceKind.Min, "prod" => ReduceKind.Prod, "l1" => ReduceKind.L1, "l2" => ReduceKind.L2, "sumsquare" => ReduceKind.SumSquare, "logsumexp" => ReduceKind.LogSumExp, _ => ReduceKind.Sum };
                AddStep(new ReduceOp(kind, x, y, red), x, y); _refs[n.Output] = y; return;
            }
            case "BatchMatMul":
            {
                var x = RefF32(ins[0]); var z = RefF32(ins[1]); var y = NewDense(n.Output);
                AddStep(new BatchMatMulOp(x, z, y, a.Get("trans_a", false), a.Get("trans_b", false), a.Get("alpha", 1f)), x, z, y);
                _refs[n.Output] = y; return;
            }
            case "Pad":
            {
                var x = RefF32(ins[0]); var y = NewDense(n.Output);
                AddStep(new PadOp(x, y, a.Get<long[]>("pads"), a.Get("mode", "constant"), a.Get("value", 0f)), x, y);
                _refs[n.Output] = y; return;
            }
            case "Resize":
            {
                var x = Dense(RefF32(ins[0])); var y = NewDense(n.Output);
                AddStep(new ResizeOp(x, y, a.Get("mode", "nearest"), a.Get("coord", "half_pixel"), a.Get("nearest_mode", "round_prefer_floor")), x, y);
                _refs[n.Output] = y; return;
            }
            case "Where":
            {
                var cnd = Ref(ins[0]); var x = Ref(ins[1]); var z = Ref(ins[2]); var y = NewDense(n.Output);
                AddStep(new WhereOp(cnd, x, z, y), cnd, x, z, y); _refs[n.Output] = y; return;
            }
            case "Expand":
            {
                var x = Ref(ins[0]); var y = NewDense(n.Output);
                int r = y.Rank; var strides = new long[r];
                for (int k = 0; k < r; k++)
                {
                    int ik = k - (r - x.Rank);
                    strides[k] = ik < 0 ? 0 : (x.Dims[ik] == 1 && !(ik == 0 && x.Sym is not null) && y.Dims[k] != 1) ? 0 : x.Strides[ik];
                }
                var view = View(x, (long[])y.Dims.Clone(), strides, 0, y.Sym);
                AddStep(new CopyOp(view, y), x, y); _refs[n.Output] = y; return;
            }
            case "Tile":
            {
                var x = Ref(ins[0]); var y = NewDense(n.Output);
                var rep = a.Get<long[]>("repeats");
                var vd = new List<long>(); var vs = new List<long>(); var yd = new List<long>();
                for (int k = 0; k < x.Rank; k++)
                {
                    if (k == 0 && x.Sym is not null) { vd.Add(x.Dims[0]); vs.Add(x.Strides[0]); yd.Add(y.Dims[0]); continue; }
                    vd.Add(rep[k]); vs.Add(0); yd.Add(rep[k]);
                    vd.Add(x.Dims[k]); vs.Add(x.Strides[k]); yd.Add(x.Dims[k]);
                }
                var view = View(x, vd.ToArray(), vs.ToArray(), 0, x.Sym);
                var yview = View(y, yd.ToArray(), TRef.RowMajor(yd.ToArray()), 0, y.Sym);
                AddStep(new CopyOp(view, yview), x, y); _refs[n.Output] = y; return;
            }
            case "GatherAxis":
            {
                var x = Dense(Ref(ins[0])); var idx = Dense(Ref(ins[1])); var y = NewDense(n.Output);
                AddStep(new GatherAxisOp(x, idx, y, Norm(a.Get("axis", 0L), x.Rank)), x, idx, y); _refs[n.Output] = y; return;
            }
            case "GatherElements":
            {
                var x = Dense(Ref(ins[0])); var idx = Dense(Ref(ins[1])); var y = NewDense(n.Output);
                AddStep(new GatherElementsOp(x, idx, y, Norm(a.Get("axis", 0L), x.Rank)), x, idx, y); _refs[n.Output] = y; return;
            }
            case "GatherND":
            {
                var x = Dense(Ref(ins[0])); var idx = Dense(Ref(ins[1])); var y = NewDense(n.Output);
                AddStep(new GatherNdOp(x, idx, y, (int)a.Get("batch_dims", 0L)), x, idx, y); _refs[n.Output] = y; return;
            }
            case "ScatterElements" or "ScatterND":
            {
                var x = Dense(Ref(ins[0])); var idx = Dense(Ref(ins[1])); var upd = Dense(Ref(ins[2])); var y = NewDense(n.Output);
                AddStep(new ScatterOp(x, idx, upd, y, op == "ScatterND" ? 0 : Norm(a.Get("axis", 0L), x.Rank), op == "ScatterND", a.Get("reduction", "none")), x, idx, upd, y);
                _refs[n.Output] = y; return;
            }
            case "CumSum":
            {
                var x = Dense(RefF32(ins[0])); var y = NewDense(n.Output);
                AddStep(new CumSumOp(x, y, Norm(a.Get("axis", 0L), x.Rank), a.Get("exclusive", false), a.Get("reverse", false)), x, y); _refs[n.Output] = y; return;
            }
            case "ArgReduce":
            {
                var x = Dense(RefF32(ins[0])); var y = NewDense(n.Output);
                AddStep(new ArgReduceOp(x, y, Norm(a.Get("axis", 0L), x.Rank), a.Get("mode", "max") == "max", a.Get("select_last_index", false)), x, y); _refs[n.Output] = y; return;
            }
            case "TopK":
            {
                var x = Dense(RefF32(ins[0])); var v = NewDense(n.Outputs[0]); var i = NewDense(n.Outputs[1]);
                AddStep(new TopKOp(x, v, i, Norm(a.Get("axis", -1L), x.Rank), (int)a.Get<long>("k"), a.Get("largest", true)), x, v, i);
                _refs[n.Outputs[0]] = v; _refs[n.Outputs[1]] = i; return;
            }
            case "GroupNorm":
            {
                var x = Dense(RefF32(ins[0]));
                TRef? w = ins.Length > 1 ? Dense(RefF32(ins[1])) : null, b = ins.Length > 2 ? Dense(RefF32(ins[2])) : null;
                var y = NewDense(n.Output);
                var touches = new List<TRef> { x, y }; if (w != null) touches.Add(w); if (b != null) touches.Add(b);
                AddStep(new GroupNormOp(x, w, b, y, (int)a.Get<long>("groups"), a.Get("eps", 1e-5f)), [.. touches]); _refs[n.Output] = y; return;
            }
            case "ConvTranspose":
            {
                var x = Dense(RefF32(ins[0])); var w = RefF32(ins[1]);
                TRef? bias = ins.Length > 2 ? Dense(RefF32(ins[2])) : null;
                var y = NewDense(n.Output);
                var st = Ints(a, "strides", 1, 1); var pd = Ints(a, "pads", 0, 0, 0, 0); var dl = Ints(a, "dilations", 1, 1);
                var op2 = new ConvTransposeOp(x, (float*)(w.Buf.Ptr + w.Offset * 4), (int)w.Dims[2], (int)w.Dims[3], bias, y, (int)a.Get("groups", 1L), st[0], st[1], pd[0], pd[1], dl[0], dl[1]);
                AddStep(op2, bias is null ? [x, y] : [x, bias, y]); _refs[n.Output] = y; return;
            }
            case "Trilu":
            {
                var x = Dense(Ref(ins[0])); var y = NewDense(n.Output);
                AddStep(new TriluOp(x, y, a.Get("upper", true), a.Get("k", 0L)), x, y); _refs[n.Output] = y; return;
            }
            case "OneHot":
            {
                var idx = Dense(Ref(ins[0])); var y = NewDense(n.Output);
                AddStep(new OneHotOp(idx, y, Norm(a.Get("axis", -1L), idx.Rank + 1), a.Get<long>("depth"), a.Get("on", 1f), a.Get("off", 0f)), idx, y); _refs[n.Output] = y; return;
            }
            case "GridSample":
            {
                var x = Dense(RefF32(ins[0])); var grid = Dense(RefF32(ins[1])); var y = NewDense(n.Output);
                AddStep(new GridSampleOp(x, grid, y, a.Get("mode", "linear"), a.Get("padding", "zeros"), a.Get("align_corners", false)), x, grid, y);
                _refs[n.Output] = y; return;
            }
            case "Scale":
            {
                var x = RowContig(RefF32(ins[0])); var y = NewDense(n.Output);
                AddStep(new UnaryOp(UnaryKind.Scale, x, y, a.Get("scale", 1f), a.Get("bias", 0f)), x, y); _refs[n.Output] = y; return;
            }
            case "Add" or "Sub" or "Mul" or "Div" or "Max" or "Min" or "Pow" or "Mod" or "PRelu" or "Equal" or "NotEqual" or "Less" or "LessEqual" or "Greater" or "GreaterEqual" or "And" or "Or" or "Xor":
            {
                var x = RefF32(ins[0]); var z = RefF32(ins[1]); var y = NewDense(n.Output);
                AddStep(new BinaryOp(Enum.Parse<BinaryKind>(op), x, z, y, a.Get("fmod", false)), x, z, y); _refs[n.Output] = y; return;
            }
            case "Cast":
            {
                var x = RowContig(Ref(ins[0])); var y = NewDense(n.Output);
                AddStep(new CastOp(x, y), x, y); _refs[n.Output] = y; return;
            }
            case "Contiguous" or "Transfer":
            {
                var x = Ref(ins[0]); var y = NewDense(n.Output);
                AddStep(new CopyOp(x, y), x, y); _refs[n.Output] = y; return;
            }
            case "Concat":
            {
                int axis = a.Get<int>("axis"); if (axis < 0) axis += n.Output.Rank;
                var xs = ins.Select(Ref).ToArray(); var y = NewDense(n.Output);
                AddStep(new ConcatOp(xs, y, axis), [.. xs, y]); _refs[n.Output] = y; return;
            }
            case "Gather":
            {
                var t = Ref(ins[0]); var idx = Ref(ins[1]); var y = NewDense(n.Output);
                var fmt = Quant.FormatOf(t.Type)!.Value;
                if (ins[0].Kind is not (ValueKind.Weight or ValueKind.Constant)) t = Dense(t);
                AddStep(new GatherOp(t, idx, y, fmt), t, idx, y); _refs[n.Output] = y; return;
            }
            case "MatMul":
            {
                var x = RowContig(RefF32(ins[0])); var w = Ref(ins[1]);
                if (ins[1].Kind is not (ValueKind.Weight or ValueKind.Constant)) w = Dense(w);
                var fmt = Quant.FormatOf(w.Type)!.Value;
                var y = NewDense(n.Output);
                AddStep(new MatMulOp(x, w, y, fmt), x, w, y); _refs[n.Output] = y; return;
            }
            case "MatMulId":
            {
                var x = RowContig(RefF32(ins[0])); var w = Ref(ins[1]); var ix = Dense(Ref(ins[2]));
                if (ins[1].Kind is not (ValueKind.Weight or ValueKind.Constant)) w = Dense(w);
                var fmt = Quant.FormatOf(w.Type)!.Value;
                var y = NewDense(n.Output);
                AddStep(new MatMulIdOp(x, w, ix, y, fmt), x, w, ix, y); _refs[n.Output] = y; return;
            }
            case "RmsNorm" or "L2Norm" or "LayerNorm" or "Softmax" or "ReduceSum" or "ReduceMax" or "ArgMax":
            {
                var kind = Enum.Parse<RowKind>(op);
                var x = RowContig(RefF32(ins[0]));
                TRef? w = ins.Length > 1 ? RefF32(ins[1]) : null, b = ins.Length > 2 ? RefF32(ins[2]) : null;
                var y = NewDense(n.Output);
                var touches = new List<TRef> { x, y }; if (w != null) touches.Add(w); if (b != null) touches.Add(b);
                AddStep(new RowOp(kind, x, w, b, y, a.Get("eps", 1e-5f), a.Get("scale", 1f), a.Get("log", false)), [.. touches]); _refs[n.Output] = y; return;
            }
            case "Rope":
            {
                var x = RowContig(RefF32(ins[0])); var pos = Ref(ins[1]);
                TRef? ff = ins.Length > 2 ? RefF32(ins[2]) : null;
                var y = NewDense(n.Output);
                var rope = new RopeOp(x, pos, ff, y, a.Get("dims", 0), a.Get("base", 10000f), a.Get("style", "neox"),
                    IntArray(a, "sections"), a.Get("interleaved", false), a.Get("freq_scale", 1f));
                AddStep(rope, ff is null ? [x, pos, y] : [x, pos, ff, y]); _refs[n.Output] = y; return;
            }
            case "CacheWrite":
            {
                var st = Ref(ins[0]); var x = RowContig(RefF32(ins[1]));
                AddStep(new CacheWriteOp(st, x, a.Get("offset", WellKnown.PastScalar)), st, x); return;
            }
            case "CachedAttention":
            {
                var q = RowContig(Ref(ins[0])); var k = Ref(ins[1]); var v = Ref(ins[2]); var y = NewDense(n.Output);
                AddStep(new AttentionOp(q, k, v, null, y, a.Get("scale", 1f), a.Get("offset", WellKnown.PastScalar), true, a.Get("window", 0), a.Get("softcap", 0f)), q, k, v, y);
                _refs[n.Output] = y; return;
            }
            case "Attention":
            {
                var q = RowContig(Ref(ins[0])); var k = RowContig(Ref(ins[1])); var v = RowContig(Ref(ins[2]));
                TRef? mask = ins.Length > 3 ? Ref(ins[3]) : null;
                var y = NewDense(n.Output);
                AddStep(new AttentionOp(q, k, v, mask, y, a.Get("scale", 1f), null, a.Get("causal", false)),
                    mask is null ? [q, k, v, y] : [q, k, v, mask, y]);
                _refs[n.Output] = y; return;
            }
            case "CausalConv1d":
            {
                var x = RowContig(RefF32(ins[0])); var w = RefF32(ins[1]); var st = Ref(ins[2]); var y = NewDense(n.Output);
                bool stat = ins[1].Kind is ValueKind.Weight or ValueKind.Constant;
                if (!stat) w = Dense(w);
                var cb = ins.Length > 3 ? Dense(RefF32(ins[3])) : null;
                AddStep(new CausalConv1dOp(x, w, st, y, a.Get("activation", "none") == "silu", stat, cb), cb is null ? [x, w, st, y] : [x, w, st, y, cb]);
                _refs[n.Output] = y; return;
            }
            case "SelectiveScan":
            {
                var x = RowContig(Ref(ins[0])); var dt = RowContig(Ref(ins[1])); var bi = Dense(RefF32(ins[2])); var A = Dense(RefF32(ins[3]));
                var Bm = RowContig(Ref(ins[4])); var Cm = RowContig(Ref(ins[5])); var D = Dense(RefF32(ins[6])); var st = Ref(ins[7]); var y = NewDense(n.Output);
                AddStep(new SelectiveScanOp(x, dt, bi, A, Bm, Cm, D, st, y, a.Get("dt_min", 0f)), x, dt, bi, A, Bm, Cm, D, st, y);
                _refs[n.Output] = y; return;
            }
            case "GatedDeltaRule":
            {
                var q = RowContig(Ref(ins[0])); var k = RowContig(Ref(ins[1])); var v = RowContig(Ref(ins[2]));
                var g = Ref(ins[3]); var beta = Ref(ins[4]); var st = Ref(ins[5]); var y = NewDense(n.Output);
                AddStep(new GatedDeltaRuleOp(q, k, v, g, beta, st, y, a.Get("scale", 1f)), q, k, v, g, beta, st, y);
                _refs[n.Output] = y; return;
            }
        }
        throw new NotSupportedException($"Op {op} is not implemented by the CPU provider.");
    }

    // ============================== arena =====================================================
    void AllocateArena()
    {
        int end = _steps.Count;
        foreach (var b in _bufs)
        {
            if (b.Pinned) { b.First = 0; b.Last = end; }
            else if (b.First == int.MaxValue) { b.First = 0; b.Last = 0; }
        }
        var order = _bufs.OrderBy(b => b.First).ToList();
        var free = new List<(long Off, long Size)>();
        long top = 0;
        int next = 0;
        var active = new List<Buf>();
        for (int step = 0; step <= end; step++)
        {
            while (next < order.Count && order[next].First == step)
            {
                var b = order[next++];
                int best = -1;
                for (int i = 0; i < free.Count; i++)
                    if (free[i].Size >= b.Bytes && (best < 0 || free[i].Size < free[best].Size)) best = i;
                if (best >= 0)
                {
                    b.ArenaOffset = free[best].Off;
                    long rest = free[best].Size - b.Bytes;
                    if (rest > 0) free[best] = (free[best].Off + b.Bytes, rest); else free.RemoveAt(best);
                }
                else if (free.Count > 0 && free[^1].Off + free[^1].Size == top)
                {
                    b.ArenaOffset = free[^1].Off;
                    top = b.ArenaOffset + b.Bytes;
                    free.RemoveAt(free.Count - 1);
                }
                else { b.ArenaOffset = top; top += b.Bytes; }
                active.Add(b);
            }
            for (int i = active.Count - 1; i >= 0; i--)
            {
                var b = active[i];
                if (b.Last > step) continue;
                active.RemoveAt(i);
                if (b.Pinned) continue;
                int pos = free.FindIndex(f => f.Off > b.ArenaOffset);
                if (pos < 0) pos = free.Count;
                free.Insert(pos, (b.ArenaOffset, b.Bytes));
                // coalesce
                if (pos + 1 < free.Count && free[pos].Off + free[pos].Size == free[pos + 1].Off)
                { free[pos] = (free[pos].Off, free[pos].Size + free[pos + 1].Size); free.RemoveAt(pos + 1); }
                if (pos > 0 && free[pos - 1].Off + free[pos - 1].Size == free[pos].Off)
                { free[pos - 1] = (free[pos - 1].Off, free[pos - 1].Size + free[pos].Size); free.RemoveAt(pos); }
            }
        }
        _arenaBytes = top;
        _arena = Mem.Alloc(Math.Max(top, 64), true);
        foreach (var b in _bufs) b.Ptr = _arena + b.ArenaOffset;
        foreach (var r in _allRefs) r.P = r.Buf.Ptr + r.Offset * r.ElemSize;
    }

    // ============================== ICompiledSegment ==========================================
    public void SetInput(Value value, ReadOnlySpan<byte> data)
    {
        if (!_inputs.TryGetValue(value, out var r)) throw new ArgumentException($"{value} is not an input of this segment.");
        if (data.Length > r.Buf.Bytes) throw new ArgumentException($"Input {value.Name}: {data.Length} bytes exceed the compiled capacity of {r.Buf.Bytes} bytes (MaxTokens too small?).");
        data.CopyTo(new Span<byte>(r.P, data.Length));
    }

    public void Execute(RunParameters parameters)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        foreach (var (sym, bound) in _symBound)
            if (parameters.Symbols.TryGetValue(sym, out var v) && v > bound)
                throw new InvalidOperationException($"Symbol '{sym}' = {v} exceeds the compiled bound {bound}.");
        _last = parameters;
        var prof = Profile;
        if (prof is null)
        {
            foreach (var (op, _) in _steps) op.Run(parameters);
        }
        else
        {
            foreach (var (op, _) in _steps)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                op.Run(parameters);
                long dt = System.Diagnostics.Stopwatch.GetTimestamp() - t0;
                prof.TryGetValue(op.Name, out var cur);
                prof[op.Name] = (cur.Ticks + dt, cur.Calls + 1);
            }
        }
    }

    public void GetOutput(Value value, Span<byte> destination)
    {
        if (!_refs.TryGetValue(value, out var r)) throw new ArgumentException($"{value} is not a value of this segment.");
        var p = _last ?? throw new InvalidOperationException("GetOutput called before Execute.");
        long elems = r.Elements(p);
        long bytes = elems * r.ElemSize;
        if (destination.Length < bytes) throw new ArgumentException($"Destination too small: need {bytes} bytes, got {destination.Length}.");
        if (elems == 0) return;
        var dims = new long[r.Rank];
        for (int i = 0; i < dims.Length; i++) dims[i] = r.Dim(i, p);
        fixed (byte* d = destination)
            Mem.CopyStrided(r.P, r.Strides, d, TRef.RowMajor(dims), r.Dims, r.Rank == 0 ? 1 : dims[0], r.ElemSize);
    }

    /// <summary>Overwrites a state tensor with host data (f32 states; used by tests and for state restore).</summary>
    public void TestSetState(Value state, float[] data)
    {
        var r = _refs[state];
        if (r.Type != DataType.F32) throw new NotSupportedException("f32 states only");
        data.AsSpan().CopyTo(new Span<float>(r.P, data.Length));
    }

    /// <summary>Reads back an f32 state tensor.</summary>
    public float[] TestGetState(Value state)
    {
        var r = _refs[state];
        var a = new float[r.MaxElements];
        new Span<float>(r.P, a.Length).CopyTo(a);
        return a;
    }

    public void ResetState()
    {
        foreach (var (buf, bytes) in _stateBufs) System.Runtime.InteropServices.NativeMemory.Clear(buf.Ptr, (nuint)bytes);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var (op, _) in _steps) op.Dispose();
        Mem.Free(_arena); _arena = null;
        foreach (var p in _owned) Mem.Free((byte*)p);
        _owned.Clear();
        _bufs.Clear(); _steps.Clear();
    }
}
