using System.Numerics;
using System.Runtime.Intrinsics;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Shared helpers of the general-purpose data-movement ops (dense inputs; element-size generic copies).</summary>
internal static unsafe class Gen
{
    public static void Cp(byte* d, byte* s, int es)
    {
        switch (es)
        {
            case 4: *(int*)d = *(int*)s; break;
            case 8: *(long*)d = *(long*)s; break;
            case 2: *(short*)d = *(short*)s; break;
            default: *d = *s; break;
        }
    }

    /// <summary>Reads element <paramref name="i"/> of a dense i32/i64 index tensor.</summary>
    public static long Idx(TRef t, long i) => t.Type == DataType.I64 ? ((long*)t.P)[i] : ((int*)t.P)[i];

    public static long[] Dims(TRef t, RunParameters p)
    {
        var d = new long[t.Rank];
        for (int i = 0; i < d.Length; i++)
        {
            d[i] = t.Dim(i, p);
        }

        return d;
    }

    public static long Prod(long[] d, int from, int to)
    {
        long n = 1;
        for (int i = from; i < to; i++)
        {
            n *= d[i];
        }

        return n;
    }

    public static long[] Strides(long[] d)
    {
        var s = new long[d.Length]; long e = 1;
        for (int i = d.Length - 1; i >= 0; i--) { s[i] = e; e *= d[i]; }
        return s;
    }

    public static long Wrap(long id, long n, string what)
    {
        if (id < 0)
        {
            id += n;
        }

        if ((ulong)id >= (ulong)n)
        {
            throw new ArgumentOutOfRangeException("indices", $"{what} index {id} out of range [0,{n}).");
        }

        return id;
    }

    public static (int chunks, long per) Split(long n, long work)
    {
        int c = (int)Math.Max(1, Math.Min(n, Par.Chunks(n * Math.Max(1, work), 8192)));
        return (c, (n + c - 1) / c);
    }

    /// <summary>Broadcast element strides of t aligned to the output rank (0 on broadcast axes).</summary>
    public static long[] Align(TRef t, int outRank)
    {
        var s = new long[outRank];
        for (int k = 0; k < outRank; k++)
        {
            int ik = k - (outRank - t.Rank);
            if (ik < 0)
            {
                s[k] = 0;
            }
            else if (t.Dims[ik] == 1 && !(ik == 0 && t.Sym is not null))
            {
                s[k] = 0;
            }
            else
            {
                s[k] = t.Strides[ik];
            }
        }
        return s;
    }
}

/// <summary>y = cond ? a : b with broadcasting; any plain element type for a/b (copied by element size).</summary>
internal sealed unsafe class WhereOp : CpuOp
{
    private readonly TRef _c, _a, _b, _y; private readonly long[] _sc, _sa, _sb; private readonly Action<int> _body;
    private long _outer, _per, _l;
    public override string Name => "Where";

    public WhereOp(TRef c, TRef a, TRef b, TRef y)
    {
        _c = c; _a = a; _b = b; _y = y; _body = Body;
        _sc = Gen.Align(c, y.Rank); _sa = Gen.Align(a, y.Rank); _sb = Gen.Align(b, y.Rank);
    }

    private bool Cond(long off)
    {
        var t = _c.Type;
        if (t == DataType.F32)
        {
            return ((float*)_c.P)[off] != 0f;
        }

        if (t == DataType.I32)
        {
            return ((int*)_c.P)[off] != 0;
        }

        if (t == DataType.I64)
        {
            return ((long*)_c.P)[off] != 0;
        }

        return _c.P[off] != 0;
    }

    public override void Run(RunParameters p)
    {
        long total = _y.Elements(p);
        if (total == 0)
        {
            return;
        }

        _l = _y.Cols(p); _outer = total / _l;
        var (ch, per) = Gen.Split(_outer, _l); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int r = _y.Rank, es = _y.ElemSize;
        long s = chunk * _per, e = Math.Min(_outer, s + _per);
        for (long row = s; row < e; row++)
        {
            long oc = 0, oa = 0, ob = 0, rr = row;
            for (int k = r - 2; k >= 0; k--)
            {
                long d = k == 0 && _y.Sym is not null ? _y.Dims[0] : _y.Dims[k];
                long ix = rr % d; rr /= d;
                oc += ix * _sc[k]; oa += ix * _sa[k]; ob += ix * _sb[k];
            }
            long sc = _sc[r - 1], sa = _sa[r - 1], sb = _sb[r - 1];
            byte* y = _y.P + row * _l * es;
            for (long i = 0; i < _l; i++)
            {
                Gen.Cp(y + i * es, (Cond(oc + i * sc) ? _a.P + (oa + i * sa) * es : _b.P + (ob + i * sb) * es), es);
            }
        }
    }
}

/// <summary>numpy take along an axis: out[o, q, i] = data[o, idx[q], i].</summary>
internal sealed unsafe class GatherAxisOp : CpuOp
{
    private readonly TRef _x, _idx, _y; private readonly int _axis; private readonly Action<int> _body;
    private long _outer, _a, _inner, _q, _per;
    public override string Name => "GatherAxis";
    public GatherAxisOp(TRef x, TRef idx, TRef y, int axis) { _x = x; _idx = idx; _y = y; _axis = axis; _body = Body; }

    public override void Run(RunParameters p)
    {
        var d = Gen.Dims(_x, p);
        _outer = Gen.Prod(d, 0, _axis); _a = d[_axis]; _inner = Gen.Prod(d, _axis + 1, d.Length);
        _q = _idx.Elements(p);
        long items = _outer * _q;
        if (items == 0 || _inner == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(items, _inner); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize;
        long s = chunk * _per, e = Math.Min(_outer * _q, s + _per);
        for (long it = s; it < e; it++)
        {
            long o = it / _q, q = it % _q;
            long id = Gen.Wrap(Gen.Idx(_idx, q), _a, "GatherAxis");
            Buffer.MemoryCopy(_x.P + ((o * _a + id) * _inner) * es, _y.P + it * _inner * es, _inner * es, _inner * es);
        }
    }
}

internal sealed unsafe class GatherElementsOp : CpuOp
{
    private readonly TRef _x, _idx, _y; private readonly int _axis; private readonly Action<int> _body;
    private long[] _od = [], _xs = []; private long _total, _per, _axisN;
    public override string Name => "GatherElements";
    public GatherElementsOp(TRef x, TRef idx, TRef y, int axis) { _x = x; _idx = idx; _y = y; _axis = axis; _body = Body; }

    public override void Run(RunParameters p)
    {
        _od = Gen.Dims(_idx, p); var xd = Gen.Dims(_x, p);
        _xs = Gen.Strides(xd); _axisN = xd[_axis];
        _total = Gen.Prod(_od, 0, _od.Length);
        if (_total == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(_total, 8); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize, r = _od.Length;
        long s = chunk * _per, e = Math.Min(_total, s + _per);
        for (long l = s; l < e; l++)
        {
            long rem = l, off = 0;
            for (int d = r - 1; d >= 0; d--)
            {
                long ix = rem % _od[d]; rem /= _od[d];
                if (d == _axis)
                {
                    ix = Gen.Wrap(Gen.Idx(_idx, l), _axisN, "GatherElements");
                }

                off += ix * _xs[d];
            }
            Gen.Cp(_y.P + l * es, _x.P + off * es, es);
        }
    }
}

internal sealed unsafe class GatherNdOp : CpuOp
{
    private readonly TRef _x, _idx, _y; private readonly int _b; private readonly Action<int> _body;
    private long[] _xd = [], _xs = []; private long _tuples, _q, _slice, _perBatch, _tuplesPerBatch, _per;
    public override string Name => "GatherND";
    public GatherNdOp(TRef x, TRef idx, TRef y, int batchDims) { _x = x; _idx = idx; _y = y; _b = batchDims; _body = Body; }

    public override void Run(RunParameters p)
    {
        var id = Gen.Dims(_idx, p); _xd = Gen.Dims(_x, p); _xs = Gen.Strides(_xd);
        _q = id[^1];
        _tuples = Gen.Prod(id, 0, id.Length - 1);
        _tuplesPerBatch = Gen.Prod(id, _b, id.Length - 1);
        _perBatch = Gen.Prod(_xd, _b, _xd.Length);
        _slice = Gen.Prod(_xd, _b + (int)_q, _xd.Length);
        if (_tuples == 0 || _slice == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(_tuples, _slice); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize;
        long s = chunk * _per, e = Math.Min(_tuples, s + _per);
        for (long t = s; t < e; t++)
        {
            long batch = t / _tuplesPerBatch, off = batch * _perBatch;
            for (int j = 0; j < _q; j++)
            {
                off += Gen.Wrap(Gen.Idx(_idx, t * _q + j), _xd[_b + j], "GatherND") * _xs[_b + j];
            }

            Buffer.MemoryCopy(_x.P + off * es, _y.P + t * _slice * es, _slice * es, _slice * es);
        }
    }
}

/// <summary>ScatterElements / ScatterND on f32, i32 or i64 data (sequential, so duplicate indices resolve in order).</summary>
internal sealed unsafe class ScatterOp : CpuOp
{
    private readonly TRef _x, _idx, _upd, _y; private readonly int _axis; private readonly bool _nd; private readonly int _red; // 0 none 1 add 2 mul 3 max 4 min
    public override string Name => _nd ? "ScatterND" : "ScatterElements";

    public ScatterOp(TRef x, TRef idx, TRef upd, TRef y, int axis, bool nd, string reduction)
    {
        _x = x; _idx = idx; _upd = upd; _y = y; _axis = axis; _nd = nd;
        _red = reduction switch { "add" => 1, "mul" => 2, "max" => 3, "min" => 4, _ => 0 };
    }

    public override void Run(RunParameters p)
    {
        long n = _x.Elements(p);
        Buffer.MemoryCopy(_x.P, _y.P, n * _x.ElemSize, n * _x.ElemSize);
        if (_x.Type == DataType.F32)
        {
            Go<float>(p);
        }
        else if (_x.Type == DataType.I32)
        {
            Go<int>(p);
        }
        else
        {
            Go<long>(p);
        }
    }

    private static T Red<T>(int red, T a, T b) where T : unmanaged, INumber<T> => red switch
    {
        1 => a + b, 2 => a * b, 3 => T.Max(a, b), 4 => T.Min(a, b), _ => b,
    };

    private void Go<T>(RunParameters p) where T : unmanaged, INumber<T>
    {
        var yd = Gen.Dims(_x, p); var ys = Gen.Strides(yd);
        T* y = (T*)_y.P; T* u = (T*)_upd.P;
        var id = Gen.Dims(_idx, p);
        if (!_nd)
        {
            long total = Gen.Prod(id, 0, id.Length);
            int r = id.Length;
            for (long l = 0; l < total; l++)
            {
                long rem = l, off = 0;
                for (int d = r - 1; d >= 0; d--)
                {
                    long ix = rem % id[d]; rem /= id[d];
                    if (d == _axis)
                    {
                        ix = Gen.Wrap(Gen.Idx(_idx, l), yd[d], "ScatterElements");
                    }

                    off += ix * ys[d];
                }
                y[off] = Red(_red, y[off], u[l]);
            }
        }
        else
        {
            int q = (int)id[^1];
            long tuples = Gen.Prod(id, 0, id.Length - 1), slice = Gen.Prod(yd, q, yd.Length);
            for (long t = 0; t < tuples; t++)
            {
                long off = 0;
                for (int j = 0; j < q; j++)
                {
                    off += Gen.Wrap(Gen.Idx(_idx, t * q + j), yd[j], "ScatterND") * ys[j];
                }

                for (long i = 0; i < slice; i++)
                {
                    y[off + i] = Red(_red, y[off + i], u[t * slice + i]);
                }
            }
        }
    }
}

internal sealed unsafe class CumSumOp : CpuOp
{
    private readonly TRef _x, _y; private readonly int _axis; private readonly bool _excl, _rev; private readonly Action<int> _body;
    private long _outer, _a, _inner, _per;
    public override string Name => "CumSum";
    public CumSumOp(TRef x, TRef y, int axis, bool exclusive, bool reverse) { _x = x; _y = y; _axis = axis; _excl = exclusive; _rev = reverse; _body = Body; }

    public override void Run(RunParameters p)
    {
        var d = Gen.Dims(_x, p);
        _outer = Gen.Prod(d, 0, _axis); _a = d[_axis]; _inner = Gen.Prod(d, _axis + 1, d.Length);
        if (_outer * _a * _inner == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(_outer, _a * _inner); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_outer, s + _per);
        float* x = (float*)_x.P; float* y = (float*)_y.P;
        long n = _inner;
        for (long o = s; o < e; o++)
        {
            float* xb = x + o * _a * n, yb = y + o * _a * n;
            float* prev = null;
            for (long step = 0; step < _a; step++)
            {
                long a = _rev ? _a - 1 - step : step;
                float* xr = xb + a * n, yr = yb + a * n;
                if (prev == null)
                {
                    if (_excl)
                    {
                        new Span<float>(yr, (int)n).Clear();
                    }
                    else
                    {
                        Buffer.MemoryCopy(xr, yr, n * 4, n * 4);
                    }
                }
                else
                {
                    // inclusive: y[a] = y[prev] + x[a];  exclusive: y[a] = y[prev] + x[prev]
                    float* xs = _excl ? xb + (_rev ? a + 1 : a - 1) * n : xr;
                    for (long i = 0; i < n; i++)
                    {
                        yr[i] = prev[i] + xs[i];
                    }
                }
                prev = yr;
            }
        }
    }
}

internal sealed unsafe class ArgReduceOp : CpuOp
{
    private readonly TRef _x, _y; private readonly int _axis; private readonly bool _max, _last; private readonly Action<int> _body;
    private long _outer, _a, _inner, _per;
    public override string Name => "ArgReduce." + (_max ? "max" : "min");
    public ArgReduceOp(TRef x, TRef y, int axis, bool max, bool selectLast) { _x = x; _y = y; _axis = axis; _max = max; _last = selectLast; _body = Body; }

    public override void Run(RunParameters p)
    {
        var d = Gen.Dims(_x, p);
        _outer = Gen.Prod(d, 0, _axis); _a = d[_axis]; _inner = Gen.Prod(d, _axis + 1, d.Length);
        long items = _outer * _inner;
        if (items == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(items, _a); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_outer * _inner, s + _per);
        float* x = (float*)_x.P; int* y = (int*)_y.P;
        for (long it = s; it < e; it++)
        {
            long o = it / _inner, i = it % _inner;
            float* b = x + o * _a * _inner + i;
            int best = 0; float bv = b[0];
            for (int a = 1; a < _a; a++)
            {
                float v = b[a * _inner];
                bool better = _max ? (_last ? v >= bv : v > bv) : (_last ? v <= bv : v < bv);
                if (better) { bv = v; best = a; }
            }
            y[it] = best;
        }
    }
}

internal sealed unsafe class TopKOp : CpuOp
{
    private readonly TRef _x, _v, _i; private readonly int _axis, _k; private readonly bool _largest; private readonly Action<int> _body;
    private long _outer, _a, _inner, _per;
    public override string Name => "TopK";
    public TopKOp(TRef x, TRef vals, TRef idx, int axis, int k, bool largest) { _x = x; _v = vals; _i = idx; _axis = axis; _k = k; _largest = largest; _body = Body; }

    public override void Run(RunParameters p)
    {
        var d = Gen.Dims(_x, p);
        _outer = Gen.Prod(d, 0, _axis); _a = d[_axis]; _inner = Gen.Prod(d, _axis + 1, d.Length);
        long items = _outer * _inner;
        if (items == 0 || _k == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(items, _a * 4); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_outer * _inner, s + _per);
        float* x = (float*)_x.P; float* ov = (float*)_v.P; int* oi = (int*)_i.P;
        var ids = new int[_a];
        for (long it = s; it < e; it++)
        {
            long o = it / _inner, i = it % _inner;
            float* b = x + o * _a * _inner + i;
            for (int a = 0; a < _a; a++)
            {
                ids[a] = a;
            }

            bool largest = _largest;
            // total order: value (desc for largest, asc for smallest; NaN counts as the extreme), then lower index first
            int Cmp(int p, int q)
            {
                float u = b[p * _inner], v = b[q * _inner];
                int c;
                if (u == v || (float.IsNaN(u) && float.IsNaN(v)))
                {
                    c = 0;
                }
                else if (float.IsNaN(u))
                {
                    c = -1;
                }
                else if (float.IsNaN(v))
                {
                    c = 1;
                }
                else
                {
                    c = largest ? (u > v ? -1 : 1) : (u < v ? -1 : 1);
                }

                if (!largest && c != 0 && (float.IsNaN(u) || float.IsNaN(v)))
                {
                    c = -c;
                }

                return c != 0 ? c : p.CompareTo(q);
            }
            if (_k >= _a)
            {
                Array.Sort(ids, Cmp);
            }
            else
            {
                // partial selection: keep the k best in an insertion-sorted prefix
                int k = _k;
                var top = new int[k]; int cnt = 0;
                for (int a = 0; a < _a; a++)
                {
                    if (cnt == k && Cmp(a, top[k - 1]) >= 0)
                    {
                        continue;
                    }

                    int pos = cnt < k ? cnt++ : k - 1;
                    while (pos > 0 && Cmp(a, top[pos - 1]) < 0) { top[pos] = top[pos - 1]; pos--; }
                    top[pos] = a;
                }
                for (int j = 0; j < k; j++)
                {
                    ids[j] = top[j];
                }
            }
            long ob = o * _k * _inner + i;
            for (int j = 0; j < _k; j++) { ov[ob + j * _inner] = b[ids[j] * _inner]; oi[ob + j * _inner] = ids[j]; }
        }
    }
}

internal sealed unsafe class GroupNormOp : CpuOp
{
    private readonly TRef _x, _y; private readonly TRef? _w, _b; private readonly int _groups; private readonly float _eps; private readonly Action<int> _body;
    private long _n, _c, _s, _per;
    public override string Name => "GroupNorm";
    public GroupNormOp(TRef x, TRef? w, TRef? b, TRef y, int groups, float eps) { _x = x; _w = w; _b = b; _y = y; _groups = groups; _eps = eps; _body = Body; }

    public override void Run(RunParameters p)
    {
        var d = Gen.Dims(_x, p);
        _n = d[0]; _c = d[1]; _s = Gen.Prod(d, 2, d.Length);
        long items = _n * _groups;
        if (items * _c * _s == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(items, _c / _groups * _s * 4); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s0 = chunk * _per, e0 = Math.Min(_n * _groups, s0 + _per);
        long cg = _c / _groups; long cnt = cg * _s;
        float* x = (float*)_x.P; float* y = (float*)_y.P;
        float* w = _w is null ? null : (float*)_w.P; float* bi = _b is null ? null : (float*)_b.P;
        for (long it = s0; it < e0; it++)
        {
            long n = it / _groups, g = it % _groups;
            long baseOff = (n * _c + g * cg) * _s;
            float* xb = x + baseOff;
            float mean = Simd.Sum(xb, cnt) / cnt;
            double v = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                var mv = Vector256.Create(mean); var acc = Vector256<float>.Zero; long i = 0;
                for (; i + 8 <= cnt; i += 8) { var dv = Vector256.Load(xb + i) - mv; acc = Vector256.FusedMultiplyAdd(dv, dv, acc); }
                v = Vector256.Sum(acc);
                for (; i < cnt; i++) { double dd = xb[i] - mean; v += dd * dd; }
            }
            else
            {
                for (long i = 0; i < cnt; i++) { double dd = xb[i] - mean; v += dd * dd; }
            }

            float inv = 1f / MathF.Sqrt((float)(v / cnt) + _eps);
            for (long c = 0; c < cg; c++)
            {
                long ch = g * cg + c;
                float sc = (w == null ? 1f : w[ch]) * inv;
                float sh = (bi == null ? 0f : bi[ch]) - mean * sc;
                float* xr = xb + c * _s; float* yr = y + baseOff + c * _s;
                long i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    var sv = Vector256.Create(sc); var hv = Vector256.Create(sh);
                    for (; i + 8 <= _s; i += 8)
                    {
                        Vector256.FusedMultiplyAdd(Vector256.Load(xr + i), sv, hv).Store(yr + i);
                    }
                }
                for (; i < _s; i++)
                {
                    yr[i] = xr[i] * sc + sh;
                }
            }
        }
    }
}

/// <summary>Triangular masking of the last two axes (any plain element type).</summary>
internal sealed unsafe class TriluOp : CpuOp
{
    private readonly TRef _x, _y; private readonly bool _upper; private readonly long _k; private readonly Action<int> _body;
    private long _batch, _m, _n, _per;
    public override string Name => "Trilu";
    public TriluOp(TRef x, TRef y, bool upper, long k) { _x = x; _y = y; _upper = upper; _k = k; _body = Body; }

    public override void Run(RunParameters p)
    {
        var d = Gen.Dims(_x, p);
        _m = d[^2]; _n = d[^1]; _batch = Gen.Prod(d, 0, d.Length - 2);
        if (_batch * _m * _n == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(_batch * _m, _n); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        int es = _x.ElemSize;
        long s = chunk * _per, e = Math.Min(_batch * _m, s + _per);
        for (long r = s; r < e; r++)
        {
            long i = r % _m;
            byte* src = _x.P + r * _n * es, dst = _y.P + r * _n * es;
            for (long j = 0; j < _n; j++)
            {
                bool keep = _upper ? j - i >= _k : j - i <= _k;
                if (keep)
                {
                    Gen.Cp(dst + j * es, src + j * es, es);
                }
                else
                {
                    new Span<byte>(dst + j * es, es).Clear();
                }
            }
        }
    }
}

internal sealed unsafe class OneHotOp : CpuOp
{
    private readonly TRef _idx, _y; private readonly int _axis; private readonly long _depth; private readonly float _on, _off; private readonly Action<int> _body;
    private long _pre, _post, _per;
    public override string Name => "OneHot";
    public OneHotOp(TRef idx, TRef y, int axis, long depth, float on, float off) { _idx = idx; _y = y; _axis = axis; _depth = depth; _on = on; _off = off; _body = Body; }

    public override void Run(RunParameters p)
    {
        var d = Gen.Dims(_idx, p);
        _pre = Gen.Prod(d, 0, _axis); _post = Gen.Prod(d, _axis, d.Length);
        if (_pre * _post * _depth == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(_pre, _post * _depth); _per = per;
        Par.For(ch, _body);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_pre, s + _per);
        float* y = (float*)_y.P;
        for (long pr = s; pr < e; pr++)
        {
            float* yb = y + pr * _depth * _post;
            new Span<float>(yb, (int)(_depth * _post)).Fill(_off);
            for (long q = 0; q < _post; q++)
            {
                long id = Gen.Idx(_idx, pr * _post + q);
                if (id < 0)
                {
                    id += _depth;
                }

                if ((ulong)id < (ulong)_depth)
                {
                    yb[id * _post + q] = _on;
                }
            }
        }
    }
}

/// <summary>Transposed 2-D convolution as GEMM (W^T · X into a column buffer) followed by col2im accumulation.</summary>
internal sealed unsafe class ConvTransposeOp : CpuOp
{
    private readonly TRef _x, _y; private readonly TRef? _bias;
    private readonly int _c, _h, _w, _o, _kh, _kw, _sh, _sw, _pt, _pl, _dh, _dw, _g, _ho, _wo, _cg, _og, _m, _hw, _mp, _nt;
    private readonly float* _apack, _col;
    private readonly Action<int> _gemmBody, _colBody;
    [ThreadStatic] private static float[]? s_scratch;
    private long _n; private int _img, _grp, _mpt, _nBlocksN; private float* _xg, _bias0;
    private long _per;
    public override string Name => "ConvTranspose";

    public ConvTransposeOp(TRef x, float* wDense, int kh, int kw, TRef? bias, TRef y, int groups, int sh, int sw, int pt, int pl, int dh, int dw)
    {
        _x = x; _y = y; _bias = bias;
        _c = (int)x.Dims[1]; _h = (int)x.Dims[2]; _w = (int)x.Dims[3];
        _o = (int)y.Dims[1]; _ho = (int)y.Dims[2]; _wo = (int)y.Dims[3];
        _kh = kh; _kw = kw; _g = groups; _sh = sh; _sw = sw; _pt = pt; _pl = pl; _dh = dh; _dw = dw;
        _cg = _c / _g; _og = _o / _g; _m = _og * kh * kw; _hw = _h * _w;
        int MR = Gemm.MR, NR = Gemm.NR;
        _mp = (_m + MR - 1) / MR; _nt = (_hw + NR - 1) / NR;
        _apack = (float*)Mem.Alloc((long)_g * _mp * _cg * MR * 4, true);
        for (int g = 0; g < _g; g++)
        {
            for (int p = 0; p < _mp; p++)
            {
                int m0 = p * MR, rows = Math.Min(MR, _m - m0);
                Gemm.PackA(_apack + ((long)(g * _mp + p) * _cg) * MR, wDense + (long)g * _cg * _m + m0, 1, _m, rows, _cg, 1f);
            }
        }

        _col = (float*)Mem.Alloc((long)_m * _hw * 4, false);
        _gemmBody = GemmBody; _colBody = ColBody;
    }

    public override void Dispose() { Mem.Free((byte*)_apack); Mem.Free((byte*)_col); }

    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p);
        _bias0 = _bias is null ? null : (float*)_bias.P;
        for (int img = 0; img < _n; img++)
        {
            for (int g = 0; g < _g; g++)
            {
                _img = img; _grp = g;
                _xg = (float*)_x.P + ((long)img * _c + (long)g * _cg) * _hw;
                int mpt = Math.Min(_mp, 16);
                long target = 4L * Par.Threads;
                while (((_mp + mpt - 1) / mpt) * (long)_nt < target && mpt > 1)
                {
                    mpt = (mpt + 1) / 2;
                }

                _mpt = mpt;
                int mBlocks = (_mp + mpt - 1) / mpt;
                Par.For(mBlocks * _nt, _gemmBody);
                int chunks = (int)Math.Min(_og, Par.Chunks((long)_og * _kh * _kw * _hw, 16384));
                _per = (_og + chunks - 1) / chunks;
                Par.For(chunks, _colBody);
            }
        }
    }

    private void GemmBody(int task)
    {
        int MR = Gemm.MR, NR = Gemm.NR, KC = Gemm.KC;
        int mb = task / _nt, t = task % _nt;
        int p0 = mb * _mpt, p1 = Math.Min(_mp, p0 + _mpt);
        var sc = s_scratch;
        if (sc is null || sc.Length < KC * NR)
        {
            s_scratch = sc = GC.AllocateUninitializedArray<float>(KC * NR, pinned: true);
        }

        long n0 = (long)t * NR; int cols = (int)Math.Min(NR, _hw - n0);
        fixed (float* bp = sc)
        {
            for (int kb = 0; kb < _cg; kb += KC)
            {
                int kc = Math.Min(KC, _cg - kb);
                Gemm.PackB(bp, _xg + (long)kb * _hw + n0, _hw, 1, kc, cols);
                for (int p = p0; p < p1; p++)
                {
                    float* a = _apack + ((long)(_grp * _mp + p) * _cg + kb) * MR;
                    float* c = _col + (long)p * MR * _hw + n0;
                    int rows = Math.Min(MR, _m - p * MR);
                    if (rows == MR && cols == NR)
                    {
                        Gemm.Tile(a, bp, NR, c, _hw, kc, null, kb > 0);
                    }
                    else
                    {
                        Gemm.TileEdge(a, bp, NR, c, _hw, kc, null, kb > 0, rows, cols);
                    }
                }
            }
        }
    }

    private void ColBody(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_og, s + _per);
        for (long o = s; o < e; o++)
        {
            long oc = (long)_grp * _og + o;
            float* yp = (float*)_y.P + ((long)_img * _o + oc) * _ho * _wo;
            new Span<float>(yp, _ho * _wo).Fill(_bias0 == null ? 0f : _bias0[oc]);
            for (int kh = 0; kh < _kh; kh++)
            {
                for (int kw = 0; kw < _kw; kw++)
                {
                    float* crow = _col + ((o * _kh + kh) * _kw + kw) * _hw;
                    int off = kw * _dw - _pl;
                    for (int ih = 0; ih < _h; ih++)
                    {
                        int oh = ih * _sh - _pt + kh * _dh;
                        if ((uint)oh >= (uint)_ho)
                        {
                            continue;
                        }

                        float* yr = yp + (long)oh * _wo; float* cr = crow + (long)ih * _w;
                        if (_sw == 1)
                        {
                            int lo = Math.Max(0, -off), hi = Math.Min(_w, _wo - off);
                            if (hi > lo)
                            {
                                Simd.Axpy(yr + lo + off, cr + lo, 1f, hi - lo);
                            }
                        }
                        else
                        {
                            for (int iw = 0; iw < _w; iw++)
                            {
                                int ow = iw * _sw + off;
                                if ((uint)ow < (uint)_wo)
                                {
                                    yr[ow] += cr[iw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// <summary>GridSample (ONNX/ORT semantics): linear, nearest, cubic; zeros / border / reflection padding.</summary>
internal sealed unsafe class GridSampleOp : CpuOp
{
    private readonly TRef _x, _g, _y; private readonly int _mode, _pad; private readonly bool _align; // mode 0 linear 1 nearest 2 cubic; pad 0 zeros 1 border 2 reflection
    private readonly int _c, _h, _w, _ho, _wo; private readonly Action<int> _body;
    private long _n, _rows, _per;
    public override string Name => "GridSample";

    public GridSampleOp(TRef x, TRef grid, TRef y, string mode, string padding, bool alignCorners)
    {
        _x = x; _g = grid; _y = y; _align = alignCorners;
        _mode = mode switch { "nearest" => 1, "cubic" => 2, _ => 0 };
        _pad = padding switch { "border" => 1, "reflection" => 2, _ => 0 };
        _c = (int)x.Dims[1]; _h = (int)x.Dims[2]; _w = (int)x.Dims[3]; _ho = (int)y.Dims[2]; _wo = (int)y.Dims[3];
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _n = _x.Dim(0, p); _rows = _n * _ho;
        if (_rows * _wo * _c == 0)
        {
            return;
        }

        var (ch, per) = Gen.Split(_rows, (long)_wo * _c * 8); _per = per;
        Par.For(ch, _body);
    }

    private float Denorm(float v, int len) => _align ? (v + 1f) * (len - 1) / 2f : ((v + 1f) * len - 1f) / 2f;

    private static float Reflect(float x, float lo, float hi)
    {
        float range = hi - lo;
        if (range == 0)
        {
            return lo;
        }

        if (x < lo)
        {
            float dx = lo - x; int n = (int)(dx / range); float r = dx - n * range;
            return n % 2 == 0 ? lo + r : hi - r;
        }
        if (x > hi)
        {
            float dx = x - hi; int n = (int)(dx / range); float r = dx - n * range;
            return n % 2 == 0 ? hi - r : lo + r;
        }
        return x;
    }

    /// <summary>Applies the padding mode to a (denormalized) coordinate.</summary>
    private float Coord(float x, int len)
    {
        if (_pad == 1)
        {
            return Math.Clamp(x, 0f, len - 1f);
        }

        if (_pad == 2)
        {
            x = _align ? Reflect(x, 0f, len - 1f) : Reflect(x, -0.5f, len - 0.5f);
            return Math.Clamp(x, 0f, len - 1f);
        }
        return x;
    }

    private float Px(float* img, int r, int c) => (uint)r < (uint)_h && (uint)c < (uint)_w ? img[r * _w + c] : 0f;

    /// <summary>Cubic taps apply the padding mode per sample (ORT PixelAtGrid), not to the coordinate.</summary>
    private float PxPad(float* img, int r, int c)
    {
        if (_pad == 0)
        {
            return Px(img, r, c);
        }

        if (_pad == 1)
        {
            return img[Math.Clamp(r, 0, _h - 1) * _w + Math.Clamp(c, 0, _w - 1)];
        }

        float xmin = _align ? 0f : -0.5f, ymin = xmin;
        float xmax = _align ? _w - 1f : _w - 0.5f, ymax = _align ? _h - 1f : _h - 0.5f;
        int rc = (int)Reflect(r, ymin, ymax), cc = (int)Reflect(c, xmin, xmax);
        return img[rc * _w + cc];
    }

    private static void CubicCoeffs(float t, float* c)
    {
        const float A = -0.75f;
        float CC1(float x) => ((A + 2f) * x - (A + 3f)) * x * x + 1f;
        float CC2(float x) => ((A * x - 5f * A) * x + 8f * A) * x - 4f * A;
        c[0] = CC2(t + 1f); c[1] = CC1(t); c[2] = CC1(1f - t); c[3] = CC2(2f - t);
    }

    private void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_rows, s + _per);
        float* coef = stackalloc float[8];
        for (long row = s; row < e; row++)
        {
            long n = row / _ho; int oh = (int)(row % _ho);
            for (int ow = 0; ow < _wo; ow++)
            {
                float* gp = (float*)_g.P + ((n * _ho + oh) * _wo + ow) * 2;
                float fx = Denorm(gp[0], _w), fy = Denorm(gp[1], _h);
                if (_mode != 2) { fx = Coord(fx, _w); fy = Coord(fy, _h); }
                for (int c = 0; c < _c; c++)
                {
                    float* img = (float*)_x.P + (n * _c + c) * _h * _w;
                    float v;
                    if (_mode == 1)
                    {
                        v = Px(img, (int)MathF.Round(fy, MidpointRounding.ToEven), (int)MathF.Round(fx, MidpointRounding.ToEven));
                    }
                    else if (_mode == 0)
                    {
                        float x1 = MathF.Floor(fx), y1 = MathF.Floor(fy);
                        int xi = (int)x1, yi = (int)y1;
                        float dx2 = x1 + 1 - fx, dx1 = fx - x1, dy2 = y1 + 1 - fy, dy1 = fy - y1;
                        v = dy2 * (dx2 * Px(img, yi, xi) + dx1 * Px(img, yi, xi + 1)) + dy1 * (dx2 * Px(img, yi + 1, xi) + dx1 * Px(img, yi + 1, xi + 1));
                    }
                    else
                    {
                        float x0 = MathF.Floor(fx) - 1, y0 = MathF.Floor(fy) - 1;
                        CubicCoeffs(fx - x0 - 1, coef); CubicCoeffs(fy - y0 - 1, coef + 4);
                        int xi = (int)x0, yi = (int)y0;
                        float acc = 0;
                        for (int h = 0; h < 4; h++)
                        {
                            float r = 0;
                            for (int w = 0; w < 4; w++)
                            {
                                r += coef[w] * PxPad(img, yi + h, xi + w);
                            }

                            acc += coef[4 + h] * r;
                        }
                        v = acc;
                    }
                    ((float*)_y.P)[((n * _c + c) * _ho + oh) * _wo + ow] = v;
                }
            }
        }
    }
}
