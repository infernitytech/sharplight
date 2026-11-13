using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>2-D max / average pooling over dense [N, C, H, W]; parallel over (n, c) planes.</summary>
internal sealed unsafe class PoolOp : CpuOp
{
    readonly TRef _x, _y; readonly bool _max, _includePad;
    readonly int _h, _w, _ho, _wo, _kh, _kw, _sh, _sw, _pt, _pl, _pb, _pr, _dh, _dw;
    readonly Action<int> _body;
    long _planes, _per;
    public override string Name => _max ? "Pool.max" : "Pool.avg";

    public PoolOp(TRef x, TRef y, bool max, int[] kernel, int[] strides, int[] pads, int[] dil, bool countIncludePad)
    {
        _x = x; _y = y; _max = max; _includePad = countIncludePad;
        _h = (int)x.Dims[2]; _w = (int)x.Dims[3]; _ho = (int)y.Dims[2]; _wo = (int)y.Dims[3];
        _kh = kernel[0]; _kw = kernel[1]; _sh = strides[0]; _sw = strides[1];
        _pt = pads[0]; _pl = pads[1]; _pb = pads[2]; _pr = pads[3]; _dh = dil[0]; _dw = dil[1];
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _planes = _x.Dim(0, p) * _x.Dims[1];
        if (_planes == 0) return;
        int chunks = (int)Math.Min(_planes, Par.Chunks(_planes * _ho * _wo * _kh * _kw, 16384));
        _per = (_planes + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_planes, s + _per);
        for (long pl = s; pl < e; pl++)
        {
            float* xp = (float*)_x.P + pl * _h * _w;
            float* yp = (float*)_y.P + pl * _ho * _wo;
            for (int oh = 0; oh < _ho; oh++)
            {
                int hs = oh * _sh - _pt;
                // valid kernel rows
                int fh0 = hs >= 0 ? 0 : (-hs + _dh - 1) / _dh;
                int fh1 = Math.Min(_kh, _h - 1 - hs < 0 ? 0 : (_h - 1 - hs) / _dh + 1);
                for (int ow = 0; ow < _wo; ow++)
                {
                    int ws = ow * _sw - _pl;
                    int fw0 = ws >= 0 ? 0 : (-ws + _dw - 1) / _dw;
                    int fw1 = Math.Min(_kw, _w - 1 - ws < 0 ? 0 : (_w - 1 - ws) / _dw + 1);
                    if (_max)
                    {
                        float m = float.NegativeInfinity;
                        for (int fh = fh0; fh < fh1; fh++)
                        {
                            float* r = xp + (long)(hs + fh * _dh) * _w + ws;
                            for (int fw = fw0; fw < fw1; fw++) { float v = r[fw * _dw]; if (v > m) m = v; }
                        }
                        yp[oh * _wo + ow] = m;
                    }
                    else
                    {
                        float sum = 0;
                        for (int fh = fh0; fh < fh1; fh++)
                        {
                            float* r = xp + (long)(hs + fh * _dh) * _w + ws;
                            for (int fw = fw0; fw < fw1; fw++) sum += r[fw * _dw];
                        }
                        long cnt;
                        if (_includePad)
                        {
                            // window clipped to the padded input (ceil_mode overhang is not counted)
                            long he = Math.Min(hs + _kh, _h + _pb), we = Math.Min(ws + _kw, _w + _pr);
                            cnt = (he - hs) * (we - ws);
                        }
                        else cnt = (long)Math.Max(0, fh1 - fh0) * Math.Max(0, fw1 - fw0);
                        yp[oh * _wo + ow] = cnt > 0 ? sum / cnt : 0f;
                    }
                }
            }
        }
    }
}

/// <summary>Constant / reflect / edge padding of an arbitrary-rank (strided) f32 tensor; negative pads crop.</summary>
internal sealed unsafe class PadOp : CpuOp
{
    readonly TRef _x, _y; readonly long[] _pads; readonly int _mode; readonly float _value; // 0 constant 1 reflect 2 edge
    readonly Action<int> _body;
    long[] _od = [], _id = []; long _rows, _per;
    public override string Name => "Pad";

    public PadOp(TRef x, TRef y, long[] pads, string mode, float value)
    {
        _x = x; _y = y; _pads = pads; _value = value;
        _mode = mode switch { "reflect" => 1, "edge" => 2, _ => 0 };
        _body = Body;
    }

    /// <summary>Source index for output index <paramref name="o"/> of an axis (padded by <paramref name="pb"/>), or -1 for constant fill.</summary>
    static long Map(long o, long pb, long n, int mode)
    {
        long i = o - pb;
        if (i >= 0 && i < n) return i;
        switch (mode)
        {
            case 0: return -1;
            case 2: return i < 0 ? 0 : n - 1;
            default:
                if (n == 1) return 0;
                long period = 2 * (n - 1);
                i %= period; if (i < 0) i += period;
                return i >= n ? period - i : i;
        }
    }

    public override void Run(RunParameters p)
    {
        int r = _x.Rank;
        _id = new long[r]; _od = new long[r];
        for (int d = 0; d < r; d++) { _id[d] = _x.Dim(d, p); _od[d] = _id[d] + _pads[d] + _pads[d + r]; }
        long total = 1; foreach (var d in _od) total *= d;
        if (total == 0) return;
        _rows = total / _od[r - 1];
        int chunks = (int)Math.Min(_rows, Par.Chunks(total, 16384));
        _per = (_rows + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int chunk)
    {
        int r = _x.Rank;
        long s = chunk * _per, e = Math.Min(_rows, s + _per);
        long lastN = _id[r - 1], lastO = _od[r - 1], pbl = _pads[r - 1], sl = _x.Strides[r - 1];
        for (long row = s; row < e; row++)
        {
            long rem = row, so = 0; bool fill = false;
            for (int d = r - 2; d >= 0; d--)
            {
                long ix = rem % _od[d]; rem /= _od[d];
                long m = Map(ix, _pads[d], _id[d], _mode);
                if (m < 0) { fill = true; break; }
                so += m * _x.Strides[d];
            }
            float* y = (float*)_y.P + row * lastO;
            if (fill) { new Span<float>(y, (int)lastO).Fill(_value); continue; }
            float* x = (float*)_x.P + so;
            for (long j = 0; j < lastO; j++)
            {
                long m = Map(j, pbl, lastN, _mode);
                y[j] = m < 0 ? _value : x[m * sl];
            }
        }
    }
}

/// <summary>2-D resize (nearest / bilinear) with ONNX Runtime coordinate semantics (single precision).</summary>
internal sealed unsafe class ResizeOp : CpuOp
{
    readonly TRef _x, _y; readonly bool _linear; readonly string _coord, _nearest;
    readonly int _h, _w, _oh, _ow;
    readonly int[] _yi0, _yi1, _xi0, _xi1; readonly float[] _yw, _xw;
    readonly Action<int> _body;
    long _planes, _per;
    public override string Name => "Resize." + (_linear ? "linear" : "nearest");

    public ResizeOp(TRef x, TRef y, string mode, string coord, string nearestMode)
    {
        _x = x; _y = y; _linear = mode == "linear"; _coord = coord; _nearest = nearestMode;
        _h = (int)x.Dims[2]; _w = (int)x.Dims[3]; _oh = (int)y.Dims[2]; _ow = (int)y.Dims[3];
        (_yi0, _yi1, _yw) = Axis(_h, _oh); (_xi0, _xi1, _xw) = Axis(_w, _ow);
        _body = Body;
    }

    float Original(int xr, int lenIn, int lenOut)
    {
        float scale = (float)lenOut / lenIn;
        switch (_coord)
        {
            case "align_corners": return lenOut == 1 ? 0f : (float)xr * (lenIn - 1) / (lenOut - 1);
            case "asymmetric": return xr / scale;
            case "pytorch_half_pixel": return lenOut > 1 ? (xr + 0.5f) / scale - 0.5f : 0f;
            default: return (xr + 0.5f) / scale - 0.5f;
        }
    }

    (int[], int[], float[]) Axis(int lenIn, int lenOut)
    {
        var i0 = new int[lenOut]; var i1 = new int[lenOut]; var w = new float[lenOut];
        for (int o = 0; o < lenOut; o++)
        {
            float x = Original(o, lenIn, lenOut);
            if (_linear)
            {
                x = Math.Max(0f, Math.Min(x, lenIn - 1));
                int a = Math.Min((int)x, lenIn - 1), b = Math.Min(a + 1, lenIn - 1);
                i0[o] = a; i1[o] = b; w[o] = x - a;
            }
            else
            {
                long idx = _nearest switch
                {
                    "floor" => (long)MathF.Floor(x),
                    "ceil" => (long)MathF.Ceiling(x),
                    "round_prefer_ceil" => x == (long)x + 0.5f ? (long)MathF.Ceiling(x) : (long)MathF.Round(x, MidpointRounding.AwayFromZero),
                    _ => x == (long)x + 0.5f ? (long)MathF.Floor(x) : (long)MathF.Round(x, MidpointRounding.AwayFromZero),
                };
                i0[o] = (int)Math.Clamp(idx, 0, lenIn - 1);
            }
        }
        return (i0, i1, w);
    }

    public override void Run(RunParameters p)
    {
        _planes = _x.Dim(0, p) * _x.Dims[1];
        if (_planes == 0) return;
        int chunks = (int)Math.Min(_planes, Par.Chunks(_planes * _oh * _ow, 16384));
        _per = (_planes + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_planes, s + _per);
        for (long pl = s; pl < e; pl++)
        {
            float* xp = (float*)_x.P + pl * _h * _w;
            float* yp = (float*)_y.P + pl * _oh * _ow;
            for (int oy = 0; oy < _oh; oy++)
            {
                float* y = yp + (long)oy * _ow;
                if (!_linear)
                {
                    float* r = xp + (long)_yi0[oy] * _w;
                    for (int ox = 0; ox < _ow; ox++) y[ox] = r[_xi0[ox]];
                }
                else
                {
                    float* r0 = xp + (long)_yi0[oy] * _w, r1 = xp + (long)_yi1[oy] * _w;
                    float dy = _yw[oy], dy2 = 1f - dy;
                    for (int ox = 0; ox < _ow; ox++)
                    {
                        float dx = _xw[ox], dx2 = 1f - dx;
                        int a = _xi0[ox], b = _xi1[ox];
                        y[ox] = r0[a] * dx2 * dy2 + r0[b] * dx * dy2 + r1[a] * dx2 * dy + r1[b] * dx * dy;
                    }
                }
            }
        }
    }
}

internal enum ReduceKind { Sum, Mean, Max, Min, Prod, L1, L2, SumSquare, LogSumExp }

/// <summary>
/// Sum / mean / max / min over arbitrary axes of a strided f32 input into a dense output. Dims are coalesced; then one of three
/// strategies is used: contiguous inner reduction (SIMD along the reduced run), contiguous inner kept axis (SIMD across outputs),
/// or a scalar odometer.
/// </summary>
internal sealed unsafe class ReduceOp : CpuOp
{
    readonly ReduceKind _kind; readonly TRef _x, _y; readonly bool[] _red;
    readonly Action<int> _body;
    // coalesced description
    long[] _kd = [], _kx = [], _ky = [];   // kept dims, x strides, y strides (outer to inner)
    long[] _rd = [], _rx = [];             // reduced dims, x strides
    int _strategy; long _nOut, _per, _redCount; int _innerItem; long _inner;

    public override string Name => "Reduce." + _kind;

    public ReduceOp(ReduceKind kind, TRef x, TRef y, bool[] reduced)
    {
        _kind = kind; _x = x; _y = y; _red = reduced; _body = Body;
    }

    public override void Run(RunParameters p)
    {
        int r = _x.Rank;
        var items = new List<(long d, long xs, long ys, bool red)>();
        var dims = new long[r];
        for (int i = 0; i < r; i++) dims[i] = _x.Dim(i, p);
        // output strides over the kept axes, row-major
        var ys = new long[r]; long acc = 1;
        for (int i = r - 1; i >= 0; i--) if (!_red[i]) { ys[i] = acc; acc *= dims[i]; }
        _nOut = acc;
        long redTotal = 1;
        for (int i = 0; i < r; i++)
        {
            if (_red[i]) redTotal *= dims[i];
            if (dims[i] == 1) continue;
            items.Add((dims[i], _x.Strides[i], _red[i] ? 0 : ys[i], _red[i]));
        }
        _redCount = redTotal;
        if (_nOut == 0) return;
        // coalesce adjacent items of the same kind
        for (int i = items.Count - 2; i >= 0; i--)
        {
            var a = items[i]; var b = items[i + 1];
            if (a.red == b.red && a.xs == b.d * b.xs && (a.red || a.ys == b.d * b.ys))
            {
                items[i] = (a.d * b.d, b.xs, b.ys, a.red);
                items.RemoveAt(i + 1);
            }
        }
        var kd = new List<long>(); var kx = new List<long>(); var ky = new List<long>(); var rd = new List<long>(); var rx = new List<long>();
        foreach (var it in items)
            if (it.red) { rd.Add(it.d); rx.Add(it.xs); } else { kd.Add(it.d); kx.Add(it.xs); ky.Add(it.ys); }
        // innermost memory axis decides the strategy
        _strategy = 2; _innerItem = -1;
        bool special = _kind >= ReduceKind.Prod;
        int ri = -1;
        for (int i = 0; i < rd.Count; i++) if (rx[i] == 1) ri = i;
        int ki = -1;
        for (int i = 0; i < kd.Count; i++) if (kx[i] == 1 && ky[i] == 1) ki = i;
        if (special) { }
        else if (ri >= 0 && rd[ri] >= 8)
        {
            _strategy = 0; _innerItem = ri; _inner = rd[ri];
            rd.RemoveAt(ri); rx.RemoveAt(ri);
        }
        else if (ki >= 0 && kd[ki] >= 8 && rd.Count > 0)
        {
            _strategy = 1; _innerItem = ki; _inner = kd[ki];
            kd.RemoveAt(ki); kx.RemoveAt(ki); ky.RemoveAt(ki);
        }
        _kd = kd.ToArray(); _kx = kx.ToArray(); _ky = ky.ToArray(); _rd = rd.ToArray(); _rx = rx.ToArray();
        long units = _strategy == 1 ? _nOut / _inner : _nOut;
        long perUnit = Math.Max(1, redTotal * (_strategy == 1 ? _inner : 1));
        int chunks = (int)Math.Min(units, Par.Chunks(units * perUnit, 16384));
        _per = (units + chunks - 1) / chunks;
        _units = units;
        Par.For(chunks, _body);
    }

    long _units;

    float Init => _kind switch { ReduceKind.Max => float.NegativeInfinity, ReduceKind.Min => float.PositiveInfinity, _ => 0f };

    static float Comb(ReduceKind k, float a, float b) => k switch
    {
        ReduceKind.Max => b > a ? b : a,
        ReduceKind.Min => b < a ? b : a,
        _ => a + b,
    };

    static float RunReduce(ReduceKind k, float* p, long n)
    {
        switch (k)
        {
            case ReduceKind.Max: return Simd.Max(p, n);
            case ReduceKind.Min:
            {
                long i = 0; float m = float.PositiveInfinity;
                if (Vector256.IsHardwareAccelerated && n >= 8)
                {
                    var s = Vector256.Create(float.PositiveInfinity);
                    for (; i + 8 <= n; i += 8) s = Vector256.Min(s, Vector256.Load(p + i));
                    for (int l = 0; l < 8; l++) m = Math.Min(m, s.GetElement(l));
                }
                for (; i < n; i++) if (p[i] < m) m = p[i];
                return m;
            }
            default: return Simd.Sum(p, n);
        }
    }

    void Body(int chunk)
    {
        long s = chunk * _per, e = Math.Min(_units, s + _per);
        int nk = _kd.Length, nr = _rd.Length;
        long* kidx = stackalloc long[nk + 1]; long* ridx = stackalloc long[nr + 1];
        float* y = (float*)_y.P; float* x = (float*)_x.P;
        float mean = _kind == ReduceKind.Mean ? 1f / _redCount : 1f;
        bool isMean = _kind == ReduceKind.Mean;
        var kind = _kind == ReduceKind.Mean ? ReduceKind.Sum : _kind;
        // decompose first unit index
        long xo = 0, yo = 0;
        {
            long rem = s;
            for (int d = nk - 1; d >= 0; d--) { long ix = rem % _kd[d]; rem /= _kd[d]; kidx[d] = ix; xo += ix * _kx[d]; yo += ix * _ky[d]; }
            if (_strategy == 1) { /* units enumerate kept dims without the inner one; yo already excludes it */ }
        }
        for (long u = s; u < e; u++)
        {
            switch (_strategy)
            {
                case 0:
                {
                    float a = Init;
                    for (int d = 0; d < nr; d++) ridx[d] = 0;
                    long ro = 0;
                    while (true)
                    {
                        a = Comb(kind, a, RunReduce(kind, x + xo + ro, _inner));
                        int d = nr - 1;
                        for (; d >= 0; d--)
                        {
                            ridx[d]++; ro += _rx[d];
                            if (ridx[d] < _rd[d]) break;
                            ro -= _rx[d] * _rd[d]; ridx[d] = 0;
                        }
                        if (d < 0) break;
                    }
                    y[yo] = isMean ? a * mean : a;
                    break;
                }
                case 1:
                {
                    float* o = y + yo;
                    long inner = _inner;
                    for (int d = 0; d < nr; d++) ridx[d] = 0;
                    long ro = 0; bool first = true;
                    while (true)
                    {
                        float* src = x + xo + ro;
                        if (first) { new ReadOnlySpan<float>(src, (int)inner).CopyTo(new Span<float>(o, (int)inner)); first = false; }
                        else VecCombine(kind, o, src, inner);
                        int d = nr - 1;
                        for (; d >= 0; d--)
                        {
                            ridx[d]++; ro += _rx[d];
                            if (ridx[d] < _rd[d]) break;
                            ro -= _rx[d] * _rd[d]; ridx[d] = 0;
                        }
                        if (d < 0) break;
                    }
                    if (isMean) Simd.Scale(o, o, mean, inner);
                    break;
                }
                default:
                {
                    if (_kind >= ReduceKind.Prod) { y[yo] = Special(x + xo, ridx, nr); break; }
                    float a = Init;
                    for (int d = 0; d < nr; d++) ridx[d] = 0;
                    long ro = 0;
                    while (true)
                    {
                        a = Comb(kind, a, x[xo + ro]);
                        int d = nr - 1;
                        for (; d >= 0; d--)
                        {
                            ridx[d]++; ro += _rx[d];
                            if (ridx[d] < _rd[d]) break;
                            ro -= _rx[d] * _rd[d]; ridx[d] = 0;
                        }
                        if (d < 0) break;
                    }
                    y[yo] = isMean ? a * mean : a;
                    break;
                }
            }
            // advance kept odometer
            for (int d = nk - 1; d >= 0; d--)
            {
                kidx[d]++; xo += _kx[d]; yo += _ky[d];
                if (kidx[d] < _kd[d]) break;
                xo -= _kx[d] * _kd[d]; yo -= _ky[d] * _kd[d]; kidx[d] = 0;
            }
        }
    }

    float Special(float* x, long* ridx, int nr)
    {
        float m = 0;
        if (_kind == ReduceKind.LogSumExp) m = SpecialPass(x, ridx, nr, 0, 0);
        float acc = SpecialPass(x, ridx, nr, 1, m);
        return _kind switch { ReduceKind.L2 => MathF.Sqrt(acc), ReduceKind.LogSumExp => m + MathF.Log(acc), _ => acc };
    }

    /// <summary>pass 0: max; pass 1: the mode's accumulation (sumexp relative to m for logsumexp).</summary>
    float SpecialPass(float* x, long* ridx, int nr, int pass, float m)
    {
        float a = pass == 0 ? float.NegativeInfinity : _kind == ReduceKind.Prod ? 1f : 0f;
        for (int d = 0; d < nr; d++) ridx[d] = 0;
        long ro = 0;
        while (true)
        {
            float v = x[ro];
            if (pass == 0) { if (v > a) a = v; }
            else switch (_kind)
            {
                case ReduceKind.Prod: a *= v; break;
                case ReduceKind.L1: a += MathF.Abs(v); break;
                case ReduceKind.L2 or ReduceKind.SumSquare: a += v * v; break;
                default: a += float.IsNegativeInfinity(m) ? 1f : MathF.Exp(v - m); break;
            }
            int d = nr - 1;
            for (; d >= 0; d--)
            {
                ridx[d]++; ro += _rx[d];
                if (ridx[d] < _rd[d]) break;
                ro -= _rx[d] * _rd[d]; ridx[d] = 0;
            }
            if (d < 0) break;
        }
        return a;
    }

    static void VecCombine(ReduceKind k, float* o, float* s, long n)
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 8 <= n; i += 8)
            {
                var a = Vector256.Load(o + i); var b = Vector256.Load(s + i);
                (k == ReduceKind.Max ? Vector256.Max(a, b) : k == ReduceKind.Min ? Vector256.Min(a, b) : a + b).Store(o + i);
            }
        }
        for (; i < n; i++) o[i] = Comb(k, o[i], s[i]);
    }
}
