using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Sum / mean / max / min over arbitrary axes of a strided f32 input into a dense output. Dims are coalesced; then one of three
/// strategies is used: contiguous inner reduction (SIMD along the reduced run), contiguous inner kept axis (SIMD across outputs),
/// or a scalar odometer.
/// </summary>
internal sealed unsafe class ReduceOp : CpuOp
{
    private readonly ReduceKind _kind;
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly bool[] _red;
    private readonly Action<int> _body;
    // coalesced description
    private long[] _kd = [], _kx = [], _ky = [];   // kept dims, x strides, y strides (outer to inner)
    private long[] _rd = [], _rx = [];             // reduced dims, x strides
    private int _strategy;
    private long _nOut;
    private long _per;
    private long _redCount;
    private int _innerItem;
    private long _inner;

    public override string Name => "Reduce." + _kind;

    public ReduceOp(ReduceKind kind, TRef x, TRef y, bool[] reduced)
    {
        _kind = kind;
        _x = x;
        _y = y;
        _red = reduced;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        int r = _x.Rank;
        var items = new List<(long d, long xs, long ys, bool red)>();
        var dims = new long[r];
        for (int i = 0; i < r; i++)
        {
            dims[i] = _x.Dim(i, p);
        }
        // output strides over the kept axes, row-major
        var ys = new long[r];
        long acc = 1;
        for (int i = r - 1; i >= 0; i--)
        {
            if (!_red[i])
            {
                ys[i] = acc;
                acc *= dims[i];
            }
        }

        _nOut = acc;
        long redTotal = 1;
        for (int i = 0; i < r; i++)
        {
            if (_red[i])
            {
                redTotal *= dims[i];
            }

            if (dims[i] == 1)
            {
                continue;
            }

            items.Add((dims[i], _x.Strides[i], _red[i] ? 0 : ys[i], _red[i]));
        }
        _redCount = redTotal;
        if (_nOut == 0)
        {
            return;
        }
        // coalesce adjacent items of the same kind
        for (int i = items.Count - 2; i >= 0; i--)
        {
            var a = items[i];
            var b = items[i + 1];
            if (a.red == b.red && a.xs == b.d * b.xs && (a.red || a.ys == b.d * b.ys))
            {
                items[i] = (a.d * b.d, b.xs, b.ys, a.red);
                items.RemoveAt(i + 1);
            }
        }
        var kd = new List<long>();
        var kx = new List<long>();
        var ky = new List<long>();
        var rd = new List<long>();
        var rx = new List<long>();
        foreach (var it in items)
        {
            if (it.red)
            {
                rd.Add(it.d);
                rx.Add(it.xs);
            }
            else
            {
                kd.Add(it.d);
                kx.Add(it.xs);
                ky.Add(it.ys);
            }
        }
        // innermost memory axis decides the strategy
        _strategy = 2;
        _innerItem = -1;
        bool special = _kind >= ReduceKind.Prod;
        int ri = -1;
        for (int i = 0; i < rd.Count; i++)
        {
            if (rx[i] == 1)
            {
                ri = i;
            }
        }

        int ki = -1;
        for (int i = 0; i < kd.Count; i++)
        {
            if (kx[i] == 1 && ky[i] == 1)
            {
                ki = i;
            }
        }

        if (special) { }
        else if (ri >= 0 && rd[ri] >= 8)
        {
            _strategy = 0;
            _innerItem = ri;
            _inner = rd[ri];
            rd.RemoveAt(ri);
            rx.RemoveAt(ri);
        }
        else if (ki >= 0 && kd[ki] >= 8 && rd.Count > 0)
        {
            _strategy = 1;
            _innerItem = ki;
            _inner = kd[ki];
            kd.RemoveAt(ki);
            kx.RemoveAt(ki);
            ky.RemoveAt(ki);
        }
        _kd = kd.ToArray();
        _kx = kx.ToArray();
        _ky = ky.ToArray();
        _rd = rd.ToArray();
        _rx = rx.ToArray();
        long units = _strategy == 1 ? _nOut / _inner : _nOut;
        long perUnit = Math.Max(1, redTotal * (_strategy == 1 ? _inner : 1));
        int chunks = (int)Math.Min(units, CpuParallel.Chunks(units * perUnit, 16384));
        _per = (units + chunks - 1) / chunks;
        _units = units;
        CpuParallel.For(chunks, _body);
    }

    private long _units;

    private float Init => _kind switch { ReduceKind.Max => float.NegativeInfinity, ReduceKind.Min => float.PositiveInfinity, _ => 0f };

    private static float Comb(ReduceKind k, float a, float b) => k switch
    {
        ReduceKind.Max => b > a ? b : a,
        ReduceKind.Min => b < a ? b : a,
        _ => a + b,
    };

    private static float RunReduce(ReduceKind k, float* p, long n)
    {
        switch (k)
        {
            case ReduceKind.Max:
                return Simd.Max(p, n);
            case ReduceKind.Min:
            {
                long i = 0;
                float m = float.PositiveInfinity;
                if (Vector256.IsHardwareAccelerated && n >= 8)
                {
                    var s = Vector256.Create(float.PositiveInfinity);
                    for (; i + 8 <= n; i += 8)
                        {
                            s = Vector256.Min(s, Vector256.Load(p + i));
                        }

                        for (int l = 0; l < 8; l++)
                        {
                            m = Math.Min(m, s.GetElement(l));
                        }
                    }
                for (; i < n; i++)
                    {
                        if (p[i] < m)
                        {
                            m = p[i];
                        }
                    }

                    return m;
            }
            default:
                return Simd.Sum(p, n);
        }
    }

    private void Body(int chunk)
    {
        long s = chunk * _per;
        long e = Math.Min(_units, s + _per);
        int nk = _kd.Length;
        int nr = _rd.Length;
        long* kidx = stackalloc long[nk + 1];
        long* ridx = stackalloc long[nr + 1];
        float* y = (float*)_y.P;
        float* x = (float*)_x.P;
        float mean = _kind == ReduceKind.Mean ? 1f / _redCount : 1f;
        bool isMean = _kind == ReduceKind.Mean;
        var kind = _kind == ReduceKind.Mean ? ReduceKind.Sum : _kind;
        // decompose first unit index
        long xo = 0;
        long yo = 0;
        {
            long rem = s;
            for (int d = nk - 1; d >= 0; d--)
            {
                long ix = rem % _kd[d];
                rem /= _kd[d];
                kidx[d] = ix;
                xo += ix * _kx[d];
                yo += ix * _ky[d];
            }
            if (_strategy == 1)
            {
                /* units enumerate kept dims without the inner one;
                yo already excludes it */
            }
        }
        for (long u = s; u < e; u++)
        {
            switch (_strategy)
            {
                case 0:
                {
                    float a = Init;
                    for (int d = 0; d < nr; d++)
                        {
                            ridx[d] = 0;
                        }

                        long ro = 0;
                    while (true)
                    {
                        a = Comb(kind, a, RunReduce(kind, x + xo + ro, _inner));
                        int d = nr - 1;
                        for (; d >= 0; d--)
                        {
                            ridx[d]++;
                            ro += _rx[d];
                            if (ridx[d] < _rd[d])
                                {
                                    break;
                                }

                                ro -= _rx[d] * _rd[d];
                                ridx[d] = 0;
                        }
                        if (d < 0)
                            {
                                break;
                            }
                        }
                    y[yo] = isMean ? a * mean : a;
                    break;
                }
                case 1:
                {
                    float* o = y + yo;
                    long inner = _inner;
                    for (int d = 0; d < nr; d++)
                        {
                            ridx[d] = 0;
                        }

                        long ro = 0;
                        bool first = true;
                    while (true)
                    {
                        float* src = x + xo + ro;
                        if (first)
                        {
                            new ReadOnlySpan<float>(src, (int)inner).CopyTo(new Span<float>(o, (int)inner));
                            first = false;
                        }
                        else
                            {
                                VecCombine(kind, o, src, inner);
                            }

                            int d = nr - 1;
                        for (; d >= 0; d--)
                        {
                            ridx[d]++;
                            ro += _rx[d];
                            if (ridx[d] < _rd[d])
                                {
                                    break;
                                }

                                ro -= _rx[d] * _rd[d];
                                ridx[d] = 0;
                        }
                        if (d < 0)
                            {
                                break;
                            }
                        }
                    if (isMean)
                        {
                            Simd.Scale(o, o, mean, inner);
                        }

                        break;
                }
                default:
                {
                    if (_kind >= ReduceKind.Prod)
                    {
                        y[yo] = Special(x + xo, ridx, nr);
                        break;
                    }
                    float a = Init;
                    for (int d = 0; d < nr; d++)
                        {
                            ridx[d] = 0;
                        }

                        long ro = 0;
                    while (true)
                    {
                        a = Comb(kind, a, x[xo + ro]);
                        int d = nr - 1;
                        for (; d >= 0; d--)
                        {
                            ridx[d]++;
                            ro += _rx[d];
                            if (ridx[d] < _rd[d])
                                {
                                    break;
                                }

                                ro -= _rx[d] * _rd[d];
                                ridx[d] = 0;
                        }
                        if (d < 0)
                            {
                                break;
                            }
                        }
                    y[yo] = isMean ? a * mean : a;
                    break;
                }
            }
            // advance kept odometer
            for (int d = nk - 1; d >= 0; d--)
            {
                kidx[d]++;
                xo += _kx[d];
                yo += _ky[d];
                if (kidx[d] < _kd[d])
                {
                    break;
                }

                xo -= _kx[d] * _kd[d];
                yo -= _ky[d] * _kd[d];
                kidx[d] = 0;
            }
        }
    }

    private float Special(float* x, long* ridx, int nr)
    {
        float m = 0;
        if (_kind == ReduceKind.LogSumExp)
        {
            m = SpecialPass(x, ridx, nr, 0, 0);
        }

        float acc = SpecialPass(x, ridx, nr, 1, m);
        return _kind switch { ReduceKind.L2 => MathF.Sqrt(acc), ReduceKind.LogSumExp => m + MathF.Log(acc), _ => acc };
    }

    /// <summary>pass 0: max; pass 1: the mode's accumulation (sumexp relative to m for logsumexp).</summary>
    private float SpecialPass(float* x, long* ridx, int nr, int pass, float m)
    {
        float a = pass == 0 ? float.NegativeInfinity : _kind == ReduceKind.Prod ? 1f : 0f;
        for (int d = 0; d < nr; d++)
        {
            ridx[d] = 0;
        }

        long ro = 0;
        while (true)
        {
            float v = x[ro];
            if (pass == 0) { if (v > a)
                {
                    a = v;
                }
            }
            else
            {
                switch (_kind)
            {
                case ReduceKind.Prod:
                    a *= v;
                    break;
                case ReduceKind.L1:
                    a += MathF.Abs(v);
                    break;
                case ReduceKind.L2 or ReduceKind.SumSquare:
                    a += v * v;
                    break;
                default:
                    a += float.IsNegativeInfinity(m) ? 1f : MathF.Exp(v - m);
                    break;
            }
            }

            int d = nr - 1;
            for (; d >= 0; d--)
            {
                ridx[d]++;
                ro += _rx[d];
                if (ridx[d] < _rd[d])
                {
                    break;
                }

                ro -= _rx[d] * _rd[d];
                ridx[d] = 0;
            }
            if (d < 0)
            {
                break;
            }
        }
        return a;
    }

    private static void VecCombine(ReduceKind k, float* o, float* s, long n)
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i + 8 <= n; i += 8)
            {
                var a = Vector256.Load(o + i);
                var b = Vector256.Load(s + i);
                (k == ReduceKind.Max ? Vector256.Max(a, b) : k == ReduceKind.Min ? Vector256.Min(a, b) : a + b).Store(o + i);
            }
        }
        for (; i < n; i++)
        {
            o[i] = Comb(k, o[i], s[i]);
        }
    }
}
