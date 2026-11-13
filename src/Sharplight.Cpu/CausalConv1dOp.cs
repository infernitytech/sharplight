using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Causal depthwise conv with carried state (+ optional SiLU).</summary>
internal sealed unsafe class CausalConv1dOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _w;
    private readonly TRef _state;
    private readonly TRef _y;
    private readonly TRef? _bias;
    private readonly bool _silu;
    private readonly bool _staticWeight;
    private readonly long _c;
    private readonly long _k;
    private readonly float* _wT; // [K, C]
    private readonly Action<int> _body;
    private bool _transposed;
    private long _t;
    private long _per;
    private readonly TRef? _runs;   // multi-sequence: [U, 4] (slot, firstRow, count, firstPos); state [S, K-1, C]
    private readonly long _slotStride;
    private long _u;
    public override string Name => "CausalConv1d";

    public CausalConv1dOp(TRef x, TRef w, TRef state, TRef y, bool silu, bool staticWeight, TRef? bias = null, TRef? runs = null)
    {
        _bias = bias;
        _x = x;
        _w = w;
        _state = state;
        _y = y;
        _silu = silu;
        _staticWeight = staticWeight;
        _runs = runs;
        _c = w.Dims[0];
        _k = w.Dims[1];
        int o = runs is null ? 0 : 1;
        if (x.Dims[1] != _c || state.Dims[o] != _k - 1 || state.Dims[o + 1] != _c)
        {
            throw new ArgumentException("CausalConv1d shape mismatch.");
        }

        _slotStride = (_k - 1) * _c;
        _wT = (float*)Mem.Alloc(_c * _k * 4, false);
        _body = Body;
    }

    public override void Dispose() => Mem.Free((byte*)_wT);

    private void Transpose()
    {
        float* w = (float*)_w.P;
        for (long c = 0; c < _c; c++)
        {
            for (long j = 0; j < _k; j++)
            {
                _wT[j * _c + c] = w[c * _w.Strides[0] + j * _w.Strides[1]];
            }
        }
    }

    /// <summary>Run u: (state slot, first row, row count).</summary>
    private (long Slot, long Row, long Count) RunAt(long u)
    {
        if (_runs is null)
        {
            return (0, 0, _t);
        }

        int* r = (int*)_runs.P + u * 4;
        return (r[0], r[1], r[2]);
    }

    public override void Run(RunParameters p)
    {
        if (!_staticWeight || !_transposed)
        {
            Transpose();
            _transposed = true;
        }
        _t = _x.Dim(0, p);
        if (_t == 0)
        {
            return;
        }

        _u = _runs is null ? 1 : _runs.Dim(0, p);
        long slots = _runs is null ? 1 : _state.Dims[0];
        for (long u = 0; u < _u; u++)
        {
            var (sl, r0, cnt) = RunAt(u);
            if ((ulong)sl >= (ulong)slots || r0 < 0 || cnt < 0 || r0 + cnt > _t)
            {
                throw new InvalidOperationException($"CausalConv1d: bad sequence run {u}.");
            }

            if (_runs is not null && ((int*)_runs.P)[u * 4 + 3] == 0)
            {
                new Span<float>((float*)_state.P + sl * _slotStride, (int)_slotStride).Clear();
            }
        }
        int chunks = (int)Math.Max(1, Math.Min(CpuParallel.Threads * 2L, _c / 256));
        _per = ((_c + chunks - 1) / chunks + 7) & ~7L;
        CpuParallel.For(chunks, _body);
        for (long u = 0; u < _u; u++)
        {
            var (sl, r0, cnt) = RunAt(u);
            UpdateState((float*)_state.P + sl * _slotStride, r0, cnt);
        }
    }

    /// <summary>state := last K-1 rows of [state; x[r0 .. r0+t)].</summary>
    private void UpdateState(float* st, long r0, long t)
    {
        long km1 = _k - 1;
        if (km1 == 0 || t == 0)
        {
            return;
        }

        if (t >= km1)
        {
            for (long i = 0; i < km1; i++)
            {
                Buffer.MemoryCopy(_x.F(r0 + t - km1 + i), st + i * _c, _c * 4, _c * 4);
            }
        }
        else
        {
            Buffer.MemoryCopy(st + t * _c, st, (km1 - t) * _c * 4, (km1 - t) * _c * 4);
            for (long i = 0; i < t; i++)
            {
                Buffer.MemoryCopy(_x.F(r0 + i), st + (km1 - t + i) * _c, _c * 4, _c * 4);
            }
        }
    }

    private void Body(int ch)
    {
        long c0 = ch * _per;
        long c1 = Math.Min(_c, c0 + _per);
        if (c0 >= c1)
        {
            return;
        }

        long km1 = _k - 1;
        for (long u = 0; u < _u; u++)
        {
        var (sl, r0, cnt) = RunAt(u);
        float* st = (float*)_state.P + sl * _slotStride;
        for (long t = 0; t < cnt; t++)
        {
            float* y = (float*)_y.P + (r0 + t) * _c;
            for (long j = 0; j < _k; j++)
            {
                long wr = t + j;
                float* src = wr < km1 ? st + wr * _c : _x.F(r0 + wr - km1);
                float* w = _wT + j * _c;
                long i = c0;
                if (j == 0)
                {
                    if (CpuIsa.Use256)
                        {
                            for (; i + 8 <= c1; i += 8)
                            {
                                (Vector256.Load(src + i) * Vector256.Load(w + i)).Store(y + i);
                            }
                        }

                        for (; i < c1; i++)
                        {
                            y[i] = src[i] * w[i];
                        }
                    }
                else
                {
                    if (CpuIsa.Use256)
                        {
                            for (; i + 8 <= c1; i += 8)
                            {
                                Vector256.FusedMultiplyAdd(Vector256.Load(src + i), Vector256.Load(w + i), Vector256.Load(y + i)).Store(y + i);
                            }
                        }

                        for (; i < c1; i++)
                        {
                            y[i] += src[i] * w[i];
                        }
                    }
            }
            if (_bias is not null) { float* bi = (float*)_bias.P; for (long i = c0; i < c1; i++)
                    {
                        y[i] += bi[i];
                    }
                }
            if (_silu)
                {
                    ScalarMath.Unary(UnaryKind.Silu, y + c0, y + c0, c1 - c0, 1, 0);
                }
            }
        }
    }
}
