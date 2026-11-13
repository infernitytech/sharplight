using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Row-wise (last axis) ops: normalizations, softmax and reductions.</summary>
internal sealed unsafe class RowOp : CpuOp
{
    private readonly RowKind _kind;
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef? _w;
    private readonly TRef? _b;
    private readonly float _eps;
    private readonly float _scale;
    private readonly bool _log;
    private readonly Action<int> _body;
    private long _rows;
    private long _cols;
    private long _per;
    public override string Name => "Row." + _kind;

    public RowOp(RowKind kind, TRef x, TRef? w, TRef? b, TRef y, float eps, float scale, bool log = false)
    {
        _log = log;
        _kind = kind;
        _x = x;
        _w = w;
        _b = b;
        _y = y;
        _eps = eps;
        _scale = scale;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _rows = _x.Rows(p);
        _cols = _x.Cols(p);
        if (_rows == 0 || _cols == 0)
        {
            return;
        }

        int chunks = (int)Math.Min(_rows, CpuParallel.Chunks(_rows * _cols, 16384));
        _per = (_rows + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_rows, s + _per);
        long n = _cols;
        float* w = _w is null ? null : (float*)_w.P;
        for (long r = s; r < e; r++)
        {
            float* x = _x.F(r);
            switch (_kind)
            {
                case RowKind.RmsNorm:
                {
                    float* y = (float*)_y.P + r * n;
                    float scale = 1f / MathF.Sqrt(Simd.SumSq(x, n) / n + _eps);
                    if (w != null)
                        {
                            Simd.ScaleMul(x, w, y, scale, n);
                        }
                        else
                        {
                            Simd.Scale(x, y, scale, n);
                        }

                        break;
                }
                case RowKind.L2Norm:
                {
                    float* y = (float*)_y.P + r * n;
                    float scale = 1f / MathF.Sqrt(Simd.SumSq(x, n) + _eps);
                    Simd.Scale(x, y, scale, n);
                    break;
                }
                case RowKind.LayerNorm:
                {
                    float* y = (float*)_y.P + r * n;
                    float mean = Simd.Sum(x, n) / n;
                    double v = 0;
                    for (long i = 0; i < n; i++)
                    {
                        double d = x[i] - mean;
                        v += d * d;
                    }
                    float inv = 1f / MathF.Sqrt((float)(v / n) + _eps);
                    float* bias = _b is null ? null : (float*)_b.P;
                    for (long i = 0; i < n; i++)
                    {
                        float o = (x[i] - mean) * inv;
                        if (w != null)
                            {
                                o *= w[i];
                            }

                            if (bias != null)
                            {
                                o += bias[i];
                            }

                            y[i] = o;
                    }
                    break;
                }
                case RowKind.Softmax:
                {
                    float* y = (float*)_y.P + r * n;
                    float sc = _scale;
                    float m = Simd.Max(x, n);
                    float mz = sc >= 0 ? m * sc : float.NaN;
                    if (sc < 0)
                    {
                        float mn = float.PositiveInfinity;
                        for (long i = 0; i < n; i++)
                        {
                            if (x[i] < mn)
                            {
                                mn = x[i];
                            }
                        }
                        mz = mn * sc;
                    }
                    long i2 = 0;
                    float sum = 0;
                    if (Vector256.IsHardwareAccelerated)
                    {
                        var sv = Vector256.Create(sc);
                        var mv = Vector256.Create(mz);
                        var acc = Vector256<float>.Zero;
                        for (; i2 + 8 <= n; i2 += 8)
                        {
                            var ev = Vector256.Exp(Vector256.Load(x + i2) * sv - mv);
                            ev.Store(y + i2);
                            acc += ev;
                        }
                        sum = Vector256.Sum(acc);
                    }
                    for (; i2 < n; i2++)
                    {
                        float ev = MathF.Exp(x[i2] * sc - mz);
                        y[i2] = ev;
                        sum += ev;
                    }
                    if (_log)
                    {
                        float ls = MathF.Log(sum);
                        for (long i = 0; i < n; i++)
                            {
                                y[i] = x[i] * sc - mz - ls;
                            }
                        }
                    else
                        {
                            Simd.Scale(y, y, 1f / sum, n);
                        }

                        break;
                }
                case RowKind.ReduceSum:
                    ((float*)_y.P)[r] = Simd.Sum(x, n);
                    break;
                case RowKind.ReduceMax:
                    ((float*)_y.P)[r] = Simd.Max(x, n);
                    break;
                case RowKind.ArgMax:
                {
                    int best = 0;
                    float bv = x[0];
                    for (long i = 1; i < n; i++)
                        {
                            if (x[i] > bv)
                            {
                                bv = x[i];
                                best = (int)i;
                            }
                        }

                        ((int*)_y.P)[r] = best;
                    break;
                }
            }
        }
    }
}
