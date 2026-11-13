using System.Runtime.Intrinsics;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

internal enum UnaryKind { Neg, Exp, Log, Sqrt, Rsqrt, Sigmoid, Silu, Gelu, GeluTanh, Tanh, Softplus, Relu, Abs, Square, Recip, Scale, Erf, HardSwish, HardSigmoid, Clip,
    Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Asinh, Acosh, Atanh, Floor, Ceil, Round, Sign, Softsign, Mish,
    Elu, Celu, Selu, LeakyRelu, ThresholdedRelu, Not, IsNaN, IsInf }

internal static unsafe class Math2
{
    /// <summary>erf in double precision (series for small |x|, continued fraction for the tail).</summary>
    public static double Erf(double x)
    {
        double ax = Math.Abs(x);
        double r;
        if (ax < 2.5)
        {
            // erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_{n>=0} 2^n x^(2n+1) / (1*3*...*(2n+1))   (all terms positive)
            double x2 = ax * ax, term = ax, sum = ax;
            for (int n = 1; n < 200; n++)
            {
                term *= 2 * x2 / (2 * n + 1);
                sum += term;
                if (term < sum * 1e-17) break;
            }
            r = 1.1283791670955126 * Math.Exp(-x2) * sum;
        }
        else if (ax > 6) r = 1.0;
        else
        {
            // erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
            double f = ax;
            for (int k = 80; k >= 1; k--) f = ax + 0.5 * k / f;
            r = 1.0 - Math.Exp(-ax * ax) / (1.7724538509055159 * f);
        }
        return x < 0 ? -r : r;
    }

    // Rational erf (Eigen/XLA style), max abs error ~1e-7 in f32; |x| is clamped where erf == +-1 to f32 precision.
    const float ErfClamp = 3.832506856900711f;
    const double A1 = -1.60960333262415e-02, A3 = -2.95459980854025e-03, A5 = -7.34990630326855e-04, A7 = -5.69250639462346e-05,
        A9 = -2.10102402082508e-06, A11 = 2.77068142495902e-08, A13 = -2.72614225801306e-10,
        B0 = -1.42647390514189e-02, B2 = -7.37332916720468e-03, B4 = -1.68282697438203e-03, B6 = -2.13374055278905e-04, B8 = -1.45660718464996e-05;

    // The rational is evaluated in double precision (f32 evaluation loses ~2e-7 to cancellation); the approximation error is ~9e-8.
    public static float ErfF(float x)
    {
        double xd = Math.Clamp((double)x, -ErfClamp, ErfClamp);
        double x2 = xd * xd;
        double p = A13;
        p = p * x2 + A11; p = p * x2 + A9; p = p * x2 + A7; p = p * x2 + A5; p = p * x2 + A3; p = p * x2 + A1;
        p *= xd;
        double q = B8;
        q = q * x2 + B6; q = q * x2 + B4; q = q * x2 + B2; q = q * x2 + B0;
        return (float)(p / q);
    }

    static Vector256<double> ErfD(Vector256<double> x)
    {
        x = Vector256.Min(Vector256.Max(x, Vector256.Create((double)-ErfClamp)), Vector256.Create((double)ErfClamp));
        var x2 = x * x;
        var p = Vector256.Create((double)A13);
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A11)); p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A9));
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A7)); p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A5));
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A3)); p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A1));
        p *= x;
        var q = Vector256.Create((double)B8);
        q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B6)); q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B4));
        q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B2)); q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B0));
        return p / q;
    }

    public static Vector256<float> ErfV(Vector256<float> x) =>
        Vector256.Narrow(ErfD(Vector256.WidenLower(x)), ErfD(Vector256.WidenUpper(x)));

    public static void Unary(UnaryKind k, float* x, float* y, long n, float scale, float bias)
    {
        long i = 0;
        switch (k)
        {
            case UnaryKind.Sin: for (; i < n; i++) y[i] = MathF.Sin(x[i]); break;
            case UnaryKind.Cos: for (; i < n; i++) y[i] = MathF.Cos(x[i]); break;
            case UnaryKind.Tan: for (; i < n; i++) y[i] = MathF.Tan(x[i]); break;
            case UnaryKind.Asin: for (; i < n; i++) y[i] = MathF.Asin(x[i]); break;
            case UnaryKind.Acos: for (; i < n; i++) y[i] = MathF.Acos(x[i]); break;
            case UnaryKind.Atan: for (; i < n; i++) y[i] = MathF.Atan(x[i]); break;
            case UnaryKind.Sinh: for (; i < n; i++) y[i] = MathF.Sinh(x[i]); break;
            case UnaryKind.Cosh: for (; i < n; i++) y[i] = MathF.Cosh(x[i]); break;
            case UnaryKind.Asinh: for (; i < n; i++) y[i] = MathF.Asinh(x[i]); break;
            case UnaryKind.Acosh: for (; i < n; i++) y[i] = MathF.Acosh(x[i]); break;
            case UnaryKind.Atanh: for (; i < n; i++) y[i] = MathF.Atanh(x[i]); break;
            case UnaryKind.Floor: for (; i < n; i++) y[i] = MathF.Floor(x[i]); break;
            case UnaryKind.Ceil: for (; i < n; i++) y[i] = MathF.Ceiling(x[i]); break;
            case UnaryKind.Round: for (; i < n; i++) y[i] = MathF.Round(x[i], MidpointRounding.ToEven); break;
            case UnaryKind.Sign: for (; i < n; i++) y[i] = x[i] > 0 ? 1f : x[i] < 0 ? -1f : x[i]; break; // NaN and +-0 propagate
            case UnaryKind.Softsign: for (; i < n; i++) y[i] = x[i] / (1f + MathF.Abs(x[i])); break;
            case UnaryKind.Mish:
                for (; i < n; i++) { float v = x[i]; y[i] = v * MathF.Tanh(v > 20f ? v : (float)Math.Log(1.0 + Math.Exp(v))); }
                break;
            case UnaryKind.Elu: for (; i < n; i++) y[i] = x[i] >= 0 ? x[i] : scale * (MathF.Exp(x[i]) - 1f); break;
            case UnaryKind.Celu: for (; i < n; i++) y[i] = MathF.Max(0f, x[i]) + MathF.Min(0f, scale * (MathF.Exp(x[i] / scale) - 1f)); break;
            case UnaryKind.Selu: for (; i < n; i++) y[i] = x[i] > 0 ? bias * x[i] : bias * scale * (MathF.Exp(x[i]) - 1f); break; // scale = alpha, bias = gamma
            case UnaryKind.LeakyRelu: for (; i < n; i++) y[i] = x[i] >= 0 ? x[i] : scale * x[i]; break;
            case UnaryKind.ThresholdedRelu: for (; i < n; i++) y[i] = x[i] > scale ? x[i] : 0f; break;
            case UnaryKind.Not: for (; i < n; i++) y[i] = x[i] != 0 ? 0f : 1f; break;
            case UnaryKind.IsNaN: for (; i < n; i++) y[i] = float.IsNaN(x[i]) ? 1f : 0f; break;
            case UnaryKind.IsInf: for (; i < n; i++) y[i] = float.IsInfinity(x[i]) ? 1f : 0f; break;
            case UnaryKind.Erf:
                if (Vector256.IsHardwareAccelerated)
                    for (; i + 8 <= n; i += 8) ErfV(Vector256.Load(x + i)).Store(y + i);
                for (; i < n; i++) y[i] = ErfF(x[i]);
                break;
            case UnaryKind.HardSwish:
            {
                // x * clamp(x/6 + 0.5, 0, 1)
                const float a = 1f / 6f;
                if (Vector256.IsHardwareAccelerated)
                {
                    var av = Vector256.Create(a); var hv = Vector256.Create(0.5f); var zero = Vector256<float>.Zero; var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8)
                    {
                        var v = Vector256.Load(x + i);
                        (v * Vector256.Min(Vector256.Max(Vector256.FusedMultiplyAdd(v, av, hv), zero), one)).Store(y + i);
                    }
                }
                for (; i < n; i++) y[i] = x[i] * Math.Clamp(x[i] * a + 0.5f, 0f, 1f);
                break;
            }
            case UnaryKind.HardSigmoid:
                if (Vector256.IsHardwareAccelerated)
                {
                    var av = Vector256.Create(scale); var bv = Vector256.Create(bias); var zero = Vector256<float>.Zero; var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8)
                        Vector256.Min(Vector256.Max(Vector256.FusedMultiplyAdd(Vector256.Load(x + i), av, bv), zero), one).Store(y + i);
                }
                for (; i < n; i++) y[i] = Math.Clamp(x[i] * scale + bias, 0f, 1f);
                break;
            case UnaryKind.Clip: // scale = min, bias = max
                if (Vector256.IsHardwareAccelerated)
                {
                    var lo = Vector256.Create(scale); var hi = Vector256.Create(bias);
                    for (; i + 8 <= n; i += 8) Vector256.Min(Vector256.Max(Vector256.Load(x + i), lo), hi).Store(y + i);
                }
                for (; i < n; i++) y[i] = Math.Min(Math.Max(x[i], scale), bias);
                break;
            case UnaryKind.Neg: for (; i < n; i++) y[i] = -x[i]; break;
            case UnaryKind.Abs: for (; i < n; i++) y[i] = MathF.Abs(x[i]); break;
            case UnaryKind.Relu: for (; i < n; i++) y[i] = x[i] > 0 ? x[i] : 0; break;
            case UnaryKind.Square: for (; i < n; i++) y[i] = x[i] * x[i]; break;
            case UnaryKind.Recip: for (; i < n; i++) y[i] = 1f / x[i]; break;
            case UnaryKind.Sqrt: for (; i < n; i++) y[i] = MathF.Sqrt(x[i]); break;
            case UnaryKind.Rsqrt: for (; i < n; i++) y[i] = 1f / MathF.Sqrt(x[i]); break;
            case UnaryKind.Log: for (; i < n; i++) y[i] = MathF.Log(x[i]); break;
            case UnaryKind.Scale:
                if (Vector256.IsHardwareAccelerated)
                {
                    var sv = Vector256.Create(scale); var bv = Vector256.Create(bias);
                    for (; i + 8 <= n; i += 8) Vector256.FusedMultiplyAdd(Vector256.Load(x + i), sv, bv).Store(y + i);
                }
                for (; i < n; i++) y[i] = x[i] * scale + bias;
                break;
            case UnaryKind.Exp:
                if (Vector256.IsHardwareAccelerated)
                    for (; i + 8 <= n; i += 8) Vector256.Exp(Vector256.Load(x + i)).Store(y + i);
                for (; i < n; i++) y[i] = MathF.Exp(x[i]);
                break;
            case UnaryKind.Sigmoid:
                if (Vector256.IsHardwareAccelerated)
                {
                    var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8) (one / (one + Vector256.Exp(-Vector256.Load(x + i)))).Store(y + i);
                }
                for (; i < n; i++) y[i] = 1f / (1f + MathF.Exp(-x[i]));
                break;
            case UnaryKind.Silu:
                if (Vector256.IsHardwareAccelerated)
                {
                    var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8) { var v = Vector256.Load(x + i); (v / (one + Vector256.Exp(-v))).Store(y + i); }
                }
                for (; i < n; i++) y[i] = x[i] / (1f + MathF.Exp(-x[i]));
                break;
            case UnaryKind.Tanh:
                for (; i < n; i++) y[i] = MathF.Tanh(x[i]);
                break;
            case UnaryKind.GeluTanh:
                // 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))) = x * sigmoid(2u), u = sqrt(2/pi)(x + 0.044715 x^3)
                if (Vector256.IsHardwareAccelerated)
                {
                    var one = Vector256<float>.One; var c0 = Vector256.Create(0.7978845608028654f * 2f); var c1 = Vector256.Create(0.044715f);
                    for (; i + 8 <= n; i += 8)
                    {
                        var v = Vector256.Load(x + i);
                        var u = c0 * Vector256.FusedMultiplyAdd(c1 * v, v * v, v);
                        (v / (one + Vector256.Exp(-u))).Store(y + i);
                    }
                }
                for (; i < n; i++)
                {
                    float v = x[i];
                    y[i] = 0.5f * v * (1f + MathF.Tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
                }
                break;
            case UnaryKind.Gelu:
                if (Vector256.IsHardwareAccelerated)
                {
                    var half = Vector256.Create(0.5f); var one = Vector256<float>.One; var r2 = Vector256.Create(0.7071067811865476f);
                    for (; i + 8 <= n; i += 8) { var v = Vector256.Load(x + i); (half * v * (one + ErfV(v * r2))).Store(y + i); }
                }
                for (; i < n; i++) { double v = x[i]; y[i] = (float)(0.5 * v * (1.0 + Erf(v * 0.7071067811865476))); }
                break;
            case UnaryKind.Softplus:
                for (; i < n; i++)
                {
                    float v = x[i];
                    // llama.cpp: x > 20 ? x : log1p(exp(x)); evaluated in double for log1p accuracy.
                    y[i] = v > 20f ? v : (float)Math.Log(1.0 + Math.Exp(v));
                }
                break;
        }
    }
}

internal sealed unsafe class UnaryOp : CpuOp
{
    readonly UnaryKind _kind; readonly TRef _x, _y; readonly float _scale, _bias;
    readonly Action<int> _body;
    RunParameters _p = null!; int _chunks; long _perChunk; long _total; long _cols;
    public override string Name => "Unary." + _kind;
    public UnaryOp(UnaryKind kind, TRef x, TRef y, float scale = 1, float bias = 0)
    {
        _kind = kind; _x = x; _y = y; _scale = scale; _bias = bias; _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _p = p;
        _total = _x.Elements(p);
        if (_total == 0) return;
        _cols = _x.Cols(p);
        if (_x.IsDense)
        {
            _chunks = Par.Chunks(_total, 32768);
            _perChunk = (_total + _chunks - 1) / _chunks;
        }
        else
        {
            long rows = _x.Rows(p);
            _chunks = Par.Chunks(rows, Math.Max(1, 32768 / Math.Max(1, _cols)));
            _perChunk = (rows + _chunks - 1) / _chunks;
        }
        Par.For(_chunks, _body);
    }

    void Body(int c)
    {
        if (_x.IsDense)
        {
            long s = c * _perChunk, e = Math.Min(_total, s + _perChunk);
            if (s < e) Math2.Unary(_kind, (float*)_x.P + s, (float*)_y.P + s, e - s, _scale, _bias);
        }
        else
        {
            long rows = _total / _cols;
            long s = c * _perChunk, e = Math.Min(rows, s + _perChunk);
            for (long r = s; r < e; r++) Math2.Unary(_kind, _x.F(r), (float*)_y.P + r * _cols, _cols, _scale, _bias);
        }
    }
}

internal enum BinaryKind { Add, Sub, Mul, Div, Max, Min, Pow, Mod, PRelu, Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual, And, Or, Xor }

internal sealed unsafe class BinaryOp : CpuOp
{
    readonly BinaryKind _kind; readonly bool _fmod; readonly TRef _a, _b, _y;
    readonly long[] _sa, _sb; // broadcast strides aligned to output rank
    readonly Action<int> _body;
    RunParameters _p = null!; long _outer, _perChunk; long _l;
    public override string Name => "Binary." + _kind;

    public BinaryOp(BinaryKind kind, TRef a, TRef b, TRef y, bool fmod = false)
    {
        _fmod = fmod; _kind = kind; _a = a; _b = b; _y = y; _body = Body;
        _sa = Align(a, y); _sb = Align(b, y);
    }

    static long[] Align(TRef t, TRef y)
    {
        int r = y.Rank; var s = new long[r];
        for (int k = 0; k < r; k++)
        {
            int ik = k - (r - t.Rank);
            if (ik < 0) s[k] = 0;
            else if (t.Dims[ik] == 1 && !(ik == 0 && t.Sym is not null)) s[k] = 0;
            else s[k] = t.Strides[ik];
        }
        return s;
    }

    public override void Run(RunParameters p)
    {
        _p = p;
        long total = _y.Elements(p);
        if (total == 0) return;
        _l = _y.Cols(p);
        _outer = total / _l;
        int chunks = Par.Chunks(total, 32768);
        chunks = (int)Math.Min(chunks, _outer);
        _perChunk = (_outer + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    void Body(int c)
    {
        long s = c * _perChunk, e = Math.Min(_outer, s + _perChunk);
        int rank = _y.Rank;
        long l = _l;
        long sa = _sa[rank - 1], sb = _sb[rank - 1];
        for (long r = s; r < e; r++)
        {
            long ao = 0, bo = 0, rr = r;
            for (int k = rank - 2; k >= 1; k--)
            {
                long d = _y.Dims[k]; long q = rr / d; long ix = rr - q * d; rr = q;
                ao += ix * _sa[k]; bo += ix * _sb[k];
            }
            if (rank >= 2) { ao += rr * _sa[0]; bo += rr * _sb[0]; }
            float* pa = (float*)_a.P + ao; float* pb = (float*)_b.P + bo; float* py = (float*)_y.P + r * l;
            Apply(_kind, pa, sa, pb, sb, py, l, _fmod);
        }
    }

    static void Apply(BinaryKind k, float* a, long sa, float* b, long sb, float* y, long n, bool fmod)
    {
        long i = 0;
        if (k >= BinaryKind.Mod)
        {
            for (; i < n; i++) y[i] = OpS2(k, a[i * sa], b[i * sb], fmod);
            return;
        }
        if (k == BinaryKind.Pow)
        {
            if (sb == 0)
            {
                float e = *b;
                if (e == 2f) { for (; i < n; i++) { float v = a[i * sa]; y[i] = v * v; } }
                else if (e == 1f) { for (; i < n; i++) y[i] = a[i * sa]; }
                else if (e == 0.5f) { for (; i < n; i++) y[i] = MathF.Sqrt(a[i * sa]); }
                else if (e == 3f) { for (; i < n; i++) { float v = a[i * sa]; y[i] = v * v * v; } }
                else for (; i < n; i++) y[i] = MathF.Pow(a[i * sa], e);
            }
            else for (; i < n; i++) y[i] = MathF.Pow(a[i * sa], b[i * sb]);
            return;
        }
        if (Vector256.IsHardwareAccelerated && (sa == 1 || sa == 0) && (sb == 1 || sb == 0))
        {
            if (sa == 1 && sb == 1)
            {
                for (; i + 8 <= n; i += 8)
                    Op(k, Vector256.Load(a + i), Vector256.Load(b + i)).Store(y + i);
            }
            else if (sa == 1)
            {
                var bv = Vector256.Create(*b);
                for (; i + 8 <= n; i += 8) Op(k, Vector256.Load(a + i), bv).Store(y + i);
            }
            else if (sb == 1)
            {
                var av = Vector256.Create(*a);
                for (; i + 8 <= n; i += 8) Op(k, av, Vector256.Load(b + i)).Store(y + i);
            }
            else
            {
                float v = OpS(k, *a, *b);
                for (; i < n; i++) y[i] = v;
            }
        }
        for (; i < n; i++) y[i] = OpS(k, a[i * sa], b[i * sb]);
    }

    static Vector256<float> Op(BinaryKind k, Vector256<float> a, Vector256<float> b) => k switch
    {
        BinaryKind.Add => a + b,
        BinaryKind.Sub => a - b,
        BinaryKind.Mul => a * b,
        BinaryKind.Div => a / b,
        BinaryKind.Max => Vector256.Max(a, b),
        _ => Vector256.Min(a, b),
    };

    static float OpS2(BinaryKind k, float a, float b, bool fmod)
    {
        switch (k)
        {
            case BinaryKind.Mod:
            {
                float r = a % b; // C fmod (sign of dividend)
                if (!fmod && r != 0 && (r < 0) != (b < 0)) r += b;
                return r;
            }
            case BinaryKind.PRelu: return a < 0 ? a * b : a;
            case BinaryKind.Equal: return a == b ? 1f : 0f;
            case BinaryKind.NotEqual: return a != b ? 1f : 0f;
            case BinaryKind.Less: return a < b ? 1f : 0f;
            case BinaryKind.LessEqual: return a <= b ? 1f : 0f;
            case BinaryKind.Greater: return a > b ? 1f : 0f;
            case BinaryKind.GreaterEqual: return a >= b ? 1f : 0f;
            case BinaryKind.And: return a != 0 && b != 0 ? 1f : 0f;
            case BinaryKind.Or: return a != 0 || b != 0 ? 1f : 0f;
            default: return (a != 0) != (b != 0) ? 1f : 0f;
        }
    }

    static float OpS(BinaryKind k, float a, float b) => k switch
    {
        BinaryKind.Add => a + b,
        BinaryKind.Sub => a - b,
        BinaryKind.Mul => a * b,
        BinaryKind.Div => a / b,
        BinaryKind.Max => MathF.Max(a, b),
        BinaryKind.Pow => MathF.Pow(a, b),
        _ => MathF.Min(a, b),
    };
}

/// <summary>Materializes (possibly strided) data into a dense destination; also implements Contiguous.</summary>
internal sealed unsafe class CopyOp : CpuOp
{
    readonly TRef _x, _y; readonly Action<int> _body;
    long _chunkRows; long _d0;
    public override string Name => "Copy";
    public CopyOp(TRef x, TRef y) { _x = x; _y = y; _body = Body; }

    public override void Run(RunParameters p)
    {
        long d0 = _x.Rank == 0 ? 1 : _x.Dim(0, p);
        _d0 = d0;
        long total = _x.Elements(p);
        if (total == 0) return;
        if (_x.Rank >= 2 && total > 65536)
        {
            int chunks = (int)Math.Min(d0, Par.Chunks(total, 32768));
            _chunkRows = (d0 + chunks - 1) / chunks;
            Par.For(chunks, _body);
        }
        else
        {
            _chunkRows = d0;
            Body(0);
        }
    }

    void Body(int c)
    {
        if (_x.Rank == 0) { Mem.CopyStrided(_x.P, _x.Strides, _y.P, _y.Strides, _x.Dims, 1, _x.ElemSize); return; }
        long s = c * _chunkRows, e = Math.Min(_d0, s + _chunkRows);
        if (s >= e) return;
        int es = _x.ElemSize;
        Mem.CopyStrided(_x.P + s * _x.Strides[0] * es, _x.Strides, _y.P + s * _y.Strides[0] * es, _y.Strides, _x.Dims, e - s, es);
    }
}

internal sealed unsafe class CastOp : CpuOp
{
    readonly TRef _x, _y; readonly Action<int> _body;
    long _rows, _cols, _per;
    public override string Name => $"Cast.{_x.Type}->{_y.Type}";
    public CastOp(TRef x, TRef y) { _x = x; _y = y; _body = Body; }

    public override void Run(RunParameters p)
    {
        _rows = _x.Rows(p); _cols = _x.Cols(p);
        if (_rows * _cols == 0) return;
        int chunks = (int)Math.Min(_rows, Par.Chunks(_rows * _cols, 32768));
        _per = (_rows + chunks - 1) / chunks;
        Par.For(chunks, _body);
    }

    static float Load(DataType t, byte* p, long i)
    {
        if (t == DataType.F32) return ((float*)p)[i];
        if (t == DataType.F16) return Simd.HalfToFloat(((ushort*)p)[i]);
        if (t == DataType.BF16) return Simd.BFloatToFloat(((ushort*)p)[i]);
        if (t == DataType.I32) return ((int*)p)[i];
        if (t == DataType.I64) return ((long*)p)[i];
        if (t == DataType.I8) return ((sbyte*)p)[i];
        if (t == DataType.U8) return p[i];
        if (t == DataType.Bool) return p[i] != 0 ? 1 : 0;
        throw new NotSupportedException($"Cast from {t}");
    }

    static void Store(DataType t, byte* p, long i, float v)
    {
        if (t == DataType.F32) ((float*)p)[i] = v;
        else if (t == DataType.F16) ((ushort*)p)[i] = Simd.FloatToHalf(v);
        else if (t == DataType.BF16) ((ushort*)p)[i] = Simd.FloatToBFloat(v);
        else if (t == DataType.I32) ((int*)p)[i] = (int)v;
        else if (t == DataType.I64) ((long*)p)[i] = (long)v;
        else if (t == DataType.I8) ((sbyte*)p)[i] = (sbyte)v;
        else if (t == DataType.U8) p[i] = (byte)v;
        else if (t == DataType.Bool) p[i] = (byte)(v != 0 ? 1 : 0);
        else throw new NotSupportedException($"Cast to {t}");
    }

    void Body(int c)
    {
        long s = c * _per, e = Math.Min(_rows, s + _per);
        var st = _x.Type; var dt = _y.Type;
        for (long r = s; r < e; r++)
        {
            byte* src = _x.RowPtr(r); byte* dst = _y.P + r * _cols * dt.BlockBytes;
            if (st == DataType.F32 && dt == DataType.F32) Buffer.MemoryCopy(src, dst, _cols * 4, _cols * 4);
            else if (st == DataType.I32 && dt == DataType.I32) Buffer.MemoryCopy(src, dst, _cols * 4, _cols * 4);
            else if (st == DataType.F16 && dt == DataType.F32) Simd.HalfToFloat((ushort*)src, (float*)dst, _cols);
            else if (st == DataType.BF16 && dt == DataType.F32) Simd.BFloatToFloat((ushort*)src, (float*)dst, _cols);
            else if (st == DataType.I32 && dt == DataType.I64) for (long i = 0; i < _cols; i++) ((long*)dst)[i] = ((int*)src)[i];
            else if (st == DataType.I64 && dt == DataType.I32) for (long i = 0; i < _cols; i++) ((int*)dst)[i] = (int)((long*)src)[i];
            else for (long i = 0; i < _cols; i++) Store(dt, dst, i, Load(st, src, i));
        }
    }
}
