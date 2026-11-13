using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>Scalar math helpers used by the elementwise kernels where the BCL has no equivalent (erf and friends).</summary>
internal static unsafe class ScalarMath
{
    /// <summary>erf in double precision (series for small |x|, continued fraction for the tail).</summary>
    public static double Erf(double x)
    {
        double ax = Math.Abs(x);
        double r;
        if (ax < 2.5)
        {
            // erf(x) = 2/sqrt(pi) * exp(-x^2) * sum_{n>=0} 2^n x^(2n+1) / (1*3*...*(2n+1))   (all terms positive)
            double x2 = ax * ax;
            double term = ax;
            double sum = ax;
            for (int n = 1; n < 200; n++)
            {
                term *= 2 * x2 / (2 * n + 1);
                sum += term;
                if (term < sum * 1e-17)
                {
                    break;
                }
            }
            r = 1.1283791670955126 * Math.Exp(-x2) * sum;
        }
        else if (ax > 6)
        {
            r = 1.0;
        }
        else
        {
            // erfc(x) = exp(-x^2)/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
            double f = ax;
            for (int k = 80; k >= 1; k--)
            {
                f = ax + 0.5 * k / f;
            }

            r = 1.0 - Math.Exp(-ax * ax) / (1.7724538509055159 * f);
        }
        return x < 0 ? -r : r;
    }

    // Rational erf (Eigen/XLA style), max abs error ~1e-7 in f32; |x| is clamped where erf == +-1 to f32 precision.
    private const float ErfClamp = 3.832506856900711f;
    private const double A1 = -1.60960333262415e-02, A3 = -2.95459980854025e-03, A5 = -7.34990630326855e-04, A7 = -5.69250639462346e-05,
        A9 = -2.10102402082508e-06, A11 = 2.77068142495902e-08, A13 = -2.72614225801306e-10,
        B0 = -1.42647390514189e-02, B2 = -7.37332916720468e-03, B4 = -1.68282697438203e-03, B6 = -2.13374055278905e-04, B8 = -1.45660718464996e-05;

    // The rational is evaluated in double precision (f32 evaluation loses ~2e-7 to cancellation); the approximation error is ~9e-8.
    public static float ErfF(float x)
    {
        double xd = Math.Clamp((double)x, -ErfClamp, ErfClamp);
        double x2 = xd * xd;
        double p = A13;
        p = p * x2 + A11;
        p = p * x2 + A9;
        p = p * x2 + A7;
        p = p * x2 + A5;
        p = p * x2 + A3;
        p = p * x2 + A1;
        p *= xd;
        double q = B8;
        q = q * x2 + B6;
        q = q * x2 + B4;
        q = q * x2 + B2;
        q = q * x2 + B0;
        return (float)(p / q);
    }

    private static Vector256<double> ErfD(Vector256<double> x)
    {
        x = Vector256.Min(Vector256.Max(x, Vector256.Create((double)-ErfClamp)), Vector256.Create((double)ErfClamp));
        var x2 = x * x;
        var p = Vector256.Create((double)A13);
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A11));
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A9));
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A7));
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A5));
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A3));
        p = Vector256.FusedMultiplyAdd(p, x2, Vector256.Create((double)A1));
        p *= x;
        var q = Vector256.Create((double)B8);
        q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B6));
        q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B4));
        q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B2));
        q = Vector256.FusedMultiplyAdd(q, x2, Vector256.Create((double)B0));
        return p / q;
    }

    public static Vector256<float> ErfV(Vector256<float> x) =>
        Vector256.Narrow(ErfD(Vector256.WidenLower(x)), ErfD(Vector256.WidenUpper(x)));

    public static void Unary(UnaryKind k, float* x, float* y, long n, float scale, float bias)
    {
        long i = 0;
        switch (k)
        {
            case UnaryKind.Sin:
                for (; i < n; i++)
                {
                    y[i] = MathF.Sin(x[i]);
                }
                break;
            case UnaryKind.Cos:
                for (; i < n; i++)
                {
                    y[i] = MathF.Cos(x[i]);
                }
                break;
            case UnaryKind.Tan:
                for (; i < n; i++)
                {
                    y[i] = MathF.Tan(x[i]);
                }
                break;
            case UnaryKind.Asin:
                for (; i < n; i++)
                {
                    y[i] = MathF.Asin(x[i]);
                }
                break;
            case UnaryKind.Acos:
                for (; i < n; i++)
                {
                    y[i] = MathF.Acos(x[i]);
                }
                break;
            case UnaryKind.Atan:
                for (; i < n; i++)
                {
                    y[i] = MathF.Atan(x[i]);
                }
                break;
            case UnaryKind.Sinh:
                for (; i < n; i++)
                {
                    y[i] = MathF.Sinh(x[i]);
                }
                break;
            case UnaryKind.Cosh:
                for (; i < n; i++)
                {
                    y[i] = MathF.Cosh(x[i]);
                }
                break;
            case UnaryKind.Asinh:
                for (; i < n; i++)
                {
                    y[i] = MathF.Asinh(x[i]);
                }
                break;
            case UnaryKind.Acosh:
                for (; i < n; i++)
                {
                    y[i] = MathF.Acosh(x[i]);
                }
                break;
            case UnaryKind.Atanh:
                for (; i < n; i++)
                {
                    y[i] = MathF.Atanh(x[i]);
                }
                break;
            case UnaryKind.Floor:
                for (; i < n; i++)
                {
                    y[i] = MathF.Floor(x[i]);
                }
                break;
            case UnaryKind.Ceil:
                for (; i < n; i++)
                {
                    y[i] = MathF.Ceiling(x[i]);
                }
                break;
            case UnaryKind.Round:
                for (; i < n; i++)
                {
                    y[i] = MathF.Round(x[i], MidpointRounding.ToEven);
                }
                break;
            case UnaryKind.Sign:
                for (; i < n; i++)
                {
                    y[i] = x[i] > 0 ? 1f : x[i] < 0 ? -1f : x[i];
                }
                break; // NaN and +-0 propagate
            case UnaryKind.Softsign:
                for (; i < n; i++)
                {
                    y[i] = x[i] / (1f + MathF.Abs(x[i]));
                }
                break;
            case UnaryKind.Mish:
                for (; i < n; i++)
                {
                    float v = x[i];
                    y[i] = v * MathF.Tanh(v > 20f ? v : (float)Math.Log(1.0 + Math.Exp(v)));
                }
                break;
            case UnaryKind.Elu:
                for (; i < n; i++)
                {
                    y[i] = x[i] >= 0 ? x[i] : scale * (MathF.Exp(x[i]) - 1f);
                }
                break;
            case UnaryKind.Celu:
                for (; i < n; i++)
                {
                    y[i] = MathF.Max(0f, x[i]) + MathF.Min(0f, scale * (MathF.Exp(x[i] / scale) - 1f));
                }
                break;
            case UnaryKind.Selu:
                for (; i < n; i++)
                {
                    y[i] = x[i] > 0 ? bias * x[i] : bias * scale * (MathF.Exp(x[i]) - 1f);
                }
                break; // scale = alpha, bias = gamma
            case UnaryKind.LeakyRelu:
                for (; i < n; i++)
                {
                    y[i] = x[i] >= 0 ? x[i] : scale * x[i];
                }
                break;
            case UnaryKind.ThresholdedRelu:
                for (; i < n; i++)
                {
                    y[i] = x[i] > scale ? x[i] : 0f;
                }
                break;
            case UnaryKind.Not:
                for (; i < n; i++)
                {
                    y[i] = x[i] != 0 ? 0f : 1f;
                }
                break;
            case UnaryKind.IsNaN:
                for (; i < n; i++)
                {
                    y[i] = float.IsNaN(x[i]) ? 1f : 0f;
                }
                break;
            case UnaryKind.IsInf:
                for (; i < n; i++)
                {
                    y[i] = float.IsInfinity(x[i]) ? 1f : 0f;
                }
                break;
            case UnaryKind.Erf:
                if (Vector256.IsHardwareAccelerated)
                {
                    for (; i + 8 <= n; i += 8)
                    {
                        ErfV(Vector256.Load(x + i)).Store(y + i);
                    }
                }

                for (; i < n; i++)
                {
                    y[i] = ErfF(x[i]);
                }

                break;
            case UnaryKind.HardSwish:
            {
                // x * clamp(x/6 + 0.5, 0, 1)
                const float a = 1f / 6f;
                if (Vector256.IsHardwareAccelerated)
                {
                    var av = Vector256.Create(a);
                    var hv = Vector256.Create(0.5f);
                    var zero = Vector256<float>.Zero;
                    var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8)
                    {
                        var v = Vector256.Load(x + i);
                        (v * Vector256.Min(Vector256.Max(Vector256.FusedMultiplyAdd(v, av, hv), zero), one)).Store(y + i);
                    }
                }
                for (; i < n; i++)
                    {
                        y[i] = x[i] * Math.Clamp(x[i] * a + 0.5f, 0f, 1f);
                    }

                    break;
            }
            case UnaryKind.HardSigmoid:
                if (Vector256.IsHardwareAccelerated)
                {
                    var av = Vector256.Create(scale);
                    var bv = Vector256.Create(bias);
                    var zero = Vector256<float>.Zero;
                    var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8)
                    {
                        Vector256.Min(Vector256.Max(Vector256.FusedMultiplyAdd(Vector256.Load(x + i), av, bv), zero), one).Store(y + i);
                    }
                }
                for (; i < n; i++)
                {
                    y[i] = Math.Clamp(x[i] * scale + bias, 0f, 1f);
                }

                break;
            case UnaryKind.Clip:
                // scale = min, bias = max
                if (Vector256.IsHardwareAccelerated)
                {
                    var lo = Vector256.Create(scale);
                    var hi = Vector256.Create(bias);
                    for (; i + 8 <= n; i += 8)
                    {
                        Vector256.Min(Vector256.Max(Vector256.Load(x + i), lo), hi).Store(y + i);
                    }
                }
                for (; i < n; i++)
                {
                    y[i] = Math.Min(Math.Max(x[i], scale), bias);
                }

                break;
            case UnaryKind.Neg:
                for (; i < n; i++)
                {
                    y[i] = -x[i];
                }
                break;
            case UnaryKind.Abs:
                for (; i < n; i++)
                {
                    y[i] = MathF.Abs(x[i]);
                }
                break;
            case UnaryKind.Relu:
                for (; i < n; i++)
                {
                    y[i] = x[i] > 0 ? x[i] : 0;
                }
                break;
            case UnaryKind.Square:
                for (; i < n; i++)
                {
                    y[i] = x[i] * x[i];
                }
                break;
            case UnaryKind.Recip:
                for (; i < n; i++)
                {
                    y[i] = 1f / x[i];
                }
                break;
            case UnaryKind.Sqrt:
                for (; i < n; i++)
                {
                    y[i] = MathF.Sqrt(x[i]);
                }
                break;
            case UnaryKind.Rsqrt:
                for (; i < n; i++)
                {
                    y[i] = 1f / MathF.Sqrt(x[i]);
                }
                break;
            case UnaryKind.Log:
                for (; i < n; i++)
                {
                    y[i] = MathF.Log(x[i]);
                }
                break;
            case UnaryKind.Scale:
                if (Vector256.IsHardwareAccelerated)
                {
                    var sv = Vector256.Create(scale);
                    var bv = Vector256.Create(bias);
                    for (; i + 8 <= n; i += 8)
                    {
                        Vector256.FusedMultiplyAdd(Vector256.Load(x + i), sv, bv).Store(y + i);
                    }
                }
                for (; i < n; i++)
                {
                    y[i] = x[i] * scale + bias;
                }

                break;
            case UnaryKind.Exp:
                if (Vector256.IsHardwareAccelerated)
                {
                    for (; i + 8 <= n; i += 8)
                    {
                        Vector256.Exp(Vector256.Load(x + i)).Store(y + i);
                    }
                }

                for (; i < n; i++)
                {
                    y[i] = MathF.Exp(x[i]);
                }

                break;
            case UnaryKind.Sigmoid:
                if (Vector256.IsHardwareAccelerated)
                {
                    var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8)
                    {
                        (one / (one + Vector256.Exp(-Vector256.Load(x + i)))).Store(y + i);
                    }
                }
                for (; i < n; i++)
                {
                    y[i] = 1f / (1f + MathF.Exp(-x[i]));
                }

                break;
            case UnaryKind.Silu:
                if (Vector256.IsHardwareAccelerated)
                {
                    var one = Vector256<float>.One;
                    for (; i + 8 <= n; i += 8)
                    {
                        var v = Vector256.Load(x + i);
                        (v / (one + Vector256.Exp(-v))).Store(y + i);
                    }
                }
                for (; i < n; i++)
                {
                    y[i] = x[i] / (1f + MathF.Exp(-x[i]));
                }

                break;
            case UnaryKind.Tanh:
                for (; i < n; i++)
                {
                    y[i] = MathF.Tanh(x[i]);
                }

                break;
            case UnaryKind.GeluTanh:
                // 0.5 x (1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))) = x * sigmoid(2u), u = sqrt(2/pi)(x + 0.044715 x^3)
                if (Vector256.IsHardwareAccelerated)
                {
                    var one = Vector256<float>.One;
                    var c0 = Vector256.Create(0.7978845608028654f * 2f);
                    var c1 = Vector256.Create(0.044715f);
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
                    var half = Vector256.Create(0.5f);
                    var one = Vector256<float>.One;
                    var r2 = Vector256.Create(0.7071067811865476f);
                    for (; i + 8 <= n; i += 8)
                    {
                        var v = Vector256.Load(x + i);
                        (half * v * (one + ErfV(v * r2))).Store(y + i);
                    }
                }
                for (; i < n; i++)
                {
                    double v = x[i];
                    y[i] = (float)(0.5 * v * (1.0 + Erf(v * 0.7071067811865476)));
                }
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
