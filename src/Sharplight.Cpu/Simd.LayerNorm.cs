using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>LayerNorm row kernels: one-pass statistics and the normalize-scale-shift pass.</summary>
internal static unsafe partial class Simd
{
    /// <summary>
    /// Mean and (population) variance of a row in one pass. The values are widened to f64 and shifted by the pivot
    /// <c>x[0]</c> (exact: the difference of two floats fits a double), and their sum and sum of squares are accumulated
    /// in f64 lanes, so rows with a large mean (mean 100, std 1) keep the accuracy of a two-pass f64 variance.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void MeanVar(float* x, long n, out float mean, out float variance)
    {
        float k = x[0];
        long i = 0;
        double s = 0;
        double q = 0;
        if (CpuIsa.Use512)
        {
            var kv = Vector512.Create((double)k);
            var s0 = Vector512<double>.Zero;
            var s1 = s0;
            var q0 = s0;
            var q1 = s0;
            for (; i + 16 <= n; i += 16)
            {
                var (lo, hi) = Vector512.Widen(Vector512.Load(x + i));
                lo -= kv;
                hi -= kv;
                s0 += lo;
                s1 += hi;
                q0 = Vector512.FusedMultiplyAdd(lo, lo, q0);
                q1 = Vector512.FusedMultiplyAdd(hi, hi, q1);
            }
            s = Vector512.Sum(s0 + s1);
            q = Vector512.Sum(q0 + q1);
        }
        else if (CpuIsa.Use256)
        {
            var kv = Vector256.Create((double)k);
            var s0 = Vector256<double>.Zero;
            var s1 = s0;
            var q0 = s0;
            var q1 = s0;
            for (; i + 8 <= n; i += 8)
            {
                var (lo, hi) = Vector256.Widen(Vector256.Load(x + i));
                lo -= kv;
                hi -= kv;
                s0 += lo;
                s1 += hi;
                q0 = Vector256.FusedMultiplyAdd(lo, lo, q0);
                q1 = Vector256.FusedMultiplyAdd(hi, hi, q1);
            }
            s = Vector256.Sum(s0 + s1);
            q = Vector256.Sum(q0 + q1);
        }
        else if (CpuIsa.Use128 || CpuIsa.UseNeon)
        {
            var kv = Vector128.Create((double)k);
            var s0 = Vector128<double>.Zero;
            var s1 = s0;
            var q0 = s0;
            var q1 = s0;
            for (; i + 4 <= n; i += 4)
            {
                var (lo, hi) = Vector128.Widen(Vector128.Load(x + i));
                lo -= kv;
                hi -= kv;
                s0 += lo;
                s1 += hi;
                q0 += lo * lo;
                q1 += hi * hi;
            }
            s = Vector128.Sum(s0 + s1);
            q = Vector128.Sum(q0 + q1);
        }
        for (; i < n; i++)
        {
            double d = (double)x[i] - k;
            s += d;
            q += d * d;
        }

        double md = s / n;
        mean = (float)(k + md);
        variance = (float)Math.Max(0, q / n - md * md);
    }

    /// <summary>
    /// y[i] = (x[i] - mean) * inv * w[i] + b[i]; <paramref name="w"/> and <paramref name="b"/> may each be null (factor 1,
    /// offset 0). <paramref name="y"/> may alias <paramref name="x"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Normalize(float* x, float* y, float mean, float inv, float* w, float* b, long n)
    {
        long i = 0;
        if (CpuIsa.Use512)
        {
            var mv = Vector512.Create(mean);
            var iv = Vector512.Create(inv);
            if (w != null && b != null)
            {
                for (; i + 16 <= n; i += 16)
                {
                    Vector512.FusedMultiplyAdd((Vector512.Load(x + i) - mv) * iv, Vector512.Load(w + i), Vector512.Load(b + i)).Store(y + i);
                }
            }
            else if (w != null)
            {
                for (; i + 16 <= n; i += 16)
                {
                    ((Vector512.Load(x + i) - mv) * iv * Vector512.Load(w + i)).Store(y + i);
                }
            }
            else if (b != null)
            {
                for (; i + 16 <= n; i += 16)
                {
                    Vector512.FusedMultiplyAdd(Vector512.Load(x + i) - mv, iv, Vector512.Load(b + i)).Store(y + i);
                }
            }
            else
            {
                for (; i + 16 <= n; i += 16)
                {
                    ((Vector512.Load(x + i) - mv) * iv).Store(y + i);
                }
            }
        }
        else if (CpuIsa.Use256)
        {
            var mv = Vector256.Create(mean);
            var iv = Vector256.Create(inv);
            if (w != null && b != null)
            {
                for (; i + 8 <= n; i += 8)
                {
                    Vector256.FusedMultiplyAdd((Vector256.Load(x + i) - mv) * iv, Vector256.Load(w + i), Vector256.Load(b + i)).Store(y + i);
                }
            }
            else if (w != null)
            {
                for (; i + 8 <= n; i += 8)
                {
                    ((Vector256.Load(x + i) - mv) * iv * Vector256.Load(w + i)).Store(y + i);
                }
            }
            else if (b != null)
            {
                for (; i + 8 <= n; i += 8)
                {
                    Vector256.FusedMultiplyAdd(Vector256.Load(x + i) - mv, iv, Vector256.Load(b + i)).Store(y + i);
                }
            }
            else
            {
                for (; i + 8 <= n; i += 8)
                {
                    ((Vector256.Load(x + i) - mv) * iv).Store(y + i);
                }
            }
        }
        else if (CpuIsa.Use128 || CpuIsa.UseNeon)
        {
            // No fused multiply-add here: the 128-bit tier doesn't assume FMA hardware (a software fallback is slow).
            var mv = Vector128.Create(mean);
            var iv = Vector128.Create(inv);
            if (w != null && b != null)
            {
                for (; i + 4 <= n; i += 4)
                {
                    ((Vector128.Load(x + i) - mv) * iv * Vector128.Load(w + i) + Vector128.Load(b + i)).Store(y + i);
                }
            }
            else if (w != null)
            {
                for (; i + 4 <= n; i += 4)
                {
                    ((Vector128.Load(x + i) - mv) * iv * Vector128.Load(w + i)).Store(y + i);
                }
            }
            else if (b != null)
            {
                for (; i + 4 <= n; i += 4)
                {
                    ((Vector128.Load(x + i) - mv) * iv + Vector128.Load(b + i)).Store(y + i);
                }
            }
            else
            {
                for (; i + 4 <= n; i += 4)
                {
                    ((Vector128.Load(x + i) - mv) * iv).Store(y + i);
                }
            }
        }
        for (; i < n; i++)
        {
            float o = (x[i] - mean) * inv;
            if (w != null)
            {
                o *= w[i];
            }

            if (b != null)
            {
                o += b[i];
            }

            y[i] = o;
        }
    }
}
