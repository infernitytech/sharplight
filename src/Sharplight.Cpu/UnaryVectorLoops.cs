using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>
/// The vector parts of the transcendental unary kernels of <see cref="ScalarMath.Unary"/>, one method each: inside one large method the
/// JIT runs out of inlining budget and calls <see cref="VectorExp"/> and <see cref="VectorActivations"/> out of line in the later cases
/// (SiLU took 34 instead of 6 µs per 8192 values). Each returns the index of the first value it left to the scalar tail.
/// </summary>
internal static unsafe class UnaryVectorLoops
{
    /// <summary>The vector part of <see cref="UnaryKind.Mish"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Mish(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use256)
        {
            for (; i + 8 <= n; i += 8)
            {
                VectorActivations.Mish(Vector256.Load(x + i)).Store(y + i);
            }
        }
        else if (CpuIsa.UseNeon)
        {
            for (; i + 4 <= n; i += 4)
            {
                VectorActivations.Mish(Vector128.Load(x + i)).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.Erf"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Erf(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use256)
        {
            for (; i + 8 <= n; i += 8)
            {
                ScalarMath.ErfV(Vector256.Load(x + i)).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.Exp"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Exp(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use512)
        {
            for (; i + 16 <= n; i += 16)
            {
                VectorExp.Exp(Vector512.Load(x + i)).Store(y + i);
            }
        }
        if (CpuIsa.Use256)
        {
            for (; i + 8 <= n; i += 8)
            {
                VectorExp.Exp(Vector256.Load(x + i)).Store(y + i);
            }
        }
        else if (CpuIsa.UseNeon)
        {
            for (; i + 4 <= n; i += 4)
            {
                VectorExp.Exp(Vector128.Load(x + i)).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.Sigmoid"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Sigmoid(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use512)
        {
            var one = Vector512<float>.One;
            for (; i + 16 <= n; i += 16)
            {
                (one / (one + VectorExp.Exp(-Vector512.Load(x + i)))).Store(y + i);
            }
        }
        if (CpuIsa.Use256)
        {
            var one = Vector256<float>.One;
            for (; i + 8 <= n; i += 8)
            {
                (one / (one + VectorExp.Exp(-Vector256.Load(x + i)))).Store(y + i);
            }
        }
        else if (CpuIsa.UseNeon)
        {
            var one = Vector128<float>.One;
            for (; i + 4 <= n; i += 4)
            {
                (one / (one + VectorExp.Exp(-Vector128.Load(x + i)))).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.Silu"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Silu(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use512)
        {
            var one = Vector512<float>.One;
            for (; i + 16 <= n; i += 16)
            {
                var v = Vector512.Load(x + i);
                (v / (one + VectorExp.Exp(-v))).Store(y + i);
            }
        }
        if (CpuIsa.Use256)
        {
            var one = Vector256<float>.One;
            for (; i + 8 <= n; i += 8)
            {
                var v = Vector256.Load(x + i);
                (v / (one + VectorExp.Exp(-v))).Store(y + i);
            }
        }
        else if (CpuIsa.UseNeon)
        {
            var one = Vector128<float>.One;
            for (; i + 4 <= n; i += 4)
            {
                var v = Vector128.Load(x + i);
                (v / (one + VectorExp.Exp(-v))).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.Tanh"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Tanh(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use256)
        {
            for (; i + 8 <= n; i += 8)
            {
                VectorActivations.Tanh(Vector256.Load(x + i)).Store(y + i);
            }
        }
        else if (CpuIsa.UseNeon)
        {
            for (; i + 4 <= n; i += 4)
            {
                VectorActivations.Tanh(Vector128.Load(x + i)).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.GeluTanh"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long GeluTanh(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use256)
        {
            var one = Vector256<float>.One;
            var c0 = Vector256.Create(0.7978845608028654f * 2f);
            var c1 = Vector256.Create(0.044715f);
            for (; i + 8 <= n; i += 8)
            {
                var v = Vector256.Load(x + i);
                var u = c0 * Vector256.FusedMultiplyAdd(c1 * v, v * v, v);
                (v / (one + VectorExp.Exp(-u))).Store(y + i);
            }
        }
        else if (CpuIsa.UseNeon)
        {
            var one = Vector128<float>.One;
            var c0 = Vector128.Create(0.7978845608028654f * 2f);
            var c1 = Vector128.Create(0.044715f);
            for (; i + 4 <= n; i += 4)
            {
                var v = Vector128.Load(x + i);
                var u = c0 * Vector128.FusedMultiplyAdd(c1 * v, v * v, v);
                (v / (one + VectorExp.Exp(-u))).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.Gelu"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Gelu(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use256)
        {
            var half = Vector256.Create(0.5f);
            var one = Vector256<float>.One;
            var r2 = Vector256.Create(0.7071067811865476f);
            for (; i + 8 <= n; i += 8)
            {
                var v = Vector256.Load(x + i);
                (half * v * (one + ScalarMath.ErfV(v * r2))).Store(y + i);
            }
        }

        return i;
    }

    /// <summary>The vector part of <see cref="UnaryKind.Softplus"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static long Softplus(float* x, float* y, long n)
    {
        long i = 0;
        if (CpuIsa.Use256)
        {
            for (; i + 8 <= n; i += 8)
            {
                VectorActivations.Softplus(Vector256.Load(x + i)).Store(y + i);
            }
        }
        else if (CpuIsa.UseNeon)
        {
            for (; i + 4 <= n; i += 4)
            {
                VectorActivations.Softplus(Vector128.Load(x + i)).Store(y + i);
            }
        }

        return i;
    }
}
