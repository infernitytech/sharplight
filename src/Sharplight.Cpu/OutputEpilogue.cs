using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>
/// Elementwise ops fused into a convolution's output while its tile is still in cache: an optional residual addend with
/// the output's dense layout (y = conv + residual), then an optional unary activation (Relu, Clip, SiLU, ...), with the
/// parameters <see cref="UnaryOp"/> would get. Saves one or two full passes over the activation per conv.
/// </summary>
internal sealed unsafe class ConvEpilogue
{
    /// <summary>Addend with the conv output's dims (dense), or null.</summary>
    public TRef? Residual { get; init; }

    /// <summary>Activation applied after the residual, or null.</summary>
    public UnaryKind? Activation { get; init; }

    /// <summary>First activation parameter (Clip: min, LeakyRelu: alpha, ...).</summary>
    public float Scale { get; init; } = 1;

    /// <summary>Second activation parameter (Clip: max, ...).</summary>
    public float Bias { get; init; }

    /// <summary>Profile name suffix, e.g. "+Add+Relu".</summary>
    public string Label => (Residual is null ? "" : "+Add") + (Activation is { } a ? "+" + a : "");

    /// <summary>Applies the epilogue to <paramref name="n"/> outputs at <paramref name="y"/>, element <paramref name="offset"/> of the output tensor.</summary>
    public void Apply(float* y, long offset, long n)
    {
        if (Residual is { } residual)
        {
            AddInPlace(y, (float*)residual.P + offset, n);
        }

        if (Activation is { } kind)
        {
            ScalarMath.Unary(kind, y, y, n, Scale, Bias);
        }
    }

    private static void AddInPlace(float* y, float* r, long n)
    {
        long i = 0;
        if (CpuIsa.Use512)
        {
            for (; i + 16 <= n; i += 16)
            {
                (Vector512.Load(y + i) + Vector512.Load(r + i)).Store(y + i);
            }
        }

        if (CpuIsa.Use256)
        {
            for (; i + 8 <= n; i += 8)
            {
                (Vector256.Load(y + i) + Vector256.Load(r + i)).Store(y + i);
            }
        }

        for (; i < n; i++)
        {
            y[i] += r[i];
        }
    }
}
