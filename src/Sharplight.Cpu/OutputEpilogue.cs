using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

/// <summary>
/// Elementwise ops fused into a convolution's or matmul's output while its tile is still in cache:
/// y = act(y + columnBias + residual). The column bias is a constant vector over the output's last axis (a MatMul's bias
/// Add), the residual an addend with the output's dense layout; the activation (Relu, Clip, SiLU, ...) gets the
/// parameters <see cref="UnaryOp"/> would get. Saves one to three full passes over the activation per op.
/// </summary>
internal sealed unsafe class OutputEpilogue
{
    /// <summary>Constant f32 vector of <see cref="Columns"/> values added along the output's last axis, or null.</summary>
    public TRef? ColumnBias { get; init; }

    /// <summary>Length of the output's last axis (the period of <see cref="ColumnBias"/>).</summary>
    public long Columns { get; init; } = 1;

    /// <summary>Addend with the output's dims (dense), or null.</summary>
    public TRef? Residual { get; init; }

    /// <summary>Activation applied after the additions, or null.</summary>
    public UnaryKind? Activation { get; init; }

    /// <summary>First activation parameter (Clip: min, LeakyRelu: alpha, ...).</summary>
    public float Scale { get; init; } = 1;

    /// <summary>Second activation parameter (Clip: max, ...).</summary>
    public float Bias { get; init; }

    /// <summary>Profile name suffix, e.g. "+Add+Relu".</summary>
    public string Label => (ColumnBias is null ? "" : "+Bias") + (Residual is null ? "" : "+Add") + (Activation is { } a ? "+" + a : "");

    /// <summary>
    /// Applies the epilogue to <paramref name="n"/> outputs at <paramref name="y"/>, element <paramref name="offset"/> of the output
    /// tensor. With a <see cref="ColumnBias"/> the range must not cross the end of an output row.
    /// </summary>
    public void Apply(float* y, long offset, long n)
    {
        if (ColumnBias is { } bias)
        {
            AddInPlace(y, (float*)bias.P + offset % Columns, n);
        }

        if (Residual is { } residual)
        {
            AddInPlace(y, (float*)residual.P + offset, n);
        }

        if (Activation is { } kind)
        {
            ScalarMath.Unary(kind, y, y, n, Scale, Bias);
        }
    }

    /// <summary>Applies the epilogue to whole output rows [<paramref name="row0"/>, <paramref name="row1"/>) of a dense [rows, <see cref="Columns"/>] output.</summary>
    public void ApplyRows(float* y, long row0, long row1)
    {
        for (long r = row0; r < row1; r++)
        {
            Apply(y + r * Columns, r * Columns, Columns);
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
