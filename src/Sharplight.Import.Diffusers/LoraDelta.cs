using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The weight difference one module of a LoRA-family file describes, as a dense [rows, columns] matrix (a convolution's kernel flattened
/// to [out, in · kh · kw]) before the file's alpha factor and the load strength: plain low-rank (LoRA / LoCon, with LyCORIS' Tucker core),
/// Hadamard product (LoHa), Kronecker product (LoKr) or a full difference.
/// </summary>
internal abstract class LoraDelta
{
    /// <summary>Output rows of the difference.</summary>
    public abstract long Rows { get; }

    /// <summary>Elements per output row.</summary>
    public abstract long Columns { get; }

    /// <summary>The rank the file's alpha is divided by, or null when the alpha does not apply (full matrices).</summary>
    public abstract long? Rank { get; }

    /// <summary>The kind, for messages (<c>LoRA</c>, <c>LoHa</c>, <c>LoKr</c>, <c>full</c>).</summary>
    public abstract string Kind { get; }

    /// <summary>The dense difference, row-major [<see cref="Rows"/>, <see cref="Columns"/>].</summary>
    public abstract float[] Compute();

    /// <summary>Product of [m, k] and [k, n] row-major matrices.</summary>
    protected static float[] Multiply(float[] left, float[] right, long m, long k, long n)
    {
        var result = new float[m * n];
        Parallel.For(0, (int)m, i =>
        {
            var row = result.AsSpan((int)(i * n), (int)n);
            for (long p = 0; p < k; p++)
            {
                float a = left[i * k + p];
                if (a == 0f)
                {
                    continue;
                }

                var source = right.AsSpan((int)(p * n), (int)n);
                for (int j = 0; j < row.Length; j++)
                {
                    row[j] += a * source[j];
                }
            }
        });
        return result;
    }

    /// <summary>
    /// LyCORIS' Tucker reconstruction <c>einsum('i j k l, j r, i p -> p r k l', core, b, a)</c>: core [R, R, kh, kw], <paramref name="b"/> [R, n],
    /// <paramref name="a"/> [R, m] → [m, n, kh, kw].
    /// </summary>
    protected static float[] Tucker(float[] core, float[] b, float[] a, long rank, long m, long n, long kernel)
    {
        // First t[i, r, s] = Σ_j core[i, j, s] · b[j, r], then out[p, r, s] = Σ_i a[i, p] · t[i, r, s].
        var t = new float[rank * n * kernel];
        for (long i = 0; i < rank; i++)
        {
            for (long j = 0; j < rank; j++)
            {
                for (long s = 0; s < kernel; s++)
                {
                    float c = core[(i * rank + j) * kernel + s];
                    for (long r = 0; r < n; r++)
                    {
                        t[(i * n + r) * kernel + s] += c * b[j * n + r];
                    }
                }
            }
        }

        var result = new float[m * n * kernel];
        for (long i = 0; i < rank; i++)
        {
            for (long p = 0; p < m; p++)
            {
                float factor = a[i * m + p];
                for (long q = 0; q < n * kernel; q++)
                {
                    result[p * n * kernel + q] += factor * t[i * n * kernel + q];
                }
            }
        }

        return result;
    }

    /// <summary>The elements of a weight after its first axis (1 for vectors).</summary>
    protected static long Inner(WeightInfo weight) => weight.Shape.Elements() / weight.Shape[0].Fixed;

    /// <summary>The spatial size of a convolution kernel (1 for matrices).</summary>
    protected static long Kernel(WeightInfo weight) => weight.Shape.Rank == 4 ? weight.Shape[2].Fixed * weight.Shape[3].Fixed : 1;
}
