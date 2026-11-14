using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Inverses of square f32 matrices [..., M, M] by Gauss-Jordan elimination with partial pivoting (in double precision).</summary>
internal sealed unsafe class InverseOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    public override string Name => "Inverse";

    public InverseOp(TRef x, TRef y)
    {
        _x = x;
        _y = y;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        int m = (int)d[^1];
        long batch = OpHelpers.Prod(d, 0, d.Length - 2);
        var a = new double[m * m];
        var inverse = new double[m * m];
        float* x = (float*)_x.P;
        float* y = (float*)_y.P;
        for (long b = 0; b < batch; b++)
        {
            float* matrix = x + b * m * m;
            for (int i = 0; i < m * m; i++)
            {
                a[i] = matrix[i];
                inverse[i] = i / m == i % m ? 1 : 0;
            }

            Invert(a, inverse, m);
            float* result = y + b * m * m;
            for (int i = 0; i < m * m; i++)
            {
                result[i] = (float)inverse[i];
            }
        }
    }

    /// <summary>Reduces <paramref name="a"/> to the identity, applying the same row operations to <paramref name="inverse"/>.</summary>
    private static void Invert(double[] a, double[] inverse, int m)
    {
        for (int col = 0; col < m; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < m; r++)
            {
                if (Math.Abs(a[r * m + col]) > Math.Abs(a[pivot * m + col]))
                {
                    pivot = r;
                }
            }

            if (pivot != col)
            {
                for (int k = 0; k < m; k++)
                {
                    (a[col * m + k], a[pivot * m + k]) = (a[pivot * m + k], a[col * m + k]);
                    (inverse[col * m + k], inverse[pivot * m + k]) = (inverse[pivot * m + k], inverse[col * m + k]);
                }
            }

            double diagonal = a[col * m + col];
            for (int k = 0; k < m; k++)
            {
                a[col * m + k] /= diagonal;
                inverse[col * m + k] /= diagonal;
            }

            for (int r = 0; r < m; r++)
            {
                double factor = a[r * m + col];
                if (r == col || factor == 0)
                {
                    continue;
                }

                for (int k = 0; k < m; k++)
                {
                    a[r * m + k] -= factor * a[col * m + k];
                    inverse[r * m + k] -= factor * inverse[col * m + k];
                }
            }
        }
    }
}
