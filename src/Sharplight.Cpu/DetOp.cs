using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Determinants of square f32 matrices [..., M, M] by LU decomposition with partial pivoting (in double precision).</summary>
internal sealed unsafe class DetOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    public override string Name => "Det";

    public DetOp(TRef x, TRef y)
    {
        _x = x;
        _y = y;
    }

    public override void Run(RunParameters p)
    {
        var d = OpHelpers.Dims(_x, p);
        int m = (int)d[^1];
        long batch = OpHelpers.Prod(d, 0, d.Length - 2);
        var lu = new double[m * m];
        float* x = (float*)_x.P;
        float* y = (float*)_y.P;
        for (long b = 0; b < batch; b++)
        {
            float* matrix = x + b * m * m;
            for (int i = 0; i < m * m; i++)
            {
                lu[i] = matrix[i];
            }

            y[b] = (float)Determinant(lu, m);
        }
    }

    private static double Determinant(double[] a, int m)
    {
        double det = 1;
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

            if (a[pivot * m + col] == 0)
            {
                return 0;
            }

            if (pivot != col)
            {
                for (int k = 0; k < m; k++)
                {
                    (a[col * m + k], a[pivot * m + k]) = (a[pivot * m + k], a[col * m + k]);
                }

                det = -det;
            }

            double diagonal = a[col * m + col];
            det *= diagonal;
            for (int r = col + 1; r < m; r++)
            {
                double factor = a[r * m + col] / diagonal;
                for (int k = col + 1; k < m; k++)
                {
                    a[r * m + k] -= factor * a[col * m + k];
                }
            }
        }

        return det;
    }
}
