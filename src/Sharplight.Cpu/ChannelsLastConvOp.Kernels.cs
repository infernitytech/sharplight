using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

// Register-blocked micro-kernels of the channels-last conv: each input value is broadcast from its pixel's row pointer.
internal sealed unsafe partial class ChannelsLastConvOp
{
    /// <summary>
    /// 8 pixels x 32 channels over segments [s0, s1) and their values [k0, k1): c[i][j] = clamp((acc ? c[i][j] : bias[j]) +
    /// sum of ptrs[s][i][k] * b[s * len + k][j] + residual[i][j], lo, hi).
    /// </summary>
    private static void Kernel8x32(float** ptrs, int s0, int s1, int k0, int k1, int len, float* b, float* bias, bool acc, float* c, long ldc, float* residual, float lo, float hi)
    {
        Vector512<float> c00;
        Vector512<float> c01;
        Vector512<float> c10;
        Vector512<float> c11;
        Vector512<float> c20;
        Vector512<float> c21;
        Vector512<float> c30;
        Vector512<float> c31;
        Vector512<float> c40;
        Vector512<float> c41;
        Vector512<float> c50;
        Vector512<float> c51;
        Vector512<float> c60;
        Vector512<float> c61;
        Vector512<float> c70;
        Vector512<float> c71;
        if (acc)
        {
            c00 = Vector512.Load(c + 0 * ldc);
            c01 = Vector512.Load(c + 0 * ldc + 16);
            c10 = Vector512.Load(c + 1 * ldc);
            c11 = Vector512.Load(c + 1 * ldc + 16);
            c20 = Vector512.Load(c + 2 * ldc);
            c21 = Vector512.Load(c + 2 * ldc + 16);
            c30 = Vector512.Load(c + 3 * ldc);
            c31 = Vector512.Load(c + 3 * ldc + 16);
            c40 = Vector512.Load(c + 4 * ldc);
            c41 = Vector512.Load(c + 4 * ldc + 16);
            c50 = Vector512.Load(c + 5 * ldc);
            c51 = Vector512.Load(c + 5 * ldc + 16);
            c60 = Vector512.Load(c + 6 * ldc);
            c61 = Vector512.Load(c + 6 * ldc + 16);
            c70 = Vector512.Load(c + 7 * ldc);
            c71 = Vector512.Load(c + 7 * ldc + 16);
        }
        else
        {
            var init0 = bias == null ? Vector512<float>.Zero : Vector512.Load(bias);
            var init1 = bias == null ? Vector512<float>.Zero : Vector512.Load(bias + 16);
            c00 = init0;
            c01 = init1;
            c10 = init0;
            c11 = init1;
            c20 = init0;
            c21 = init1;
            c30 = init0;
            c31 = init1;
            c40 = init0;
            c41 = init1;
            c50 = init0;
            c51 = init1;
            c60 = init0;
            c61 = init1;
            c70 = init0;
            c71 = init1;
        }

        for (int s = s0; s < s1; s++)
        {
            float** ps = ptrs + s * 8;
            float* a0 = ps[0];
            float* a1 = ps[1];
            float* a2 = ps[2];
            float* a3 = ps[3];
            float* a4 = ps[4];
            float* a5 = ps[5];
            float* a6 = ps[6];
            float* a7 = ps[7];
            float* bs = b + ((long)s * len + k0) * 32;
            for (int k = k0; k < k1; k++)
            {
                var b0 = Vector512.Load(bs);
                var b1 = Vector512.Load(bs + 16);
                var x0 = Vector512.Create(a0[k]);
                c00 = Vector512.FusedMultiplyAdd(x0, b0, c00);
                c01 = Vector512.FusedMultiplyAdd(x0, b1, c01);
                var x1 = Vector512.Create(a1[k]);
                c10 = Vector512.FusedMultiplyAdd(x1, b0, c10);
                c11 = Vector512.FusedMultiplyAdd(x1, b1, c11);
                var x2 = Vector512.Create(a2[k]);
                c20 = Vector512.FusedMultiplyAdd(x2, b0, c20);
                c21 = Vector512.FusedMultiplyAdd(x2, b1, c21);
                var x3 = Vector512.Create(a3[k]);
                c30 = Vector512.FusedMultiplyAdd(x3, b0, c30);
                c31 = Vector512.FusedMultiplyAdd(x3, b1, c31);
                var x4 = Vector512.Create(a4[k]);
                c40 = Vector512.FusedMultiplyAdd(x4, b0, c40);
                c41 = Vector512.FusedMultiplyAdd(x4, b1, c41);
                var x5 = Vector512.Create(a5[k]);
                c50 = Vector512.FusedMultiplyAdd(x5, b0, c50);
                c51 = Vector512.FusedMultiplyAdd(x5, b1, c51);
                var x6 = Vector512.Create(a6[k]);
                c60 = Vector512.FusedMultiplyAdd(x6, b0, c60);
                c61 = Vector512.FusedMultiplyAdd(x6, b1, c61);
                var x7 = Vector512.Create(a7[k]);
                c70 = Vector512.FusedMultiplyAdd(x7, b0, c70);
                c71 = Vector512.FusedMultiplyAdd(x7, b1, c71);
                bs += 32;
            }
        }

        if (residual != null)
        {
            c00 += Vector512.Load(residual + 0 * ldc);
            c01 += Vector512.Load(residual + 0 * ldc + 16);
            c10 += Vector512.Load(residual + 1 * ldc);
            c11 += Vector512.Load(residual + 1 * ldc + 16);
            c20 += Vector512.Load(residual + 2 * ldc);
            c21 += Vector512.Load(residual + 2 * ldc + 16);
            c30 += Vector512.Load(residual + 3 * ldc);
            c31 += Vector512.Load(residual + 3 * ldc + 16);
            c40 += Vector512.Load(residual + 4 * ldc);
            c41 += Vector512.Load(residual + 4 * ldc + 16);
            c50 += Vector512.Load(residual + 5 * ldc);
            c51 += Vector512.Load(residual + 5 * ldc + 16);
            c60 += Vector512.Load(residual + 6 * ldc);
            c61 += Vector512.Load(residual + 6 * ldc + 16);
            c70 += Vector512.Load(residual + 7 * ldc);
            c71 += Vector512.Load(residual + 7 * ldc + 16);
        }

        var vlo = Vector512.Create(lo);
        var vhi = Vector512.Create(hi);
        Vector512.Min(Vector512.Max(c00, vlo), vhi).Store(c + 0 * ldc);
        Vector512.Min(Vector512.Max(c01, vlo), vhi).Store(c + 0 * ldc + 16);
        Vector512.Min(Vector512.Max(c10, vlo), vhi).Store(c + 1 * ldc);
        Vector512.Min(Vector512.Max(c11, vlo), vhi).Store(c + 1 * ldc + 16);
        Vector512.Min(Vector512.Max(c20, vlo), vhi).Store(c + 2 * ldc);
        Vector512.Min(Vector512.Max(c21, vlo), vhi).Store(c + 2 * ldc + 16);
        Vector512.Min(Vector512.Max(c30, vlo), vhi).Store(c + 3 * ldc);
        Vector512.Min(Vector512.Max(c31, vlo), vhi).Store(c + 3 * ldc + 16);
        Vector512.Min(Vector512.Max(c40, vlo), vhi).Store(c + 4 * ldc);
        Vector512.Min(Vector512.Max(c41, vlo), vhi).Store(c + 4 * ldc + 16);
        Vector512.Min(Vector512.Max(c50, vlo), vhi).Store(c + 5 * ldc);
        Vector512.Min(Vector512.Max(c51, vlo), vhi).Store(c + 5 * ldc + 16);
        Vector512.Min(Vector512.Max(c60, vlo), vhi).Store(c + 6 * ldc);
        Vector512.Min(Vector512.Max(c61, vlo), vhi).Store(c + 6 * ldc + 16);
        Vector512.Min(Vector512.Max(c70, vlo), vhi).Store(c + 7 * ldc);
        Vector512.Min(Vector512.Max(c71, vlo), vhi).Store(c + 7 * ldc + 16);
    }

    /// <summary>
    /// 12 pixels x 16 channels over segments [s0, s1) and their values [k0, k1): c[i][j] = clamp((acc ? c[i][j] : bias[j]) +
    /// sum of ptrs[s][i][k] * b[s * len + k][j] + residual[i][j], lo, hi).
    /// </summary>
    private static void Kernel12x16(float** ptrs, int s0, int s1, int k0, int k1, int len, float* b, float* bias, bool acc, float* c, long ldc, float* residual, float lo, float hi)
    {
        Vector512<float> c00;
        Vector512<float> c10;
        Vector512<float> c20;
        Vector512<float> c30;
        Vector512<float> c40;
        Vector512<float> c50;
        Vector512<float> c60;
        Vector512<float> c70;
        Vector512<float> c80;
        Vector512<float> c90;
        Vector512<float> c100;
        Vector512<float> c110;
        if (acc)
        {
            c00 = Vector512.Load(c + 0 * ldc);
            c10 = Vector512.Load(c + 1 * ldc);
            c20 = Vector512.Load(c + 2 * ldc);
            c30 = Vector512.Load(c + 3 * ldc);
            c40 = Vector512.Load(c + 4 * ldc);
            c50 = Vector512.Load(c + 5 * ldc);
            c60 = Vector512.Load(c + 6 * ldc);
            c70 = Vector512.Load(c + 7 * ldc);
            c80 = Vector512.Load(c + 8 * ldc);
            c90 = Vector512.Load(c + 9 * ldc);
            c100 = Vector512.Load(c + 10 * ldc);
            c110 = Vector512.Load(c + 11 * ldc);
        }
        else
        {
            var init0 = bias == null ? Vector512<float>.Zero : Vector512.Load(bias);
            c00 = init0;
            c10 = init0;
            c20 = init0;
            c30 = init0;
            c40 = init0;
            c50 = init0;
            c60 = init0;
            c70 = init0;
            c80 = init0;
            c90 = init0;
            c100 = init0;
            c110 = init0;
        }

        for (int s = s0; s < s1; s++)
        {
            float** ps = ptrs + s * 12;
            float* a0 = ps[0];
            float* a1 = ps[1];
            float* a2 = ps[2];
            float* a3 = ps[3];
            float* a4 = ps[4];
            float* a5 = ps[5];
            float* a6 = ps[6];
            float* a7 = ps[7];
            float* a8 = ps[8];
            float* a9 = ps[9];
            float* a10 = ps[10];
            float* a11 = ps[11];
            float* bs = b + ((long)s * len + k0) * 16;
            for (int k = k0; k < k1; k++)
            {
                var b0 = Vector512.Load(bs);
                var x0 = Vector512.Create(a0[k]);
                c00 = Vector512.FusedMultiplyAdd(x0, b0, c00);
                var x1 = Vector512.Create(a1[k]);
                c10 = Vector512.FusedMultiplyAdd(x1, b0, c10);
                var x2 = Vector512.Create(a2[k]);
                c20 = Vector512.FusedMultiplyAdd(x2, b0, c20);
                var x3 = Vector512.Create(a3[k]);
                c30 = Vector512.FusedMultiplyAdd(x3, b0, c30);
                var x4 = Vector512.Create(a4[k]);
                c40 = Vector512.FusedMultiplyAdd(x4, b0, c40);
                var x5 = Vector512.Create(a5[k]);
                c50 = Vector512.FusedMultiplyAdd(x5, b0, c50);
                var x6 = Vector512.Create(a6[k]);
                c60 = Vector512.FusedMultiplyAdd(x6, b0, c60);
                var x7 = Vector512.Create(a7[k]);
                c70 = Vector512.FusedMultiplyAdd(x7, b0, c70);
                var x8 = Vector512.Create(a8[k]);
                c80 = Vector512.FusedMultiplyAdd(x8, b0, c80);
                var x9 = Vector512.Create(a9[k]);
                c90 = Vector512.FusedMultiplyAdd(x9, b0, c90);
                var x10 = Vector512.Create(a10[k]);
                c100 = Vector512.FusedMultiplyAdd(x10, b0, c100);
                var x11 = Vector512.Create(a11[k]);
                c110 = Vector512.FusedMultiplyAdd(x11, b0, c110);
                bs += 16;
            }
        }

        if (residual != null)
        {
            c00 += Vector512.Load(residual + 0 * ldc);
            c10 += Vector512.Load(residual + 1 * ldc);
            c20 += Vector512.Load(residual + 2 * ldc);
            c30 += Vector512.Load(residual + 3 * ldc);
            c40 += Vector512.Load(residual + 4 * ldc);
            c50 += Vector512.Load(residual + 5 * ldc);
            c60 += Vector512.Load(residual + 6 * ldc);
            c70 += Vector512.Load(residual + 7 * ldc);
            c80 += Vector512.Load(residual + 8 * ldc);
            c90 += Vector512.Load(residual + 9 * ldc);
            c100 += Vector512.Load(residual + 10 * ldc);
            c110 += Vector512.Load(residual + 11 * ldc);
        }

        var vlo = Vector512.Create(lo);
        var vhi = Vector512.Create(hi);
        Vector512.Min(Vector512.Max(c00, vlo), vhi).Store(c + 0 * ldc);
        Vector512.Min(Vector512.Max(c10, vlo), vhi).Store(c + 1 * ldc);
        Vector512.Min(Vector512.Max(c20, vlo), vhi).Store(c + 2 * ldc);
        Vector512.Min(Vector512.Max(c30, vlo), vhi).Store(c + 3 * ldc);
        Vector512.Min(Vector512.Max(c40, vlo), vhi).Store(c + 4 * ldc);
        Vector512.Min(Vector512.Max(c50, vlo), vhi).Store(c + 5 * ldc);
        Vector512.Min(Vector512.Max(c60, vlo), vhi).Store(c + 6 * ldc);
        Vector512.Min(Vector512.Max(c70, vlo), vhi).Store(c + 7 * ldc);
        Vector512.Min(Vector512.Max(c80, vlo), vhi).Store(c + 8 * ldc);
        Vector512.Min(Vector512.Max(c90, vlo), vhi).Store(c + 9 * ldc);
        Vector512.Min(Vector512.Max(c100, vlo), vhi).Store(c + 10 * ldc);
        Vector512.Min(Vector512.Max(c110, vlo), vhi).Store(c + 11 * ldc);
    }

    /// <summary>
    /// 6 pixels x 16 channels over segments [s0, s1) and their values [k0, k1): c[i][j] = clamp((acc ? c[i][j] : bias[j]) +
    /// sum of ptrs[s][i][k] * b[s * len + k][j] + residual[i][j], lo, hi).
    /// </summary>
    private static void Kernel6x16(float** ptrs, int s0, int s1, int k0, int k1, int len, float* b, float* bias, bool acc, float* c, long ldc, float* residual, float lo, float hi)
    {
        Vector256<float> c00;
        Vector256<float> c01;
        Vector256<float> c10;
        Vector256<float> c11;
        Vector256<float> c20;
        Vector256<float> c21;
        Vector256<float> c30;
        Vector256<float> c31;
        Vector256<float> c40;
        Vector256<float> c41;
        Vector256<float> c50;
        Vector256<float> c51;
        if (acc)
        {
            c00 = Vector256.Load(c + 0 * ldc);
            c01 = Vector256.Load(c + 0 * ldc + 8);
            c10 = Vector256.Load(c + 1 * ldc);
            c11 = Vector256.Load(c + 1 * ldc + 8);
            c20 = Vector256.Load(c + 2 * ldc);
            c21 = Vector256.Load(c + 2 * ldc + 8);
            c30 = Vector256.Load(c + 3 * ldc);
            c31 = Vector256.Load(c + 3 * ldc + 8);
            c40 = Vector256.Load(c + 4 * ldc);
            c41 = Vector256.Load(c + 4 * ldc + 8);
            c50 = Vector256.Load(c + 5 * ldc);
            c51 = Vector256.Load(c + 5 * ldc + 8);
        }
        else
        {
            var init0 = bias == null ? Vector256<float>.Zero : Vector256.Load(bias);
            var init1 = bias == null ? Vector256<float>.Zero : Vector256.Load(bias + 8);
            c00 = init0;
            c01 = init1;
            c10 = init0;
            c11 = init1;
            c20 = init0;
            c21 = init1;
            c30 = init0;
            c31 = init1;
            c40 = init0;
            c41 = init1;
            c50 = init0;
            c51 = init1;
        }

        for (int s = s0; s < s1; s++)
        {
            float** ps = ptrs + s * 6;
            float* a0 = ps[0];
            float* a1 = ps[1];
            float* a2 = ps[2];
            float* a3 = ps[3];
            float* a4 = ps[4];
            float* a5 = ps[5];
            float* bs = b + ((long)s * len + k0) * 16;
            for (int k = k0; k < k1; k++)
            {
                var b0 = Vector256.Load(bs);
                var b1 = Vector256.Load(bs + 8);
                var x0 = Vector256.Create(a0[k]);
                c00 = Vector256.FusedMultiplyAdd(x0, b0, c00);
                c01 = Vector256.FusedMultiplyAdd(x0, b1, c01);
                var x1 = Vector256.Create(a1[k]);
                c10 = Vector256.FusedMultiplyAdd(x1, b0, c10);
                c11 = Vector256.FusedMultiplyAdd(x1, b1, c11);
                var x2 = Vector256.Create(a2[k]);
                c20 = Vector256.FusedMultiplyAdd(x2, b0, c20);
                c21 = Vector256.FusedMultiplyAdd(x2, b1, c21);
                var x3 = Vector256.Create(a3[k]);
                c30 = Vector256.FusedMultiplyAdd(x3, b0, c30);
                c31 = Vector256.FusedMultiplyAdd(x3, b1, c31);
                var x4 = Vector256.Create(a4[k]);
                c40 = Vector256.FusedMultiplyAdd(x4, b0, c40);
                c41 = Vector256.FusedMultiplyAdd(x4, b1, c41);
                var x5 = Vector256.Create(a5[k]);
                c50 = Vector256.FusedMultiplyAdd(x5, b0, c50);
                c51 = Vector256.FusedMultiplyAdd(x5, b1, c51);
                bs += 16;
            }
        }

        if (residual != null)
        {
            c00 += Vector256.Load(residual + 0 * ldc);
            c01 += Vector256.Load(residual + 0 * ldc + 8);
            c10 += Vector256.Load(residual + 1 * ldc);
            c11 += Vector256.Load(residual + 1 * ldc + 8);
            c20 += Vector256.Load(residual + 2 * ldc);
            c21 += Vector256.Load(residual + 2 * ldc + 8);
            c30 += Vector256.Load(residual + 3 * ldc);
            c31 += Vector256.Load(residual + 3 * ldc + 8);
            c40 += Vector256.Load(residual + 4 * ldc);
            c41 += Vector256.Load(residual + 4 * ldc + 8);
            c50 += Vector256.Load(residual + 5 * ldc);
            c51 += Vector256.Load(residual + 5 * ldc + 8);
        }

        var vlo = Vector256.Create(lo);
        var vhi = Vector256.Create(hi);
        Vector256.Min(Vector256.Max(c00, vlo), vhi).Store(c + 0 * ldc);
        Vector256.Min(Vector256.Max(c01, vlo), vhi).Store(c + 0 * ldc + 8);
        Vector256.Min(Vector256.Max(c10, vlo), vhi).Store(c + 1 * ldc);
        Vector256.Min(Vector256.Max(c11, vlo), vhi).Store(c + 1 * ldc + 8);
        Vector256.Min(Vector256.Max(c20, vlo), vhi).Store(c + 2 * ldc);
        Vector256.Min(Vector256.Max(c21, vlo), vhi).Store(c + 2 * ldc + 8);
        Vector256.Min(Vector256.Max(c30, vlo), vhi).Store(c + 3 * ldc);
        Vector256.Min(Vector256.Max(c31, vlo), vhi).Store(c + 3 * ldc + 8);
        Vector256.Min(Vector256.Max(c40, vlo), vhi).Store(c + 4 * ldc);
        Vector256.Min(Vector256.Max(c41, vlo), vhi).Store(c + 4 * ldc + 8);
        Vector256.Min(Vector256.Max(c50, vlo), vhi).Store(c + 5 * ldc);
        Vector256.Min(Vector256.Max(c51, vlo), vhi).Store(c + 5 * ldc + 8);
    }
}
