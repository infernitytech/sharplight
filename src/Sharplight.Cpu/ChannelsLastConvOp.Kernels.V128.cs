using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sharplight.Cpu;

// The 128-bit micro-kernels of the channels-last conv: 6 pixels x 8 channels in 12 Vector128 accumulators for the v128 tier
// (fits the 16 xmm registers of SSE), and 6 pixels x 16 channels in 24 accumulators for Arm64 NEON (32 registers), which
// multiplies by a scalar lane (fmla v.4s, v.4s, v.s[0]) instead of broadcasting.
internal sealed unsafe partial class ChannelsLastConvOp
{
    /// <summary>
    /// 6 pixels x 8 channels (v128 tier) over segments [s0, s1) and their values [k0, k1): c[i][j] = clamp((acc ? c[i][j] : bias[j]) +
    /// sum of ptrs[s][i][k] * b[s * len + k][j] + residual[i][j], lo, hi).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel6x8V128(float** ptrs, int s0, int s1, int k0, int k1, int len, float* b, float* bias, bool acc, float* c, long ldc, float* residual, float lo, float hi)
    {
        Vector128<float> c00;
        Vector128<float> c01;
        Vector128<float> c10;
        Vector128<float> c11;
        Vector128<float> c20;
        Vector128<float> c21;
        Vector128<float> c30;
        Vector128<float> c31;
        Vector128<float> c40;
        Vector128<float> c41;
        Vector128<float> c50;
        Vector128<float> c51;
        if (acc)
        {
            c00 = Vector128.Load(c + 0 * ldc + 0);
            c01 = Vector128.Load(c + 0 * ldc + 4);
            c10 = Vector128.Load(c + 1 * ldc + 0);
            c11 = Vector128.Load(c + 1 * ldc + 4);
            c20 = Vector128.Load(c + 2 * ldc + 0);
            c21 = Vector128.Load(c + 2 * ldc + 4);
            c30 = Vector128.Load(c + 3 * ldc + 0);
            c31 = Vector128.Load(c + 3 * ldc + 4);
            c40 = Vector128.Load(c + 4 * ldc + 0);
            c41 = Vector128.Load(c + 4 * ldc + 4);
            c50 = Vector128.Load(c + 5 * ldc + 0);
            c51 = Vector128.Load(c + 5 * ldc + 4);
        }
        else
        {
            var init0 = bias == null ? Vector128<float>.Zero : Vector128.Load(bias + 0);
            var init1 = bias == null ? Vector128<float>.Zero : Vector128.Load(bias + 4);
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
            float* bs = b + ((long)s * len + k0) * 8;
            for (int k = k0; k < k1; k++)
            {
                var b0 = Vector128.Load(bs + 0);
                var b1 = Vector128.Load(bs + 4);
                var x0 = Vector128.Create(a0[k]);
                c00 = Vec128.MultiplyAdd(x0, b0, c00);
                c01 = Vec128.MultiplyAdd(x0, b1, c01);
                var x1 = Vector128.Create(a1[k]);
                c10 = Vec128.MultiplyAdd(x1, b0, c10);
                c11 = Vec128.MultiplyAdd(x1, b1, c11);
                var x2 = Vector128.Create(a2[k]);
                c20 = Vec128.MultiplyAdd(x2, b0, c20);
                c21 = Vec128.MultiplyAdd(x2, b1, c21);
                var x3 = Vector128.Create(a3[k]);
                c30 = Vec128.MultiplyAdd(x3, b0, c30);
                c31 = Vec128.MultiplyAdd(x3, b1, c31);
                var x4 = Vector128.Create(a4[k]);
                c40 = Vec128.MultiplyAdd(x4, b0, c40);
                c41 = Vec128.MultiplyAdd(x4, b1, c41);
                var x5 = Vector128.Create(a5[k]);
                c50 = Vec128.MultiplyAdd(x5, b0, c50);
                c51 = Vec128.MultiplyAdd(x5, b1, c51);
                bs += 8;
            }
        }

        if (residual != null)
        {
            c00 += Vector128.Load(residual + 0 * ldc + 0);
            c01 += Vector128.Load(residual + 0 * ldc + 4);
            c10 += Vector128.Load(residual + 1 * ldc + 0);
            c11 += Vector128.Load(residual + 1 * ldc + 4);
            c20 += Vector128.Load(residual + 2 * ldc + 0);
            c21 += Vector128.Load(residual + 2 * ldc + 4);
            c30 += Vector128.Load(residual + 3 * ldc + 0);
            c31 += Vector128.Load(residual + 3 * ldc + 4);
            c40 += Vector128.Load(residual + 4 * ldc + 0);
            c41 += Vector128.Load(residual + 4 * ldc + 4);
            c50 += Vector128.Load(residual + 5 * ldc + 0);
            c51 += Vector128.Load(residual + 5 * ldc + 4);
        }

        var vlo = Vector128.Create(lo);
        var vhi = Vector128.Create(hi);
        Vector128.Min(Vector128.Max(c00, vlo), vhi).Store(c + 0 * ldc + 0);
        Vector128.Min(Vector128.Max(c01, vlo), vhi).Store(c + 0 * ldc + 4);
        Vector128.Min(Vector128.Max(c10, vlo), vhi).Store(c + 1 * ldc + 0);
        Vector128.Min(Vector128.Max(c11, vlo), vhi).Store(c + 1 * ldc + 4);
        Vector128.Min(Vector128.Max(c20, vlo), vhi).Store(c + 2 * ldc + 0);
        Vector128.Min(Vector128.Max(c21, vlo), vhi).Store(c + 2 * ldc + 4);
        Vector128.Min(Vector128.Max(c30, vlo), vhi).Store(c + 3 * ldc + 0);
        Vector128.Min(Vector128.Max(c31, vlo), vhi).Store(c + 3 * ldc + 4);
        Vector128.Min(Vector128.Max(c40, vlo), vhi).Store(c + 4 * ldc + 0);
        Vector128.Min(Vector128.Max(c41, vlo), vhi).Store(c + 4 * ldc + 4);
        Vector128.Min(Vector128.Max(c50, vlo), vhi).Store(c + 5 * ldc + 0);
        Vector128.Min(Vector128.Max(c51, vlo), vhi).Store(c + 5 * ldc + 4);
    }

    /// <summary>
    /// 6 pixels x 16 channels (Arm64 NEON) over segments [s0, s1) and their values [k0, k1): c[i][j] = clamp((acc ? c[i][j] : bias[j]) +
    /// sum of ptrs[s][i][k] * b[s * len + k][j] + residual[i][j], lo, hi).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Kernel6x16Neon(float** ptrs, int s0, int s1, int k0, int k1, int len, float* b, float* bias, bool acc, float* c, long ldc, float* residual, float lo, float hi)
    {
        Vector128<float> c00;
        Vector128<float> c01;
        Vector128<float> c02;
        Vector128<float> c03;
        Vector128<float> c10;
        Vector128<float> c11;
        Vector128<float> c12;
        Vector128<float> c13;
        Vector128<float> c20;
        Vector128<float> c21;
        Vector128<float> c22;
        Vector128<float> c23;
        Vector128<float> c30;
        Vector128<float> c31;
        Vector128<float> c32;
        Vector128<float> c33;
        Vector128<float> c40;
        Vector128<float> c41;
        Vector128<float> c42;
        Vector128<float> c43;
        Vector128<float> c50;
        Vector128<float> c51;
        Vector128<float> c52;
        Vector128<float> c53;
        if (acc)
        {
            c00 = Vector128.Load(c + 0 * ldc + 0);
            c01 = Vector128.Load(c + 0 * ldc + 4);
            c02 = Vector128.Load(c + 0 * ldc + 8);
            c03 = Vector128.Load(c + 0 * ldc + 12);
            c10 = Vector128.Load(c + 1 * ldc + 0);
            c11 = Vector128.Load(c + 1 * ldc + 4);
            c12 = Vector128.Load(c + 1 * ldc + 8);
            c13 = Vector128.Load(c + 1 * ldc + 12);
            c20 = Vector128.Load(c + 2 * ldc + 0);
            c21 = Vector128.Load(c + 2 * ldc + 4);
            c22 = Vector128.Load(c + 2 * ldc + 8);
            c23 = Vector128.Load(c + 2 * ldc + 12);
            c30 = Vector128.Load(c + 3 * ldc + 0);
            c31 = Vector128.Load(c + 3 * ldc + 4);
            c32 = Vector128.Load(c + 3 * ldc + 8);
            c33 = Vector128.Load(c + 3 * ldc + 12);
            c40 = Vector128.Load(c + 4 * ldc + 0);
            c41 = Vector128.Load(c + 4 * ldc + 4);
            c42 = Vector128.Load(c + 4 * ldc + 8);
            c43 = Vector128.Load(c + 4 * ldc + 12);
            c50 = Vector128.Load(c + 5 * ldc + 0);
            c51 = Vector128.Load(c + 5 * ldc + 4);
            c52 = Vector128.Load(c + 5 * ldc + 8);
            c53 = Vector128.Load(c + 5 * ldc + 12);
        }
        else
        {
            var init0 = bias == null ? Vector128<float>.Zero : Vector128.Load(bias + 0);
            var init1 = bias == null ? Vector128<float>.Zero : Vector128.Load(bias + 4);
            var init2 = bias == null ? Vector128<float>.Zero : Vector128.Load(bias + 8);
            var init3 = bias == null ? Vector128<float>.Zero : Vector128.Load(bias + 12);
            c00 = init0;
            c01 = init1;
            c02 = init2;
            c03 = init3;
            c10 = init0;
            c11 = init1;
            c12 = init2;
            c13 = init3;
            c20 = init0;
            c21 = init1;
            c22 = init2;
            c23 = init3;
            c30 = init0;
            c31 = init1;
            c32 = init2;
            c33 = init3;
            c40 = init0;
            c41 = init1;
            c42 = init2;
            c43 = init3;
            c50 = init0;
            c51 = init1;
            c52 = init2;
            c53 = init3;
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
                var b0 = Vector128.Load(bs + 0);
                var b1 = Vector128.Load(bs + 4);
                var b2 = Vector128.Load(bs + 8);
                var b3 = Vector128.Load(bs + 12);
                var x0 = Vector64.CreateScalarUnsafe(a0[k]);
                c00 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c00, b0, x0);
                c01 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c01, b1, x0);
                c02 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c02, b2, x0);
                c03 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c03, b3, x0);
                var x1 = Vector64.CreateScalarUnsafe(a1[k]);
                c10 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c10, b0, x1);
                c11 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c11, b1, x1);
                c12 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c12, b2, x1);
                c13 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c13, b3, x1);
                var x2 = Vector64.CreateScalarUnsafe(a2[k]);
                c20 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c20, b0, x2);
                c21 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c21, b1, x2);
                c22 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c22, b2, x2);
                c23 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c23, b3, x2);
                var x3 = Vector64.CreateScalarUnsafe(a3[k]);
                c30 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c30, b0, x3);
                c31 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c31, b1, x3);
                c32 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c32, b2, x3);
                c33 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c33, b3, x3);
                var x4 = Vector64.CreateScalarUnsafe(a4[k]);
                c40 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c40, b0, x4);
                c41 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c41, b1, x4);
                c42 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c42, b2, x4);
                c43 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c43, b3, x4);
                var x5 = Vector64.CreateScalarUnsafe(a5[k]);
                c50 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c50, b0, x5);
                c51 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c51, b1, x5);
                c52 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c52, b2, x5);
                c53 = AdvSimd.Arm64.FusedMultiplyAddByScalar(c53, b3, x5);
                bs += 16;
            }
        }

        if (residual != null)
        {
            c00 += Vector128.Load(residual + 0 * ldc + 0);
            c01 += Vector128.Load(residual + 0 * ldc + 4);
            c02 += Vector128.Load(residual + 0 * ldc + 8);
            c03 += Vector128.Load(residual + 0 * ldc + 12);
            c10 += Vector128.Load(residual + 1 * ldc + 0);
            c11 += Vector128.Load(residual + 1 * ldc + 4);
            c12 += Vector128.Load(residual + 1 * ldc + 8);
            c13 += Vector128.Load(residual + 1 * ldc + 12);
            c20 += Vector128.Load(residual + 2 * ldc + 0);
            c21 += Vector128.Load(residual + 2 * ldc + 4);
            c22 += Vector128.Load(residual + 2 * ldc + 8);
            c23 += Vector128.Load(residual + 2 * ldc + 12);
            c30 += Vector128.Load(residual + 3 * ldc + 0);
            c31 += Vector128.Load(residual + 3 * ldc + 4);
            c32 += Vector128.Load(residual + 3 * ldc + 8);
            c33 += Vector128.Load(residual + 3 * ldc + 12);
            c40 += Vector128.Load(residual + 4 * ldc + 0);
            c41 += Vector128.Load(residual + 4 * ldc + 4);
            c42 += Vector128.Load(residual + 4 * ldc + 8);
            c43 += Vector128.Load(residual + 4 * ldc + 12);
            c50 += Vector128.Load(residual + 5 * ldc + 0);
            c51 += Vector128.Load(residual + 5 * ldc + 4);
            c52 += Vector128.Load(residual + 5 * ldc + 8);
            c53 += Vector128.Load(residual + 5 * ldc + 12);
        }

        var vlo = Vector128.Create(lo);
        var vhi = Vector128.Create(hi);
        Vector128.Min(Vector128.Max(c00, vlo), vhi).Store(c + 0 * ldc + 0);
        Vector128.Min(Vector128.Max(c01, vlo), vhi).Store(c + 0 * ldc + 4);
        Vector128.Min(Vector128.Max(c02, vlo), vhi).Store(c + 0 * ldc + 8);
        Vector128.Min(Vector128.Max(c03, vlo), vhi).Store(c + 0 * ldc + 12);
        Vector128.Min(Vector128.Max(c10, vlo), vhi).Store(c + 1 * ldc + 0);
        Vector128.Min(Vector128.Max(c11, vlo), vhi).Store(c + 1 * ldc + 4);
        Vector128.Min(Vector128.Max(c12, vlo), vhi).Store(c + 1 * ldc + 8);
        Vector128.Min(Vector128.Max(c13, vlo), vhi).Store(c + 1 * ldc + 12);
        Vector128.Min(Vector128.Max(c20, vlo), vhi).Store(c + 2 * ldc + 0);
        Vector128.Min(Vector128.Max(c21, vlo), vhi).Store(c + 2 * ldc + 4);
        Vector128.Min(Vector128.Max(c22, vlo), vhi).Store(c + 2 * ldc + 8);
        Vector128.Min(Vector128.Max(c23, vlo), vhi).Store(c + 2 * ldc + 12);
        Vector128.Min(Vector128.Max(c30, vlo), vhi).Store(c + 3 * ldc + 0);
        Vector128.Min(Vector128.Max(c31, vlo), vhi).Store(c + 3 * ldc + 4);
        Vector128.Min(Vector128.Max(c32, vlo), vhi).Store(c + 3 * ldc + 8);
        Vector128.Min(Vector128.Max(c33, vlo), vhi).Store(c + 3 * ldc + 12);
        Vector128.Min(Vector128.Max(c40, vlo), vhi).Store(c + 4 * ldc + 0);
        Vector128.Min(Vector128.Max(c41, vlo), vhi).Store(c + 4 * ldc + 4);
        Vector128.Min(Vector128.Max(c42, vlo), vhi).Store(c + 4 * ldc + 8);
        Vector128.Min(Vector128.Max(c43, vlo), vhi).Store(c + 4 * ldc + 12);
        Vector128.Min(Vector128.Max(c50, vlo), vhi).Store(c + 5 * ldc + 0);
        Vector128.Min(Vector128.Max(c51, vlo), vhi).Store(c + 5 * ldc + 4);
        Vector128.Min(Vector128.Max(c52, vlo), vhi).Store(c + 5 * ldc + 8);
        Vector128.Min(Vector128.Max(c53, vlo), vhi).Store(c + 5 * ldc + 12);
    }
}
