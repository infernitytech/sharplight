using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sharplight.Cpu;

// The Arm64 NEON micro-kernel: a 6 x 16 tile (the AVX2 tile shape, so packing and blocking are shared) in 24 Vector128
// accumulators, multiplied by lane from one load of four and one load of two A values per k (fmla v.4s, v.4s, v.s[i]).
internal static unsafe partial class Gemm
{
    /// <summary>
    /// One full 6 x 16 tile. B row k is at <paramref name="b"/> + k * <paramref name="ldb"/>, or at <paramref name="b"/> +
    /// <paramref name="bOffsets"/>[k] when <paramref name="bOffsets"/> is not null (the gathered form of <see cref="TileGather"/>).
    /// </summary>
    private static void K6x16Neon(float* a, float* b, long ldb, long* bOffsets, float* c, long ldc, int kc, float* bias, bool acc)
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
        else if (bias != null)
        {
            c00 = Vector128.Create(bias[0]);
            c01 = c00;
            c02 = c00;
            c03 = c00;
            c10 = Vector128.Create(bias[1]);
            c11 = c10;
            c12 = c10;
            c13 = c10;
            c20 = Vector128.Create(bias[2]);
            c21 = c20;
            c22 = c20;
            c23 = c20;
            c30 = Vector128.Create(bias[3]);
            c31 = c30;
            c32 = c30;
            c33 = c30;
            c40 = Vector128.Create(bias[4]);
            c41 = c40;
            c42 = c40;
            c43 = c40;
            c50 = Vector128.Create(bias[5]);
            c51 = c50;
            c52 = c50;
            c53 = c50;
        }
        else
        {
            c00 = Vector128<float>.Zero;
            c01 = c00;
            c02 = c00;
            c03 = c00;
            c10 = c00;
            c11 = c00;
            c12 = c00;
            c13 = c00;
            c20 = c00;
            c21 = c00;
            c22 = c00;
            c23 = c00;
            c30 = c00;
            c31 = c00;
            c32 = c00;
            c33 = c00;
            c40 = c00;
            c41 = c00;
            c42 = c00;
            c43 = c00;
            c50 = c00;
            c51 = c00;
            c52 = c00;
            c53 = c00;
        }

        for (int k = 0; k < kc; k++)
        {
            float* bk = bOffsets == null ? b + k * ldb : b + bOffsets[k];
            var b0 = Vector128.Load(bk + 0);
            var b1 = Vector128.Load(bk + 4);
            var b2 = Vector128.Load(bk + 8);
            var b3 = Vector128.Load(bk + 12);
            var a03 = Vector128.Load(a);
            var a45 = Vector64.Load(a + 4);
            c00 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c00, b0, a03, 0);
            c01 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c01, b1, a03, 0);
            c02 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c02, b2, a03, 0);
            c03 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c03, b3, a03, 0);
            c10 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c10, b0, a03, 1);
            c11 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c11, b1, a03, 1);
            c12 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c12, b2, a03, 1);
            c13 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c13, b3, a03, 1);
            c20 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c20, b0, a03, 2);
            c21 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c21, b1, a03, 2);
            c22 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c22, b2, a03, 2);
            c23 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c23, b3, a03, 2);
            c30 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c30, b0, a03, 3);
            c31 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c31, b1, a03, 3);
            c32 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c32, b2, a03, 3);
            c33 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c33, b3, a03, 3);
            c40 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c40, b0, a45, 0);
            c41 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c41, b1, a45, 0);
            c42 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c42, b2, a45, 0);
            c43 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c43, b3, a45, 0);
            c50 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c50, b0, a45, 1);
            c51 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c51, b1, a45, 1);
            c52 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c52, b2, a45, 1);
            c53 = AdvSimd.Arm64.FusedMultiplyAddBySelectedScalar(c53, b3, a45, 1);
            a += 6;
        }

        c00.Store(c + 0 * ldc + 0);
        c01.Store(c + 0 * ldc + 4);
        c02.Store(c + 0 * ldc + 8);
        c03.Store(c + 0 * ldc + 12);
        c10.Store(c + 1 * ldc + 0);
        c11.Store(c + 1 * ldc + 4);
        c12.Store(c + 1 * ldc + 8);
        c13.Store(c + 1 * ldc + 12);
        c20.Store(c + 2 * ldc + 0);
        c21.Store(c + 2 * ldc + 4);
        c22.Store(c + 2 * ldc + 8);
        c23.Store(c + 2 * ldc + 12);
        c30.Store(c + 3 * ldc + 0);
        c31.Store(c + 3 * ldc + 4);
        c32.Store(c + 3 * ldc + 8);
        c33.Store(c + 3 * ldc + 12);
        c40.Store(c + 4 * ldc + 0);
        c41.Store(c + 4 * ldc + 4);
        c42.Store(c + 4 * ldc + 8);
        c43.Store(c + 4 * ldc + 12);
        c50.Store(c + 5 * ldc + 0);
        c51.Store(c + 5 * ldc + 4);
        c52.Store(c + 5 * ldc + 8);
        c53.Store(c + 5 * ldc + 12);
    }
}
