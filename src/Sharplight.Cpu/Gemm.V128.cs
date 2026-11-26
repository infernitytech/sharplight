using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Sharplight.Cpu;

// The 128-bit micro-kernel of the v128 tier (x86 without AVX2): a 6 x 8 tile in 12 Vector128 accumulators, which with the
// two B vectors and one broadcast A value fits the 16 xmm registers of SSE. Fused multiply-adds where the host has FMA3.
internal static unsafe partial class Gemm
{
    /// <summary>
    /// One full 6 x 8 tile. B row k is at <paramref name="b"/> + k * <paramref name="ldb"/>, or at <paramref name="b"/> +
    /// <paramref name="bOffsets"/>[k] when <paramref name="bOffsets"/> is not null (the gathered form of <see cref="TileGather"/>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void K6x8V128(float* a, float* b, long ldb, long* bOffsets, float* c, long ldc, int kc, float* bias, bool acc)
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
        else if (bias != null)
        {
            c00 = Vector128.Create(bias[0]);
            c01 = c00;
            c10 = Vector128.Create(bias[1]);
            c11 = c10;
            c20 = Vector128.Create(bias[2]);
            c21 = c20;
            c30 = Vector128.Create(bias[3]);
            c31 = c30;
            c40 = Vector128.Create(bias[4]);
            c41 = c40;
            c50 = Vector128.Create(bias[5]);
            c51 = c50;
        }
        else
        {
            c00 = Vector128<float>.Zero;
            c01 = c00;
            c10 = c00;
            c11 = c00;
            c20 = c00;
            c21 = c00;
            c30 = c00;
            c31 = c00;
            c40 = c00;
            c41 = c00;
            c50 = c00;
            c51 = c00;
        }

        for (int k = 0; k < kc; k++)
        {
            float* bk = bOffsets == null ? b + k * ldb : b + bOffsets[k];
            var b0 = Vector128.Load(bk);
            var b1 = Vector128.Load(bk + 4);
            var a0 = Vector128.Create(a[0]);
            c00 = Vec128.MultiplyAdd(a0, b0, c00);
            c01 = Vec128.MultiplyAdd(a0, b1, c01);
            var a1 = Vector128.Create(a[1]);
            c10 = Vec128.MultiplyAdd(a1, b0, c10);
            c11 = Vec128.MultiplyAdd(a1, b1, c11);
            var a2 = Vector128.Create(a[2]);
            c20 = Vec128.MultiplyAdd(a2, b0, c20);
            c21 = Vec128.MultiplyAdd(a2, b1, c21);
            var a3 = Vector128.Create(a[3]);
            c30 = Vec128.MultiplyAdd(a3, b0, c30);
            c31 = Vec128.MultiplyAdd(a3, b1, c31);
            var a4 = Vector128.Create(a[4]);
            c40 = Vec128.MultiplyAdd(a4, b0, c40);
            c41 = Vec128.MultiplyAdd(a4, b1, c41);
            var a5 = Vector128.Create(a[5]);
            c50 = Vec128.MultiplyAdd(a5, b0, c50);
            c51 = Vec128.MultiplyAdd(a5, b1, c51);
            a += 6;
        }

        c00.Store(c + 0 * ldc + 0);
        c01.Store(c + 0 * ldc + 4);
        c10.Store(c + 1 * ldc + 0);
        c11.Store(c + 1 * ldc + 4);
        c20.Store(c + 2 * ldc + 0);
        c21.Store(c + 2 * ldc + 4);
        c30.Store(c + 3 * ldc + 0);
        c31.Store(c + 3 * ldc + 4);
        c40.Store(c + 4 * ldc + 0);
        c41.Store(c + 4 * ldc + 4);
        c50.Store(c + 5 * ldc + 0);
        c51.Store(c + 5 * ldc + 4);
    }
}
