using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The skinny tensor-core GEMM (<c>matmul_skinny.slang</c>) on matrices wide enough for it to serve 2-16 rows (N of at least
/// 4096): single matmuls and the fused gate / up pair, for the staged path and, on devices with the MMA fragment layout, the
/// direct path (fragments filled from registers, with the K order permuted per lane). Block-scaled FP8 rows of K % 512 == 0
/// are repacked into the aligned split layout.
/// </summary>
public sealed class SkinnyGemmTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in new[] { "f16", "bf16", "f8e4m3_b128", "f8e4m3_b32", "nvfp4", "q8_0" })
        {
            foreach (var t in new[] { 2, 5, 16 })
            {
                yield return [fmt, t];
            }
        }
    }

    // f16 activations on tensor cores (~1 % error); q8_0 runs its rows up to 4 or 8 on the integer-dot matvec (8-bit activations).
    private static double Atol(string fmt) => fmt == "q8_0" ? 3.0 : 1.5e-1;

    [Theory]
    [MemberData(nameof(Cases))]
    public void Single(string fmt, int t)
    {
        Gpu.Require();
        int K = 1024, N = 4112;                                     // N % 32 != 0: a partial last workgroup
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 1f, N, K)));
        Diff.Check("Skinny." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(fmt), rtol: 1e-2, what: $"T={t}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Dual(string fmt, int t)
    {
        Gpu.Require();
        int K = 1536, N = 4096;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.1f, N, K));
        var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.1f, N, K));
        tg.B.Output("y", tg.B.Mul(tg.B.Silu(g), u));
        // silu(g) * u multiplies the errors of both products. Those are proportional to the weight rows' norms, not to the
        // outputs: NVFP4's random rows give products above 100, so its bound is wider (the generic f16 GEMM deviates alike).
        double atol = fmt switch { "q8_0" => 6e-1, "nvfp4" => 3.0, _ => 1e-1 };
        Diff.Check("Skinny.dual." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: atol, rtol: 2e-2, what: $"T={t}");
    }
}
