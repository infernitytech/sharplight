using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Row norms feeding tensor-core matmuls also write the f16 activation copy those GEMMs read (<c>SHARPLIGHT_NORM_HALF_COPY</c>):
/// norm → matmul and norm → gate / up pair against the CPU reference at row counts on the matvec, skinny GEMM and tiled GEMM
/// paths, with the norm output also read by a second matmul of another width (one copy serves both).
/// </summary>
public sealed class NormHalfCopyTests
{
    [Theory]
    [InlineData("f8e4m3_b128", 1)]
    [InlineData("f8e4m3_b128", 5)]
    [InlineData("f8e4m3_b128", 16)]
    [InlineData("f8e4m3_b128", 40)]
    [InlineData("q8_0", 16)]
    [InlineData("q8_0", 100)]
    [InlineData("f16", 12)]
    [InlineData("bf16", 64)]
    public void NormThenMatMuls(string fmt, int t)
    {
        const int K = 1024;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var n = tg.B.RmsNorm(x, tg.RandW("nw", DataType.F32, 1f, K), 1e-6f);
        var type = DataType.Get(fmt);
        tg.B.Output("y", tg.B.MatMul(n, tg.RandW("w", type, 0.05f, 4096, K)));
        tg.B.Output("z", tg.B.MatMul(n, tg.RandW("w2", DataType.BF16, 0.05f, 48, K)));
        var g = tg.B.MatMul(n, tg.RandW("wg", type, 0.05f, 2048, K));
        var u = tg.B.MatMul(n, tg.RandW("wu", type, 0.05f, 2048, K));
        tg.B.Output("h", tg.B.Mul(tg.B.Silu(g), u));
        Diff.Check("NormHalfCopy", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: 2e-2, rtol: 2e-2, maxTokens: 128,
            what: $"{fmt} T={t}");
    }

    /// <summary>
    /// From 128 rows a norm whose every reader is a tensor-core GEMM writes the GEMMs' activations (f16 copy or E4M3 rows of the
    /// fp8 GEMM) and, once all readers are known to take them, no f32 rows (<c>SHARPLIGHT_NORM_PREFILL_COPY</c>): residual add +
    /// norm → matmul and gate / up pair, below and above that size. With <paramref name="keepNorm"/> the norm output is also a
    /// graph output, so the norm must keep writing its f32 rows.
    /// </summary>
    [Theory]
    [InlineData("f8e4m3_b128", 100, false)]
    [InlineData("f8e4m3_b128", 130, false)]
    [InlineData("f8e4m3_b128", 300, false)]
    [InlineData("f8e4m3_b128", 300, true)]
    [InlineData("bf16", 130, false)]
    [InlineData("bf16", 300, true)]
    [InlineData("f16", 300, false)]
    [InlineData("q4_k", 300, false)]
    public void NormThenPrefillGemms(string fmt, int t, bool keepNorm)
    {
        const int K = 1024;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var r = tg.InF("r", TG.T, K);
        var sum = tg.B.Add(x, r);
        var n = tg.B.RmsNorm(sum, tg.RandW("nw", DataType.F32, 1f, K), 1e-6f);
        var type = DataType.Get(fmt);
        tg.B.Output("s", sum);
        tg.B.Output("y", tg.B.MatMul(n, tg.RandW("w", type, 0.05f, 1024, K)));
        var g = tg.B.MatMul(n, tg.RandW("wg", type, 0.05f, 2048, K));
        var u = tg.B.MatMul(n, tg.RandW("wu", type, 0.05f, 2048, K));
        tg.B.Output("h", tg.B.Mul(tg.B.Silu(g), u));
        if (keepNorm)
        {
            tg.B.Output("n", n);
        }

        Diff.Check("NormPrefillCopy", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3)), (r, Rnd.F((long)t * K, 4))), atol: 2e-2, rtol: 2e-2,
            maxTokens: 320, what: $"{fmt} T={t} keepNorm={keepNorm}");
    }

    /// <summary>
    /// Residual add + norm of 4096-value rows (the 1024-thread norm of decode-size steps on CUDA, the 256-thread one above one wave of
    /// rows and on Vulkan) writing E4M3 rows, q8_1 rows (Q8_0 int-dot readers) or the f16 copy, at single-row, decode and prefill sizes.
    /// </summary>
    [Theory]
    [InlineData("f8e4m3_b128", 1)]
    [InlineData("f8e4m3_b128", 40)]
    [InlineData("f8e4m3_b128", 64)]
    [InlineData("f8e4m3_b128", 130)]
    [InlineData("q8_0", 1)]
    [InlineData("q8_0", 16)]
    [InlineData("bf16", 64)]
    public void WideRowsNormThenGemms(string fmt, int t)
    {
        const int K = 4096;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var r = tg.InF("r", TG.T, K);
        var sum = tg.B.Add(x, r);
        var n = tg.B.RmsNorm(sum, tg.RandW("nw", DataType.F32, 1f, K), 1e-6f);
        var type = DataType.Get(fmt);
        tg.B.Output("s", sum);
        tg.B.Output("y", tg.B.MatMul(n, tg.RandW("w", type, 0.05f, 1024, K)));
        var g = tg.B.MatMul(n, tg.RandW("wg", type, 0.05f, 1024, K));
        var u = tg.B.MatMul(n, tg.RandW("wu", type, 0.05f, 1024, K));
        tg.B.Output("h", tg.B.Mul(tg.B.Silu(g), u));
        Diff.Check("NormWideRows", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3)), (r, Rnd.F((long)t * K, 4))), atol: 4e-2, rtol: 2e-2,
            maxTokens: 160, what: $"{fmt} T={t}");
    }
}
