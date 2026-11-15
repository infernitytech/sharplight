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
}
