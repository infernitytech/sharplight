using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Strict-precision f32 matmuls run on the SIMT gemm (16- and 64-token tiles, split over K for small grids). One
/// compiled model is run at several row counts so every tile height and split factor is selected at run time.
/// </summary>
public sealed class SimtGemmTests
{
    [Theory]
    [InlineData(768, 768)]
    [InlineData(3072, 96)]
    [InlineData(200, 520)]
    public void RowCountsAndSplits(int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var w = tg.RandW("w", DataType.F32, 0.05f, n, k);
        tg.B.Output("y", tg.B.MatMul(x, w));
        int[] rows = [9, 16, 20, 33, 64, 197];
        var steps = rows.Select((t, i) => new Step(t, 0, (x, Rnd.F((long)t * k, 20 + i)))).ToArray();
        Diff.Check("MatMul.simt", tg.B.Build(), steps, atol: 1e-4, rtol: 1e-4, maxTokens: 256, what: $"K={k} N={n}");
    }

    /// <summary>
    /// Quantized weights take the same SIMT kernels when the matrix features are disabled (and the int-dot matvec, which
    /// would otherwise take the 12-row step of q8_0).
    /// </summary>
    [Theory]
    [InlineData("q8_0")]
    [InlineData("q4_k")]
    [InlineData("f16")]
    public void WithoutMatrixFeatures(string format)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, 512);
        var w = tg.RandW("w", DataType.Get(format), 1f, 96, 512);
        tg.B.Output("y", tg.B.MatMul(x, w));
        int[] rows = [12, 40, 100];
        var steps = rows.Select((t, i) => new Step(t, 0, (x, Rnd.F((long)t * 512, 30 + i)))).ToArray();
        Diff.Check("MatMul.simt." + format, tg.B.Build(), steps, atol: 2e-3, rtol: 1e-3, maxTokens: 128, what: format,
            disabledFeatures: new HashSet<string> { DeviceFeatures.MatrixF16, DeviceFeatures.MatrixWorkgroup, DeviceFeatures.IntDot });
    }

    /// <summary>
    /// Large grids take the 128 x 128 gemm_wide (unsplit and split over K): partial row and column tiles, K not a multiple of
    /// its 16-value step, and the same tiles for a non-f32 weight format (load8 path) without matrix features.
    /// </summary>
    [Theory]
    [InlineData("f32", 768, 3072)]
    [InlineData("f32", 200, 520)]
    [InlineData("f32", 1544, 300)]
    [InlineData("f16", 768, 384)]
    public void WideTiles(string format, int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var w = tg.RandW("w", DataType.Get(format), 0.05f, n, k);
        tg.B.Output("y", tg.B.MatMul(x, w));
        int[] rows = [512, 600, 1576];
        var steps = rows.Select((t, i) => new Step(t, 0, (x, Rnd.F((long)t * k, 40 + i)))).ToArray();
        Diff.Check("MatMul.simt.wide." + format, tg.B.Build(), steps, atol: 1e-4, rtol: 1e-4, maxTokens: 1576, what: $"{format} K={k} N={n}",
            disabledFeatures: format == "f32" ? null : new HashSet<string> { DeviceFeatures.MatrixF16, DeviceFeatures.MatrixWorkgroup, DeviceFeatures.IntDot });
    }
}
