using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>The opt-in workgroup-scope matrix GEMM (DeviceFeatures.MatrixWorkgroup, VK_NV_cooperative_matrix2).</summary>
[Collection("WorkgroupMatrixGemm")]
public sealed class WorkgroupMatrixGemmTests
{
    [Theory]
    [InlineData("f16", 512, 96)]
    [InlineData("q8_0", 512, 300)]
    [InlineData("q8_0", 192, 130)]
    [InlineData("f16", 4096, 64)]
    [InlineData("q8_0", 4096, 520)]
    public void MatchesCpu(string fmt, int K, int N)
    {
        Gpu.Require();
        Assert.SkipUnless(Gpu.Vulkan!.EnabledFeatures.Contains(Runtime.DeviceFeatures.MatrixWorkgroup), "device has no workgroup-scope matrices");
        bool old = Shaders.Lowering.GraphLowerer.PreferWorkgroupMatrixGemm;
        Shaders.Lowering.GraphLowerer.PreferWorkgroupMatrixGemm = true;
        try
        {
            foreach (var t in new[] { 9, 20, 130, 300 })
            {
                var tg = new TG();
                var x = tg.InF("x", TG.T, K);
                var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
                tg.B.Output("y", tg.B.MatMul(x, w));
                // Up to 16 rows q8_0 with K % 256 == 0 runs on the integer-dot matvec (8-bit activations, see MatMulTests).
                double atol = t <= 16 && fmt == "q8_0" && K % 256 == 0 ? 3.0 : 1.5e-1;
                Diff.Check("MatMul.nv2." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: atol, rtol: 1e-2,
                    maxTokens: 300, what: $"T={t} K={K} N={N}");
            }
        }
        finally
        {
            Shaders.Lowering.GraphLowerer.PreferWorkgroupMatrixGemm = old;
        }
    }
}
