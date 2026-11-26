using System.Diagnostics;
using Sharplight.Runtime;
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

    // f16 activations on tensor cores (~1 % error); q8_0 and nvfp4 run their rows up to 4 or 8 on the integer-dot matvec
    // (8-bit activations, an error proportional to the weight rows' norms).
    private static double Atol(string fmt) => fmt is "q8_0" or "nvfp4" ? 3.0 : 1.5e-1;

    // The skinny GEMM's rows of NVFP4 weights go to the FP4 GEMM where the device has block_scaled_fp4 (Fp4GemmTests checks it).
    private static IReadOnlySet<string>? Disabled(string fmt) => fmt == "nvfp4" ? new HashSet<string> { DeviceFeatures.BlockScaledFp4 } : null;

    [Theory]
    [MemberData(nameof(Cases))]
    public void Single(string fmt, int t)
    {
        Gpu.Require();
        int K = 1024, N = 4112;                                     // N % 32 != 0: a partial last workgroup
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 1f, N, K)));
        Diff.Check("Skinny." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(fmt), rtol: 1e-2, what: $"T={t}",
            disabledFeatures: Disabled(fmt));
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
        // NVFP4 rows up to 8 run on the integer-dot matvec, where 8-bit activation rounding times those norms would exceed
        // even that bound: they get activations that 8-bit block quantization represents exactly, and a tight bound.
        bool exactActivations = fmt == "nvfp4" && t <= 8;
        double atol = exactActivations ? 1e-1 : fmt switch { "q8_0" => 6e-1, "nvfp4" => 3.0, _ => 1e-1 };
        var xs = exactActivations ? Rnd.ExactInt8((long)t * K, 3) : Rnd.F((long)t * K, 3);
        Diff.Check("Skinny.dual." + fmt, tg.B.Build(), new Step(t, 0, (x, xs)), atol: atol, rtol: 2e-2, what: $"T={t}", disabledFeatures: Disabled(fmt));
    }

    /// <summary>
    /// Long K, where the direct path splits K over 16 subgroups (GraphLowerer.DirectSkinnyWideMinBlocks: from 6144 single, 4096 dual).
    /// </summary>
    [Theory]
    [InlineData("f8e4m3_b128", 9, false)]
    [InlineData("f8e4m3_b128", 16, true)]
    [InlineData("q8_0", 12, false)]
    [InlineData("q8_0", 16, true)]
    public void LongK(string fmt, int t, bool dual)
    {
        Gpu.Require();
        int K = dual ? 4096 : 6144, N = 4096;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.1f, N, K));
        tg.B.Output("y", dual ? tg.B.Mul(tg.B.Silu(g), tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.1f, N, K))) : g);
        Diff.Check($"Skinny.long{(dual ? ".dual." : ".")}{fmt}", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))),
            atol: dual ? 6e-1 : 1.5e-1, rtol: 2e-2, what: $"T={t}");
    }

    /// <summary>
    /// Matmuls narrower than 4096 outputs, where the skinny GEMM serves at most 8 rows (1024-4095 outputs) or none (below 1024)
    /// and gemm_cm2 the rest: 9-16 rows read the shared f16 copy, which must cover them whatever the skinny GEMM reads.
    /// </summary>
    [Theory]
    [InlineData(1024, 9)]
    [InlineData(1024, 16)]
    [InlineData(2048, 12)]
    [InlineData(512, 9)]
    [InlineData(4096, 16)]
    public void NarrowAboveSkinnyRange(int n, int t)
    {
        Gpu.Require();
        const int K = 1024;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.BF16, 1f, n, K)));
        Diff.Check("Skinny.narrow", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: 1.5e-1, rtol: 1e-2, what: $"N={n} T={t}");
    }

    /// <summary>
    /// <see cref="NarrowAboveSkinnyRange"/> with the skinny GEMM converting f32 rows itself (<c>SHARPLIGHT_SKINNY_X16=0</c>, read once
    /// per process, so in a child run of this test assembly): the f16 copy then starts above the skinny range of each width. It
    /// started at 17 rows for every width, and 9-16 rows of matmuls below 4096 outputs returned zeros.
    /// </summary>
    [Fact]
    public void NarrowAboveSkinnyRangeWithF32Staging()
    {
        Gpu.Require();
        string method = $"{typeof(SkinnyGemmTests).FullName}.{nameof(NarrowAboveSkinnyRange)}";
        string host = Environment.ProcessPath ?? throw new InvalidOperationException("no process path");
        var info = new ProcessStartInfo(host) { RedirectStandardOutput = true, RedirectStandardError = true };
        if (Path.GetFileNameWithoutExtension(host) == "dotnet")
        {
            info.ArgumentList.Add(typeof(SkinnyGemmTests).Assembly.Location);
        }

        info.ArgumentList.Add("-method");
        info.ArgumentList.Add(method);
        info.Environment["SHARPLIGHT_SKINNY_X16"] = "0";
        using var process = Process.Start(info) ?? throw new InvalidOperationException("child test run did not start");
        var stderr = process.StandardError.ReadToEndAsync(TestContext.Current.CancellationToken);
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, $"child run failed ({process.ExitCode}):\n{output}\n{stderr.Result}");
        Assert.Contains("Total: 5,", output);
        Assert.Contains("Failed: 0,", output);
    }
}
