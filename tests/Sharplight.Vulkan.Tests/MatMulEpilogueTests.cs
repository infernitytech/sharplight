using System.Diagnostics;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Elementwise groups consuming a weight matmul (bias, GELU, per-column scale, residual) applied in the split-f16 GEMM's store
/// (<see cref="Shaders.Lowering.GraphLowerer"/> matmul epilogues), on the plain and split-K GEMMs, and as a separate launch
/// after the matvec for a few rows, all against the CPU.
/// </summary>
public sealed class MatMulEpilogueTests
{
    /// <summary>y = Gelu(x·Wᵀ + b) (a transformer MLP's first projection) for matvec, split-K and plain GEMM row counts.</summary>
    [Theory]
    [InlineData(768, 3072)]
    [InlineData(3072, 768)]
    [InlineData(96, 384)]
    public void BiasGelu(int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var w = tg.Const("w", Rnd.F((long)n * k, 2, 0.03f), n, k);
        var bias = tg.Const("b", Rnd.F(n, 3, 0.5f), n);
        tg.B.Output("y", tg.B.Unary("Gelu", tg.B.Add(tg.B.MatMul(x, w), bias)));
        var g = tg.B.Build();
        Step Rows(int t) => new(t, 0, (x, Rnd.F((long)t * k, 1 + t)));
        Diff.Check("MatMul+bias+Gelu", g, [Rows(4), Rows(300), Rows(1576)], atol: 2e-5, rtol: 2e-5, maxTokens: 1576, what: $"{k}->{n}");
    }

    /// <summary>
    /// q / k / v = norm(x)·Wᵀ + b with bias weights (a transformer layer's attention projections: one multi-output GEMM at 9-256 rows)
    /// and a 32-wide router with its bias, for single-row, skinny, split-K and plain GEMM row counts: f16 weights in gpt-oss-20b's
    /// shapes (the biases in the split-K reduction), Q8_0 / Q4_K weights in gpt-oss / Qwen2.5-1.5B / 7B shapes and FP8 weights (the
    /// biases as their own launch after the GEMM, which must follow the multi-output GEMM that writes the products).
    /// </summary>
    [Theory]
    [InlineData("f16", 2880, 4096, 512)]
    [InlineData("f16", 2880, 1024, 128)]
    [InlineData("q8_0", 2880, 4096, 512)]
    [InlineData("q8_0", 1536, 1536, 256)]
    [InlineData("q8_0", 3584, 3584, 512)]
    [InlineData("q4_k", 3584, 3584, 512)]
    [InlineData("f8e4m3_b128", 3584, 3584, 512)]
    public void ProjectionBiases(string type, int k, int qWidth, int kvWidth)
    {
        var weightType = DataType.Get(type);
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        // The projections read a norm's rows, as in a transformer layer: an arena value whose activation copies they share (the
        // multi-output GEMM needs one copy for all members).
        var h = tg.B.RmsNorm(x, tg.WF("norm", Rnd.F(k, 2, 1f), k), 1e-6f);
        foreach (var (name, n) in new[] { ("q", qWidth), ("k", kvWidth), ("v", kvWidth), ("router", 32) })
        {
            var w = tg.RandW("w" + name, weightType, 0.03f, n, k);
            var bias = tg.WF("b" + name, Rnd.F(n, 3 + n, 0.5f), n);
            tg.B.Output(name, tg.B.Add(tg.B.MatMul(h, w), bias));
        }

        var g = tg.B.Build();
        Step Rows(int t) => new(t, 0, (x, Rnd.F((long)t * k, 1 + t)));
        // Int-dot weights: the q8_1 activations of the int-dot rows err by up to ~1 %; fp8 weights: the E4M3 activations by up to ~3 %
        // per value. A bias read before its product is written (or a product missing its bias, 0.5 on average) is far off either way.
        bool fp8 = type.StartsWith("f8", StringComparison.Ordinal);
        double tolerance = fp8 ? 0.15 : weightType.IsQuantized ? 6e-2 : 2e-3;
        Diff.Check($"MatMul({type}) q/k/v/router + bias", g, [Rows(1), Rows(5), Rows(16), Rows(40), Rows(64), Rows(300)], atol: tolerance,
            rtol: fp8 ? 5e-2 : tolerance,
            maxTokens: 300, what: $"{k}->{qWidth}+{kvWidth}+{kvWidth}");
    }

    /// <summary>
    /// q / k / v of prompt-sized batches, with or without bias rows (the hand-written f16 GEMM's multi-output form over the members'
    /// weight rows on CUDA: from 256 rows without biases, from 1,024 with them; its 128 x 256 tiles where every member boundary is a
    /// multiple of 256, else its 128 x 128 tiles), and a 32-wide router (gemm_cm2's K split at every row count).
    /// </summary>
    [Theory]
    [InlineData(2880, 4096, 512, true)]
    [InlineData(2880, 1024, 128, false)]
    public void PromptProjections(int k, int qWidth, int kvWidth, bool bias)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var h = tg.B.RmsNorm(x, tg.WF("norm", Rnd.F(k, 2, 1f), k), 1e-6f);
        foreach (var (name, n) in new[] { ("q", qWidth), ("k", kvWidth), ("v", kvWidth), ("router", 32) })
        {
            var w = tg.RandW("w" + name, DataType.F16, 0.03f, n, k);
            var product = tg.B.MatMul(h, w);
            tg.B.Output(name, bias ? tg.B.Add(product, tg.WF("b" + name, Rnd.F(n, 3 + n, 0.5f), n)) : product);
        }

        var g = tg.B.Build();
        Step Rows(int t) => new(t, 0, (x, Rnd.F((long)t * k, 1 + t)));
        Diff.Check($"MatMul(f16) prompt q/k/v/router{(bias ? " + bias" : "")}", g, [Rows(300), Rows(1030)], atol: 2e-3, rtol: 2e-3, maxTokens: 1030,
            what: $"{k}->{qWidth}+{kvWidth}+{kvWidth}");
    }

    /// <summary>
    /// <see cref="ProjectionBiases"/> with the biases of f16 / bf16 projections as their own launch instead of in the split-K
    /// reduction (<c>SHARPLIGHT_SPLITK_EPILOGUE=0</c>, read once per process, so in a child run): the path 572eeb80 took for
    /// gpt-oss-20b, whose q / k / v biases read the products before the multi-output GEMM wrote them.
    /// </summary>
    [Fact]
    public void ProjectionBiasesSeparateEpilogue()
    {
        Gpu.Require();
        string method = $"{typeof(MatMulEpilogueTests).FullName}.{nameof(ProjectionBiases)}";
        string host = Environment.ProcessPath ?? throw new InvalidOperationException("no process path");
        var info = new ProcessStartInfo(host) { RedirectStandardOutput = true, RedirectStandardError = true };
        if (Path.GetFileNameWithoutExtension(host) == "dotnet")
        {
            info.ArgumentList.Add(typeof(MatMulEpilogueTests).Assembly.Location);
        }

        info.ArgumentList.Add("-method");
        info.ArgumentList.Add(method);
        info.Environment["SHARPLIGHT_SPLITK_EPILOGUE"] = "0";
        using var process = Process.Start(info) ?? throw new InvalidOperationException("child test run did not start");
        var stderr = process.StandardError.ReadToEndAsync(TestContext.Current.CancellationToken);
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, $"child run failed ({process.ExitCode}):\n{output}\n{stderr.Result}");
        Assert.Contains("Total: 7,", output);
        Assert.Contains("Failed: 0,", output);
    }

    /// <summary>y = (x·Wᵀ + b)·γ + r (a ConvNeXt block's last projection with its layer scale and the residual).</summary>
    [Theory]
    [InlineData(1536, 384)]
    [InlineData(384, 96)]
    public void BiasScaleResidual(int k, int n)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        var r = tg.InF("r", TG.T, n);
        var w = tg.Const("w", Rnd.F((long)n * k, 2, 0.03f), n, k);
        var bias = tg.Const("b", Rnd.F(n, 3, 0.5f), n);
        var gamma = tg.Const("gamma", Rnd.F(n, 4, 0.1f), n);
        tg.B.Output("y", tg.B.Add(tg.B.Mul(tg.B.Add(tg.B.MatMul(x, w), bias), gamma), r));
        var g = tg.B.Build();
        Step Rows(int t) => new(t, 0, (x, Rnd.F((long)t * k, 1 + t)), (r, Rnd.F((long)t * n, 7 + t)));
        Diff.Check("MatMul+bias*gamma+residual", g, [Rows(3), Rows(197), Rows(2000)], atol: 2e-5, rtol: 2e-5, maxTokens: 2000, what: $"{k}->{n}");
    }
}
