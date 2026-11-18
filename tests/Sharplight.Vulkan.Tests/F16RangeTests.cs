using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Activations beyond f16's range (65504) on the tensor-core GEMMs, which read f16 copies of their activations: the copies of
/// values no norm bounds are row-ranged (<c>f16_range_scale</c> in common.slang), so a row reaching 2^15 is scaled by a power of
/// two and its outputs unscaled. Qwen3-TTS's code predictor (a small Qwen3 decoder) reaches 1.5e5 at a down projection's input
/// from hidden states with outliers near 100; before, those rows became f16 infinities on the skinny GEMM (2-16 rows) and
/// gemm_cm2, and the dual gemm_cm2's f16 copy of its gate / up product.
/// </summary>
public sealed class F16RangeTests
{
    /// <summary>
    /// A weight matmul whose activation rows hold values up to 1.5e5 next to ordinary ones, at row counts on the skinny GEMM
    /// (2-16), split-K gemm_cm2 (40, 130) and gemm_cm2 with the separate unscaling launch (600), and rows without outliers (left
    /// unscaled).
    /// </summary>
    [Theory]
    [InlineData("bf16", 2)]
    [InlineData("bf16", 5)]
    [InlineData("bf16", 16)]
    [InlineData("bf16", 40)]
    [InlineData("bf16", 130)]
    [InlineData("f16", 5)]
    [InlineData("f16", 130)]
    [InlineData("bf16", 600)]
    public void MatMul_reads_rows_beyond_f16_range(string fmt, int t)
    {
        Gpu.Require();
        const int K = 3072;
        const int N = 1024;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 0.05f, N, K)));
        tg.B.Output("y2", tg.B.MatMul(x, tg.RandW("w2", DataType.Get(fmt), 0.05f, 4096, K)));
        var xs = Rnd.F((long)t * K, 3);
        for (int r = 0; r < t; r += 2)
        {
            xs[((long)r * K) + 17] = 1.5e5f;
            xs[((long)r * K) + 900] = -7e4f;
            xs[((long)r * K) + 2047] = 4e4f;
        }

        // Errors scale with the outliers (f16 rounding of 1.5e5 times weights of 0.05: about 4 per term).
        Diff.Check("F16Range.matmul", tg.B.Build(), new Step(t, 0, (x, xs)), atol: 8.0, rtol: 1e-2, maxTokens: 640, what: $"{fmt} T={t}");
    }

    /// <summary>
    /// A tiny transformer FFN block fed hidden states with outliers near 100 (as Qwen3-TTS's code predictor gets from its talker):
    /// residual + RMS norm → gate / up → silu(g)·u (beyond 65504 in some rows) → down projection → residual, at row counts on the
    /// skinny GEMMs (2, 5), split-K gemm_cm2 (130, 300) and the dual gemm_cm2 storing its f16 copy of the product (600 rows of a
    /// 8192-wide FFN: the product's rows past 2^15 are flagged and rescaled by the fixup launch).
    /// </summary>
    [Theory]
    [InlineData("bf16", 2, 3072)]
    [InlineData("bf16", 5, 3072)]
    [InlineData("bf16", 130, 3072)]
    [InlineData("f16", 300, 3072)]
    [InlineData("bf16", 600, 8192)]
    public void Ffn_with_massive_activations(string fmt, int t, int intermediate)
    {
        Gpu.Require();
        const int K = 1024;
        int I = intermediate;
        var type = DataType.Get(fmt);
        var tg = new TG();
        var h = tg.InF("h", TG.T, K);
        var ones = new float[K];
        Array.Fill(ones, 1f);
        var n = tg.B.RmsNorm(h, tg.WF("nw", ones, K), 1e-6f);
        var g = tg.B.MatMul(n, tg.RandW("wg", type, 16f, I, K));
        var u = tg.B.MatMul(n, tg.RandW("wu", type, 16f, I, K));
        var a = tg.B.Mul(tg.B.Silu(g), u);
        var d = tg.B.MatMul(a, tg.RandW("wd", type, 0.002f, K, I));
        tg.B.Output("y", tg.B.Add(h, d));
        var hs = Rnd.F((long)t * K, 5);
        for (int r = 0; r < t; r++)
        {
            hs[((long)r * K) + 7] = 95f;
            hs[((long)r * K) + 300] = -60f;
        }

        // The f16 rounding of the norm output, times gate / up weights of 16, leaves errors of up to about 10 in y with or without
        // outliers; f16 infinities would leave NaN or infinite rows.
        Diff.Check("F16Range.ffn", tg.B.Build(), new Step(t, 0, (h, hs)), atol: 15.0, rtol: 2e-2, maxTokens: 640,
            what: $"{fmt} T={t} I={I}");
    }

    /// <summary>
    /// The grouped expert GEMM (MatMulId above the per-item matvec range) reads its activations as expert-sorted f16 rows
    /// (<c>moe_gather_x16</c>): rows with values up to 1.5e5, as a down projection's per-slot input (a gated FFN's product) or as
    /// one token row shared by the slots, at item counts on the 32-, 64- and 128-row tiles. The gather scales rows reaching 2^15
    /// and gemm_cm2 undoes the scale in its store; before, those rows gave infinities.
    /// </summary>
    [Theory]
    [InlineData("bf16", 40, true)]
    [InlineData("bf16", 40, false)]
    [InlineData("mxfp4", 48, true)]
    [InlineData("f16", 300, true)]
    [InlineData("bf16", 1300, true)]
    public void Expert_gemm_reads_rows_beyond_f16_range(string fmt, int t, bool perSlot)
    {
        Gpu.Require();
        const int E = 8, S = 4, K = 1024, N = 256;
        var tg = new TG();
        var x = perSlot ? tg.InF("x", TG.T, S, K) : tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, S);
        tg.B.Output("y", tg.B.Emit1("MatMulId", [x, tg.RandW("w", DataType.Get(fmt), 0.05f, E, N, K), ids]));
        int rows = perSlot ? t * S : t;
        var xs = Rnd.F((long)rows * K, 3);
        for (int r = 0; r < rows; r += 3)
        {
            xs[((long)r * K) + 17] = 1.5e5f;
            xs[((long)r * K) + 900] = -7e4f;
            xs[((long)r * K) + 1001] = 4e4f;
        }

        var rng = new Random(5 + t);
        int[] idData = [.. Enumerable.Range(0, t * S).Select(_ => rng.Next(E))];
        // Errors scale with the outliers (f16 rounding of 1.5e5 times weights of 0.05), as in the dense case.
        Diff.Check("F16Range.moe", tg.B.Build(), new Step(t, 0, (x, xs), (ids, idData)), atol: 8.0, rtol: 1e-2, maxTokens: t,
            what: $"{fmt} T={t} perSlot={perSlot}");
    }

    /// <summary>
    /// A gate / up expert pair over rows beyond f16's range (the dual grouped gemm_cm2 above the skinny range): both products of a
    /// scaled row are unscaled before the fused silu(gate) · up, on the 32-, 64- and 128-row tiles.
    /// </summary>
    [Theory]
    [InlineData("bf16", 40)]
    [InlineData("mxfp4", 48)]
    [InlineData("mxfp4", 300)]
    [InlineData("f16", 1300)]
    public void Dual_expert_gemm_reads_rows_beyond_f16_range(string fmt, int t)
    {
        Gpu.Require();
        const int E = 8, S = 4, K = 1024, N = 256;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var ids = tg.In("ids", DataType.I32, TG.T, S);
        var gate = tg.B.Emit1("MatMulId", [x, tg.RandW("gate", DataType.Get(fmt), 0.05f, E, N, K), ids]);
        var up = tg.B.Emit1("MatMulId", [x, tg.RandW("up", DataType.Get(fmt), 0.05f, E, N, K), ids]);
        tg.B.Output("y", tg.B.Mul(tg.B.Silu(gate), up));
        var xs = Rnd.F((long)t * K, 3);
        for (int r = 0; r < t; r += 3)
        {
            xs[((long)r * K) + 17] = 1.5e5f;
            xs[((long)r * K) + 900] = -7e4f;
        }

        var rng = new Random(7 + t);
        int[] idData = [.. Enumerable.Range(0, t * S).Select(_ => rng.Next(E))];
        Diff.Check("F16Range.moe-dual", tg.B.Build(), new Step(t, 0, (x, xs), (ids, idData)), atol: 8.0, rtol: 1e-2, maxTokens: t,
            what: $"{fmt} T={t}");
    }
}
