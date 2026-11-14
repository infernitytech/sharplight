using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Graph patterns the Vulkan lowerer fuses into single kernels (checked against the unfused CPU reference):
/// elementwise + row norm, norm + elementwise, dual matvec + elementwise epilogue, L2Norm inside the delta rule,
/// RmsNorm + rope (+ cache write).
/// </summary>
public sealed class FusionTests
{
    [Theory]
    [InlineData(1, 64)]
    [InlineData(5, 100)]
    [InlineData(3, 5120)]
    public void ResidualAddThenRmsNorm(int t, int d)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, d);
        var r = tg.InF("r", TG.T, d);
        var h = tg.B.Add(x, r);
        tg.B.Output("h", h);                                       // group output still materialized
        tg.B.Output("n", tg.B.RmsNorm(h, tg.RandW("w", DataType.F32, 1f, d), 1e-6f));
        tg.B.Output("l2", tg.B.L2Norm(tg.B.Mul(x, x), 1e-6f));   // group feeding L2Norm, intermediate not materialized
        Diff.Check("Fusion.add+norm", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * d, 1)), (r, Rnd.F((long)t * d, 2))), atol: 2e-5, rtol: 1e-4, what: $"T={t} D={d}");
    }

    [Theory]
    [InlineData(1, 4, 128)]
    [InlineData(6, 3, 32)]
    [InlineData(2, 2, 300)]
    public void GatedRmsNorm(int t, int h, int d)
    {
        Gpu.Require();
        var tg = new TG();
        var o = tg.InF("o", TG.T, h, d);
        var z = tg.InF("z", TG.T, h * d);
        var n = tg.B.RmsNorm(o, tg.RandW("w", DataType.F32, 1f, d), 1e-6f);
        tg.B.Output("y", tg.B.Mul(n, tg.B.Silu(tg.B.Reshape(z, 0, h, d))));
        Diff.Check("Fusion.norm+gate", tg.B.Build(), new Step(t, 0, (o, Rnd.F((long)t * h * d, 1)), (z, Rnd.F((long)t * h * d, 2))), atol: 2e-5, rtol: 1e-4, what: $"T={t}");
    }

    public static IEnumerable<object[]> DualCases()
    {
        foreach (var fmt in new[] { "f32", "bf16", "f16", "f8e4m3_b128", "q8_0", "q4_k" })
        {
            foreach (var t in new[] { 1, 3, 8, 9, 16, 20, 70, 140 })
            {
                yield return [fmt, t];
            }
        }
    }

    [Theory]
    [MemberData(nameof(DualCases))]
    public void SwiGluDualMatvec(string fmt, int t)
    {
        Gpu.Require();
        int K = 1024, N = 96;                                      // K >= 1024: split-K decode kernel
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.1f, N, K));
        var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.1f, N, K));
        tg.B.Output("y", tg.B.Mul(tg.B.Silu(g), u));
        // Small K variant goes through the one-subgroup-per-row matvec.
        var x2 = tg.InF("x2", TG.T, 256);
        var g2 = tg.B.MatMul(x2, tg.RandW("wg2", DataType.Get(fmt), 0.1f, 40, 256));
        var u2 = tg.B.MatMul(x2, tg.RandW("wu2", DataType.Get(fmt), 0.1f, 40, 256));
        tg.B.Output("y2", tg.B.Mul(g2, tg.B.Sigmoid(u2)));
        // From 2 rows on tensor-core products with f16 activations (~1 % error; the skinny GEMM for 2-16 rows of formats without
        // an integer-dot matvec, gemm_cm2 above). For q8_0 / q4_k the integer-dot matvec quantizes the activations to
        // 8 bits, an absolute error proportional to the weight row norms (large: random K-quant rows carry big offsets), which
        // silu(g) * u multiplies by the other factor.
        bool intDot = fmt is "q8_0" or "q4_k";
        double atol = intDot ? 6e-1 : t > 1 ? 1e-1 : 2e-3, rtol = intDot || t > 1 ? 1e-2 : 1e-3;
        // 17+ rows of fp8 weights: the dual fp8 GEMM (E4M3 activations, 64- and 128-row tiles).
        Diff.Check("Fusion.dual." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3)), (x2, Rnd.F((long)t * 256, 4))), atol: atol, rtol: rtol,
            maxTokens: Math.Max(64, t), what: $"T={t}");
    }

    [Theory]
    [InlineData(2, 4, 32, 16)]
    [InlineData(1, 2, 128, 128)]
    public void L2NormedQkInDeltaRule(int Hk, int Hv, int Dk, int Dv)
    {
        Gpu.Require();
        var tg = new TG();
        var T = TG.T;
        var qkv = tg.InF("qkv", T, 2 * Hk * Dk + Hv * Dv);         // one strided buffer, as after the causal conv
        var q = tg.B.L2Norm(tg.B.Reshape(tg.B.Slice(qkv, 1, 0, Hk * Dk), 0, Hk, Dk), 1e-6f);
        var k = tg.B.L2Norm(tg.B.Reshape(tg.B.Slice(qkv, 1, Hk * Dk, Hk * Dk), 0, Hk, Dk), 1e-6f);
        var v = tg.B.Reshape(tg.B.Slice(qkv, 1, 2 * Hk * Dk, Hv * Dv), 0, Hv, Dv);
        var gi = tg.InF("g", T, Hv);
        var bi = tg.InF("b", T, Hv);
        var st = tg.State("ssm", DataType.F32, StateKind.Recurrent, Hv, Dv, Dk);
        tg.B.Output("o", tg.B.Emit("GatedDeltaRule", [q, k, v, gi, bi, st], Attributes.From(("scale", 1f / MathF.Sqrt(Dk))), writes: [st])[0]);
        int C = 2 * Hk * Dk + Hv * Dv;
        int[] chunks = [3, 1, 2, 1];
        var steps = chunks.Select((c, i) => new Step(c, 0, (qkv, Rnd.F(c * C, 1 + i)),
            (gi, Rnd.F(c * Hv, 31 + i).Select(x => -Math.Abs(x)).ToArray()), (bi, Rnd.F(c * Hv, 41 + i).Select(Math.Abs).ToArray()))).ToList();
        Diff.Check("Fusion.l2+delta", tg.B.Build(), steps, atol: 2e-4, rtol: 2e-3, maxTokens: 8, what: $"Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}");
    }

    [Theory]
    [InlineData("f16", 16, 16)]
    [InlineData("f32", 32, 32)]
    [InlineData("f16", 256, 64)]
    public void NormRopeIntoCache(string cacheType, int D, int rot)
    {
        Gpu.Require();
        int Hq = 4;
        int Hkv = 2;
        int ctx = 32;
        var tg = new TG();
        var T = TG.T;
        var qf = tg.InF("qf", T, Hq, 2 * D);                       // q and gate interleaved per head (strided q)
        var q = tg.B.Slice(qf, 2, 0, D);
        var ki = tg.InF("k", T, Hkv * D);
        var vi = tg.InF("v", T, Hkv, D);
        var pos = tg.In("pos", DataType.I32, T, 4);
        var rope = Attributes.From(("dims", rot), ("base", 10000f), ("style", "neox"), ("interleaved", true)).With("sections", new[] { rot / 8, rot / 8, rot / 4, 0 });
        q = tg.B.Emit1("Rope", [tg.B.RmsNorm(q, tg.RandW("qn", DataType.F32, 1f, D), 1e-6f), pos], rope);
        var k = tg.B.Emit1("Rope", [tg.B.RmsNorm(tg.B.Reshape(ki, 0, Hkv, D), tg.RandW("kn", DataType.F32, 1f, D), 1e-6f), pos], rope);
        var ct = DataType.Get(cacheType);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, k); tg.B.CacheWrite(vc, vi);
        tg.B.Output("o", tg.B.CachedAttention(q, kc, vc, 1f / MathF.Sqrt(D)));
        int[] chunks = [5, 1, 3, 1];
        int past = 0;
        var steps = new List<Step>();
        foreach (var (c, i) in chunks.Select((c, i) => (c, i)))
        {
            var p = Enumerable.Range(0, c * 4).Select(j => past + j / 4).ToArray();
            steps.Add(new Step(c, past, (qf, Rnd.F(c * Hq * 2 * D, 10 + i)), (ki, Rnd.F(c * Hkv * D, 20 + i)), (vi, Rnd.F(c * Hkv * D, 30 + i)), (pos, p)));
            past += c;
        }
        Diff.Check("Fusion.norm+rope+cache", tg.B.Build(), steps, atol: 3e-4, rtol: 2e-3, maxTokens: 8, ctx: ctx, what: $"{cacheType} D={D}");
    }
}
