using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class StatefulTests
{
    [Theory]
    [InlineData("f16", 16, 16, 6, 2)]
    [InlineData("f32", 16, 16, 6, 2)]
    [InlineData("f16", 24, 32, 4, 4)]
    [InlineData("f32", 64, 64, 8, 1)]
    [InlineData("f16", 128, 128, 4, 2)]
    [InlineData("q8_0", 64, 64, 8, 1)]
    [InlineData("q8_0", 128, 128, 4, 2)]
    [InlineData("q4_0", 128, 128, 4, 2)]
    [InlineData("q4_0", 64, 96, 6, 3)]
    public void CachedAttentionAcrossSteps(string cacheType, int D, int Dv, int Hq, int Hkv)
        => CachedAttention(cacheType, D, Dv, Hq, Hkv, [5, 3, 1, 1, 7, 1], 40);

    // Long prefill chunks: several key blocks, partial query blocks, GQA (the tensor-core flash-attention path when available).
    [Theory]
    [InlineData("f16", 128, 128, 8, 2)]
    [InlineData("f16", 256, 256, 4, 1)]
    [InlineData("f16", 64, 128, 6, 3)]
    [InlineData("f32", 128, 128, 4, 2)]
    [InlineData("q8_0", 128, 128, 8, 2)]
    [InlineData("q4_0", 256, 256, 4, 1)]
    public void CachedAttentionLongPrefill(string cacheType, int D, int Dv, int Hq, int Hkv)
        => CachedAttention(cacheType, D, Dv, Hq, Hkv, [70, 33, 1, 5], 120);

    private static void CachedAttention(string cacheType, int D, int Dv, int Hq, int Hkv, int[] chunks, int ctx)
    {
        Gpu.Require();
        float scale = 1f / MathF.Sqrt(D);
        int total = chunks.Sum();
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, Dv);
        var ct = DataType.Get(cacheType);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, Dv);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
        tg.B.Output("o", tg.B.CachedAttention(qi, kc, vc, scale));
        var q = Rnd.F(total * Hq * D, 1);
        var k = Rnd.F(total * Hkv * D, 2);
        var v = Rnd.F(total * Hkv * Dv, 3);
        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]), (vi, v[(past * Hkv * Dv)..((past + c) * Hkv * Dv)])));
            past += c;
        }
        // f16 caches may run on tensor cores with f16 queries and probabilities: allow for their rounding. Quantized caches are
        // quantized by both backends, which may round a value to neighbouring quants (one quant step of a block).
        var (atol, rtol) = cacheType switch { "f16" => (3e-4, 3e-3), "f32" => (3e-5, 2e-4), "q8_0" => (2e-3, 1e-2), _ => (2e-2, 5e-2) };
        Diff.Check("CachedAttention", tg.B.Build(), steps, atol: atol, rtol: rtol, maxTokens: chunks.Max(), ctx: ctx, what: $"{cacheType} D={D} Hq={Hq} Hkv={Hkv}");
    }

    /// <summary>Sliding window, soft cap and ALiBi biases on the scalar, split-K decode and flash kernels.</summary>
    [Theory]
    [InlineData(4, 0f, false)]
    [InlineData(0, 20f, false)]
    [InlineData(5, 30f, false)]
    [InlineData(150, 0f, false)]
    [InlineData(0, 0f, true)]
    [InlineData(5, 0f, true)]
    public void CachedAttentionWindowAndSoftcap(int window, float softcap, bool alibi)
    {
        Gpu.Require();
        int Hq = 4;
        int Hkv = 2;
        int D = 32;
        int ctx = 256;
        float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1, 180, 1];
        int total = chunks.Sum();
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
        var attrs = new List<(string, object)> { ("scale", scale), ("offset", "n_past") };
        if (window > 0)
        {
            attrs.Add(("window", window));
        }

        if (softcap > 0)
        {
            attrs.Add(("softcap", softcap));
        }

        if (alibi)
        {
            attrs.Add(("alibi_slopes", new[] { 0.25f, 0.0625f, 0.015625f, 0.5f }));
        }

        tg.B.RuntimeScalar("n_past");
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc], Attributes.From([.. attrs])));
        var q = Rnd.F(total * Hq * D, 1).Select(x => x * 4).ToArray();
        var k = Rnd.F(total * Hkv * D, 2);
        var v = Rnd.F(total * Hkv * D, 3);
        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]), (vi, v[(past * Hkv * D)..((past + c) * Hkv * D)])));
            past += c;
        }
        // f16 caches may run on the tensor-core flash kernel (f16 Q and P).
        Diff.Check("CachedAttention.window", tg.B.Build(), steps, atol: 1e-3, rtol: 3e-3, maxTokens: 192, ctx: ctx, what: $"window={window} softcap={softcap} alibi={alibi}");
    }

    /// <summary>
    /// One cache for keys and values (multi-head latent attention): <c>v_dims</c> values are the prefix of each D-wide row. The
    /// single-row steps take the split-K decode kernel, the long chunk the flash kernel.
    /// </summary>
    [Theory]
    [InlineData("f16", 96, 64, 4, 4)]
    [InlineData("f32", 96, 64, 8, 2)]
    public void CachedAttentionValuePrefix(string cacheType, int D, int valueDims, int Hq, int Hkv)
        => ValuePrefixAttention(cacheType, D, valueDims, Hq, Hkv, [6, 3, 1, 1, 180, 1], 256, 1f);

    /// <summary>
    /// Absorbed multi-head latent attention shapes (DeepSeek-V2: 576-wide rows, 512 values, 16 heads; MiniCPM3: 288 / 256,
    /// 40 heads, so a query's heads span three 16-row tiles): the tensor-core <c>fa_mqa</c> kernel for prefill chunks and, for
    /// the single rows at a few hundred keys, its split-key form with the combine. f32 caches keep the exact kernels.
    /// </summary>
    [Theory]
    [InlineData("f16", 576, 512, 16, 1)]
    [InlineData("f16", 288, 256, 40, 1)]
    [InlineData("f16", 576, 512, 32, 2)]
    [InlineData("f32", 576, 512, 16, 1)]
    [InlineData("q8_0", 576, 512, 16, 1)]
    [InlineData("q4_0", 288, 256, 40, 1)]
    public void CachedAttentionLatentShapes(string cacheType, int D, int valueDims, int Hq, int Hkv)
        => ValuePrefixAttention(cacheType, D, valueDims, Hq, Hkv, [70, 3, 1, 250, 1, 1, 40], 384, 8f);

    // queryScale > 1 spreads the scores, so running maxima grow during the key loop (the lazy rescale of fa_mqa).
    private static void ValuePrefixAttention(string cacheType, int D, int valueDims, int Hq, int Hkv, int[] chunks, int ctx, float queryScale)
    {
        Gpu.Require();
        int total = chunks.Sum();
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var kc = tg.State("kc", DataType.Get(cacheType), StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki);
        tg.B.RuntimeScalar("n_past");
        var attrs = Attributes.From(("scale", 1f / MathF.Sqrt(D)), ("offset", "n_past"), ("v_dims", (long)valueDims));
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, kc], attrs));
        var q = Rnd.F(total * Hq * D, 1, queryScale);
        var k = Rnd.F(total * Hkv * D, 2);
        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)])));
            past += c;
        }
        var (atol, rtol) = cacheType switch { "q8_0" => (5e-3, 1e-2), "q4_0" => (3e-2, 5e-2), _ => (1e-3, 3e-3) };
        Diff.Check("CachedAttention.vprefix", tg.B.Build(), steps, atol: atol, rtol: rtol, maxTokens: chunks.Max(), ctx: ctx, what: $"{cacheType} D={D} v_dims={valueDims}");
    }

    // Attention sinks: decode (split-K + combine), short chunks, and long prefill chunks (flash kernel), full and windowed.
    [Theory]
    [InlineData("f16", 0)]
    [InlineData("f16", 17)]
    [InlineData("f32", 0)]
    [InlineData("f32", 40)]
    public void CachedAttentionSinks(string cacheType, int window)
    {
        Gpu.Require();
        int Hq = 8;
        int Hkv = 2;
        int D = 64;
        int ctx = 256;
        float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1, 150, 1, 1];
        int total = chunks.Sum();
        var tg = new TG();
        var T = TG.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, D);
        var ct = DataType.Get(cacheType);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        var sinks = tg.RandW("sinks", DataType.F32, 4f, Hq);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
        var attrs = new List<(string, object)> { ("scale", scale), ("offset", "n_past") };
        if (window > 0)
        {
            attrs.Add(("window", window));
        }

        tg.B.RuntimeScalar("n_past");
        attrs.Add(("sinks", true));
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc, sinks], Attributes.From([.. attrs])));
        var q = Rnd.F(total * Hq * D, 1).Select(x => x * 3).ToArray();
        var k = Rnd.F(total * Hkv * D, 2);
        var v = Rnd.F(total * Hkv * D, 3);
        var steps = new List<Step>();
        int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]), (vi, v[(past * Hkv * D)..((past + c) * Hkv * D)])));
            past += c;
        }
        Diff.Check("CachedAttention.sinks", tg.B.Build(), steps, atol: 1e-3, rtol: 3e-3, maxTokens: 160, ctx: ctx, what: $"{cacheType} window={window}");
    }

    [Fact]
    public void CachedAttentionWithStridedQueryAndKv()
    {
        Gpu.Require();
        int Hq = 4;
        int Hkv = 2;
        int D = 16;
        int ctx = 32;
        var tg = new TG();
        var T = TG.T;
        var qf = tg.InF("qf", T, Hq, 2 * D);                      // q and gate interleaved per head (Qwen3.5 style)
        var q = tg.B.Slice(qf, 2, 0, D);
        var kf = tg.InF("kf", T, Hkv, 2 * D);
        var k = tg.B.Slice(kf, 2, D, D);
        var vf = tg.InF("vf", T, Hkv, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, k); tg.B.CacheWrite(vc, vf);
        tg.B.Output("o", tg.B.CachedAttention(q, kc, vc, 0.25f));
        int[] chunks = [4, 1, 2]; int[] pasts = [0, 4, 5];
        var steps = chunks.Select((c, i) => new Step(c, pasts[i], (qf, Rnd.F(c * Hq * 2 * D, 10 + i)), (kf, Rnd.F(c * Hkv * 2 * D, 20 + i)), (vf, Rnd.F(c * Hkv * D, 30 + i)))).ToList();
        Diff.Check("CachedAttention.strided", tg.B.Build(), steps, atol: 3e-5, rtol: 2e-4, maxTokens: 8, ctx: ctx);
    }

    [Theory]
    [InlineData(20, 4, true)]
    [InlineData(20, 4, false)]
    [InlineData(600, 4, true)]
    [InlineData(33, 2, true)]
    [InlineData(16, 6, false)]
    public void CausalConvAcrossSteps(int C, int K, bool silu)
    {
        Gpu.Require();
        var tg = new TG();
        var x = tg.InF("x", TG.T, C);
        var w = tg.RandW("w", DataType.F32, 1f, C, K);
        var state = tg.State("cs", DataType.F32, StateKind.Recurrent, K - 1, C);
        tg.B.Output("y", tg.B.Emit("CausalConv1d", [x, w, state], Attributes.From(("activation", silu ? "silu" : "none")), writes: [state])[0]);
        int[] chunks = [3, 1, 1, 5, 2];
        var steps = chunks.Select((c, i) => new Step(c, 0, (x, Rnd.F((long)c * C, 100 + i)))).ToList();
        Diff.Check("CausalConv1d", tg.B.Build(), steps, atol: 2e-5, rtol: 1e-4, maxTokens: 8, what: $"C={C} K={K} silu={silu}");
    }

    [Theory]
    [InlineData(2, 4, 16, 8)]
    [InlineData(1, 3, 128, 128)]
    [InlineData(4, 4, 32, 64)]
    [InlineData(2, 2, 64, 32)]
    public void GatedDeltaRuleAcrossSteps(int Hk, int Hv, int Dk, int Dv)
    {
        Gpu.Require();
        var tg = new TG();
        var D = TG.T;
        var qi = tg.InF("q", D, Hk, Dk);
        var ki = tg.InF("k", D, Hk, Dk);
        var vi = tg.InF("v", D, Hv, Dv);
        var gi = tg.InF("g", D, Hv);
        var bi = tg.InF("b", D, Hv);
        var st = tg.State("ssm", DataType.F32, StateKind.Recurrent, Hv, Dv, Dk);
        tg.B.Output("o", tg.B.Emit("GatedDeltaRule", [qi, ki, vi, gi, bi, st], Attributes.From(("scale", 1f / MathF.Sqrt(Dk))), writes: [st])[0]);
        int[] chunks = [3, 1, 2, 1];
        var steps = chunks.Select((c, i) => new Step(c, 0,
            (qi, Rnd.F(c * Hk * Dk, 1 + i, 1f)), (ki, Rnd.F(c * Hk * Dk, 11 + i, 0.3f)), (vi, Rnd.F(c * Hv * Dv, 21 + i)),
            (gi, Rnd.F(c * Hv, 31 + i).Select(v => -Math.Abs(v)).ToArray()), (bi, Rnd.F(c * Hv, 41 + i).Select(Math.Abs).ToArray()))).ToList();
        Diff.Check("GatedDeltaRule", tg.B.Build(), steps, atol: 2e-4, rtol: 2e-3, maxTokens: 8, what: $"Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}");
    }
}
