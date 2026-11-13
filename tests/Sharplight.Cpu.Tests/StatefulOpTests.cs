using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

public class StatefulOpTests
{
    [Fact]
    public void GatedDeltaRuleCarriesStateAcrossCalls()
    {
        int T = 4;
        int Hk = 2;
        int Hv = 4;
        int Dk = 16;
        int Dv = 8;
        float scale = 0.25f;
        var q = Kit.Rand(T * Hk * Dk, 1);
        var k = Kit.Rand(T * Hk * Dk, 2);
        var v = Kit.Rand(T * Hv * Dv, 3);
        var g = Kit.Rand(T * Hv, 4).Select(x => -Math.Abs(x)).ToArray();
        var b = Kit.Rand(T * Hv, 5).Select(x => Math.Abs(x)).ToArray();
        var S = new float[Hv * Dv * Dk];
        var exp = Ref.Delta(q, k, v, g, b, S, T, Hk, Hv, Dk, Dv, scale);
        var once = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [4], out var s1);
        var single = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [1, 1, 1, 1], out var s2);
        var split = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [3, 1], out var s3);
        Kit.AssertClose(exp, once, 1e-5, 1e-4, "T=4");
        Kit.AssertClose(exp, single, 1e-5, 1e-4, "4xT=1");
        Kit.AssertClose(exp, split, 1e-5, 1e-4, "3+1");
        Kit.AssertClose(S, s1, 1e-5, 1e-4);
        Kit.AssertClose(S, s2, 1e-5, 1e-4);
        Kit.AssertClose(S, s3, 1e-5, 1e-4);
    }

    [Fact]
    public void GatedDeltaRuleLargeHeadDimParallelRows()
    {
        int T = 3;
        int Hk = 1;
        int Hv = 3;
        int Dk = 128;
        int Dv = 128;
        float scale = 1f / MathF.Sqrt(Dk);
        var q = Kit.Rand(T * Hk * Dk, 1);
        var k = Kit.Rand(T * Hk * Dk, 2, 0.2f);
        var v = Kit.Rand(T * Hv * Dv, 3);
        var g = Kit.Rand(T * Hv, 4).Select(x => -Math.Abs(x)).ToArray();
        var b = Kit.Rand(T * Hv, 5).Select(x => Math.Abs(x)).ToArray();
        var S = new float[Hv * Dv * Dk];
        var exp = Ref.Delta(q, k, v, g, b, S, T, Hk, Hv, Dk, Dv, scale);
        var got = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [2, 1], out var st);
        Kit.AssertClose(exp, got, 1e-4, 1e-3);
        Kit.AssertClose(S, st, 1e-4, 1e-3);
    }

    [Theory]
    [InlineData(20, 4, true), InlineData(20, 4, false), InlineData(600, 4, true), InlineData(33, 2, true), InlineData(16, 6, false)]
    public void CausalConvStateCarryOver(int C, int K, bool silu)
    {
        int T = 7;
        var x = Kit.Rand(T * C, 1);
        var w = Kit.Rand(C * K, 2);
        var tg = new TestGraph();
        var xin = tg.InF("x", TestGraph.T, C);
        var wv = tg.WF("w", w, C, K);
        var state = tg.State("cs", DataType.F32, StateKind.Recurrent, K - 1, C);
        tg.B.Output("y", tg.B.Emit("CausalConv1d", [xin, wv, state], Attributes.From(("activation", silu ? "silu" : "none")), writes: [state])[0]);
        var graph = tg.B.Build();
        var refState = new float[(K - 1) * C];
        var exp = Ref.Conv(x, T, C, w, K, refState, silu);
        foreach (var chunks in new[] { new[] { 7 }, [1, 1, 1, 1, 1, 1, 1], [2, 5], [1, 2, 4], [3, 3, 1] })
        {
            using var m = Kit.Compile(graph, maxTokens: 16);
            var got = new List<float>();
            int t0 = 0;
            foreach (int c in chunks)
            {
                Kit.Set(m, xin, x[(t0 * C)..((t0 + c) * C)]);
                var p = Kit.P(c);
                m.Run(p);
                got.AddRange(Kit.Out(m, "y", p));
                t0 += c;
            }
            Kit.AssertClose(exp, got.ToArray(), 1e-5, 1e-4, $"chunks {string.Join("+", chunks)}");
            // ResetState restarts from zero history
            m.ResetState();
            Kit.Set(m, xin, x[..(3 * C)]);
            var p3 = Kit.P(3);
            m.Run(p3);
            var again = Kit.Out(m, "y", p3);
            Kit.AssertClose(exp[..(3 * C)], again, 1e-5, 1e-4, "after reset");
        }
    }

    /// <summary>KV-cache attention across two Execute calls (n_past advancing), GQA, f16 and f32 caches.</summary>
    [Theory]
    [InlineData("f16", 16, 16), InlineData("f32", 16, 16), InlineData("f16", 24, 32)]
    public void CachedAttentionAcrossCalls(string cacheType, int D, int Dv)
    {
        int Hq = 6;
        int Hkv = 2;
        int ctx = 40;
        float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [5, 3, 1, 1];
        int total = chunks.Sum();
        var q = Kit.Rand(total * Hq * D, 1);
        var k = Kit.Rand(total * Hkv * D, 2);
        var v = Kit.Rand(total * Hkv * Dv, 3);
        var ct = DataType.Get(cacheType);
        var tg = new TestGraph();
        var T = TestGraph.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, Dv);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, Dv);
        tg.B.CacheWrite(kc, ki);
        tg.B.CacheWrite(vc, vi);
        tg.B.Output("o", tg.B.CachedAttention(qi, kc, vc, scale));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        // reference keys/values rounded like the cache
        Func<float, float> rnd = cacheType == "f16" ? (x => (float)(Half)x) : (x => x);
        var kr = k.Select(rnd).ToArray();
        var vr = v.Select(rnd).ToArray();
        int past = 0;
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(past * Hq * D)..((past + c) * Hq * D)]);
            Kit.Set(m, ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]);
            Kit.Set(m, vi, v[(past * Hkv * Dv)..((past + c) * Hkv * Dv)]);
            var p = Kit.P(c, past);
            m.Run(p);
            var got = Kit.Out(m, "o", p);
            var exp = Ref.Attention(q[(past * Hq * D)..((past + c) * Hq * D)], c, Hq, D, kr, vr, past + c, Hkv, Dv, scale, past, true);
            Kit.AssertClose(exp, got, 2e-5, 2e-4, $"past {past} T {c}");
            past += c;
        }
    }

    /// <summary>Sliding-window and soft-capped cached attention (Gemma 2/3) across calls.</summary>
    [Theory]
    [InlineData(4, 0f), InlineData(0, 20f), InlineData(5, 30f), InlineData(100, 0f)]
    public void CachedAttentionWindowAndSoftcap(int window, float softcap)
    {
        int Hq = 4;
        int Hkv = 2;
        int D = 16;
        int ctx = 40;
        float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1];
        int total = chunks.Sum();
        var q = Kit.Rand(total * Hq * D, 1);
        var k = Kit.Rand(total * Hkv * D, 2);
        var v = Kit.Rand(total * Hkv * D, 3);
        for (int i = 0; i < q.Length; i++)
        {
            q[i] *= 4;   // large scores so that capping / windowing matter
        }

        var tg = new TestGraph();
        var T = TestGraph.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", DataType.F32, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", DataType.F32, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki);
        tg.B.CacheWrite(vc, vi);
        var attrs = new List<(string, object)> { ("scale", scale), ("offset", "n_past") };
        if (window > 0)
        {
            attrs.Add(("window", window));
        }

        if (softcap > 0)
        {
            attrs.Add(("softcap", softcap));
        }

        tg.B.RuntimeScalar("n_past");
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc], Attributes.From([.. attrs])));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        int past = 0;
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(past * Hq * D)..((past + c) * Hq * D)]);
            Kit.Set(m, ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]);
            Kit.Set(m, vi, v[(past * Hkv * D)..((past + c) * Hkv * D)]);
            var p = Kit.P(c, past);
            m.Run(p);
            var exp = Ref.Attention(q[(past * Hq * D)..((past + c) * Hq * D)], c, Hq, D, k, v, past + c, Hkv, D, scale, past, true, null, window, softcap);
            Kit.AssertClose(exp, Kit.Out(m, "o", p), 2e-5, 2e-4, $"window {window} cap {softcap} past {past} T {c}");
            past += c;
        }
    }

    /// <summary>Attention sinks (a learned per-head logit that only joins the softmax denominator), with and without a sliding window, f16/f32 caches.</summary>
    [Theory]
    [InlineData("f32", 0), InlineData("f16", 0), InlineData("f32", 5), InlineData("f16", 9)]
    public void CachedAttentionSinks(string cacheType, int window)
    {
        int Hq = 4;
        int Hkv = 2;
        int D = 16;
        int ctx = 40;
        float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1];
        int total = chunks.Sum();
        var q = Kit.Rand(total * Hq * D, 1);
        var k = Kit.Rand(total * Hkv * D, 2);
        var v = Kit.Rand(total * Hkv * D, 3);
        var sinks = new[] { 3.5f, -2f, 0.7f, 6f };
        var ct = DataType.Get(cacheType);
        var tg = new TestGraph();
        var T = TestGraph.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        var sk = tg.WF("sinks", sinks, Hq);
        tg.B.CacheWrite(kc, ki);
        tg.B.CacheWrite(vc, vi);
        var attrs = new List<(string, object)> { ("scale", scale), ("offset", "n_past") };
        if (window > 0)
        {
            attrs.Add(("window", window));
        }

        tg.B.RuntimeScalar("n_past");
        attrs.Add(("sinks", true));
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc, sk], Attributes.From([.. attrs])));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        Func<float, float> rnd = cacheType == "f16" ? (x => (float)(Half)x) : (x => x);
        var kr = k.Select(rnd).ToArray();
        var vr = v.Select(rnd).ToArray();
        int past = 0;
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(past * Hq * D)..((past + c) * Hq * D)]);
            Kit.Set(m, ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]);
            Kit.Set(m, vi, v[(past * Hkv * D)..((past + c) * Hkv * D)]);
            var p = Kit.P(c, past);
            m.Run(p);
            var exp = Ref.Attention(q[(past * Hq * D)..((past + c) * Hq * D)], c, Hq, D, kr, vr, past + c, Hkv, D, scale, past, true, null, window, 0f, sinks);
            Kit.AssertClose(exp, Kit.Out(m, "o", p), 2e-5, 2e-4, $"sinks window {window} past {past} T {c}");
            past += c;
        }
    }

    /// <summary>ALiBi: every score gets slope · (key position − query position) added, with and without a sliding window, f16/f32 caches.</summary>
    [Theory]
    [InlineData("f32", 0), InlineData("f16", 0), InlineData("f32", 5)]
    public void CachedAttentionAlibi(string cacheType, int window)
    {
        int Hq = 4;
        int Hkv = 2;
        int D = 16;
        int ctx = 40;
        float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1];
        int total = chunks.Sum();
        var q = Kit.Rand(total * Hq * D, 1);
        var k = Kit.Rand(total * Hkv * D, 2);
        var v = Kit.Rand(total * Hkv * D, 3);
        var slopes = new[] { 0.5f, 0.25f, 0.125f, 0.0625f };
        var ct = DataType.Get(cacheType);
        var tg = new TestGraph();
        var T = TestGraph.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki);
        tg.B.CacheWrite(vc, vi);
        var attrs = new List<(string, object)> { ("scale", scale), ("offset", "n_past"), ("alibi_slopes", slopes) };
        if (window > 0)
        {
            attrs.Add(("window", window));
        }

        tg.B.RuntimeScalar("n_past");
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc], Attributes.From([.. attrs])));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        Func<float, float> rnd = cacheType == "f16" ? (x => (float)(Half)x) : (x => x);
        var kr = k.Select(rnd).ToArray();
        var vr = v.Select(rnd).ToArray();
        int past = 0;
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(past * Hq * D)..((past + c) * Hq * D)]);
            Kit.Set(m, ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]);
            Kit.Set(m, vi, v[(past * Hkv * D)..((past + c) * Hkv * D)]);
            var p = Kit.P(c, past);
            m.Run(p);
            var exp = Ref.Attention(q[(past * Hq * D)..((past + c) * Hq * D)], c, Hq, D, kr, vr, past + c, Hkv, D, scale, past, true, null, window, 0f, null, slopes);
            Kit.AssertClose(exp, Kit.Out(m, "o", p), 2e-5, 2e-4, $"alibi window {window} past {past} T {c}");
            past += c;
        }
    }

    [Fact]
    public void CacheOverflowThrowsAndResetStateClears()
    {
        int Hq = 2;
        int Hkv = 1;
        int D = 8;
        int ctx = 4;
        var tg = new TestGraph();
        var T = TestGraph.T;
        var qi = tg.InF("q", T, Hq, D);
        var ki = tg.InF("k", T, Hkv, D);
        var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
        var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki);
        tg.B.CacheWrite(vc, vi);
        tg.B.Output("o", tg.B.CachedAttention(qi, kc, vc, 0.3f));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        Kit.Set(m, qi, Kit.Rand(3 * Hq * D, 1));
        Kit.Set(m, ki, Kit.Rand(3 * D, 2));
        Kit.Set(m, vi, Kit.Rand(3 * D, 3));
        m.Run(Kit.P(3, 0));
        Assert.Throws<InvalidOperationException>(() => m.Run(Kit.P(3, 3)));
        var first = Kit.Out(m, "o", Kit.P(3));
        m.ResetState();
        var p = Kit.P(3, 0);
        m.Run(p);
        Kit.AssertClose(first, Kit.Out(m, "o", p), 0, 0);
    }

    [Theory, InlineData(true, false), InlineData(false, false), InlineData(true, true)]
    public void PlainAttention(bool causal, bool useMask)
    {
        int T = 6;
        int S = 6;
        int Hq = 4;
        int Hkv = 2;
        int D = 16;
        int Dv = 8;
        float scale = 0.2f;
        var q = Kit.Rand(T * Hq * D, 1);
        var k = Kit.Rand(S * Hkv * D, 2);
        var v = Kit.Rand(S * Hkv * Dv, 3);
        var mask = Kit.Rand(T * S, 4);
        var tg = new TestGraph();
        var Tm = TestGraph.T;
        var qi = tg.InF("q", Tm, Hq, D);
        var ki = tg.InF("k", Tm, Hkv, D);
        var vi = tg.InF("v", Tm, Hkv, Dv);
        var mi = tg.InF("mask", Tm, S);
        var ins = useMask ? new[] { qi, ki, vi, mi } : [qi, ki, vi];
        tg.B.Output("o", tg.B.Emit1("Attention", ins, Attributes.From(("scale", scale), ("causal", causal))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, qi, q);
        Kit.Set(m, ki, k);
        Kit.Set(m, vi, v);
        Kit.Set(m, mi, mask);
        var p = Kit.P(T);
        m.Run(p);
        var exp = Ref.Attention(q, T, Hq, D, k, v, S, Hkv, Dv, scale, 0, causal, useMask ? mask : null);
        Kit.AssertClose(exp, Kit.Out(m, "o", p), 2e-5, 2e-4);
    }
}
