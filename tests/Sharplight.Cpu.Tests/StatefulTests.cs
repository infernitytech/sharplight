using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Cross-checks against outputs recorded from llama.cpp's ggml CPU backend (rope, ssm_conv, gated_delta_net).</summary>
public class GgmlCrossCheckTests
{
    private static float[] Floats(BinaryReader r, int n) { var a = new float[n]; for (int i = 0; i < n; i++) { a[i] = r.ReadSingle(); } return a; }
    private static int[] Ints(BinaryReader r, int n) { var a = new int[n]; for (int i = 0; i < n; i++) { a[i] = r.ReadInt32(); } return a; }

    private record RopeRec(int Mode, int Dims, int[] Sec, float FreqScale, int T, int H, int D, float[] X, int[] Pos, float[] Out);
    private record DeltaRec(int T, int Hk, int Hv, int S, float[] Q, float[] K, float[] V, float[] G, float[] B, float[] State, float[] Out);
    private record ConvRec(int T, int C, int K, float[] State, float[] X, float[] W, float[] Out);

    private static (List<RopeRec> Rope, List<DeltaRec> Delta, List<ConvRec> Conv) Load()
    {
        var r = new BinaryReader(File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml_ops.bin")));
        r.ReadInt32();
        List<RopeRec> rope = []; List<DeltaRec> delta = []; List<ConvRec> conv = [];
        while (r.BaseStream.Position < r.BaseStream.Length)
        {
            int tag = r.ReadInt32();
            if (tag == 1)
            {
                int mode = r.ReadInt32(), dims = r.ReadInt32(); var sec = Ints(r, 4); float fs = r.ReadSingle();
                int T = r.ReadInt32(), H = r.ReadInt32(), D = r.ReadInt32();
                rope.Add(new(mode, dims, sec, fs, T, H, D, Floats(r, T * H * D), Ints(r, T * 4), Floats(r, T * H * D)));
            }
            else if (tag == 2)
            {
                int T = r.ReadInt32(), Hk = r.ReadInt32(), Hv = r.ReadInt32(), S = r.ReadInt32();
                delta.Add(new(T, Hk, Hv, S, Floats(r, S * Hk * T), Floats(r, S * Hk * T), Floats(r, S * Hv * T), Floats(r, Hv * T), Floats(r, Hv * T),
                    Floats(r, S * S * Hv), Floats(r, S * Hv * T + S * S * Hv)));
            }
            else
            {
                int T = r.ReadInt32(), C = r.ReadInt32(), K = r.ReadInt32();
                conv.Add(new(T, C, K, Floats(r, (K - 1) * C), Floats(r, T * C), Floats(r, K * C), Floats(r, T * C)));
            }
        }
        return (rope, delta, conv);
    }

    [Fact]
    public void RopeMatchesGgml()
    {
        var (ropes, _, _) = Load();
        Assert.Equal(7, ropes.Count);
        foreach (var c in ropes)
        {
            bool multi = c.Mode >= 8;
            string style = c.Mode == 0 ? "gptj" : "neox";
            int axes = multi ? 4 : 1;
            var posIn = multi ? c.Pos : Enumerable.Range(0, c.T).Select(t => c.Pos[t * 4]).ToArray();
            var tg = new TG(); var xin = tg.InF("x", TG.T, c.H, c.D); var pin = tg.In("pos", DataType.I32, TG.T, axes);
            var attrs = Attributes.From(("dims", c.Dims), ("base", 10000f), ("style", style), ("interleaved", c.Mode == 40), ("freq_scale", c.FreqScale)).With("sections", multi ? c.Sec : new int[4]);
            tg.B.Output("y", tg.B.Emit1("Rope", [xin, pin], attrs));
            using var m = Kit.Compile(tg.B.Build()); Kit.Set(m, xin, c.X); Kit.Set(m, pin, posIn);
            var p = Kit.P(c.T); m.Run(p);
            Kit.AssertClose(c.Out, Kit.Out(m, "y", p), 2e-4, 2e-4, $"rope mode {c.Mode} dims {c.Dims}");
        }
    }

    [Fact]
    public void GatedDeltaRuleMatchesGgml()
    {
        var (_, deltas, _) = Load();
        Assert.Equal(3, deltas.Count);
        foreach (var c in deltas)
        {
            var o = RunDelta(c.Q, c.K, c.V, c.G, c.B, c.State, c.T, c.Hk, c.Hv, c.S, c.S, 1f / MathF.Sqrt(c.S), [c.T], out var finalState);
            Kit.AssertClose(c.Out[..(c.S * c.Hv * c.T)], o, 1e-4, 1e-4, "delta out");
            Kit.AssertClose(c.Out[(c.S * c.Hv * c.T)..], finalState, 1e-4, 1e-4, "delta state");
        }
    }

    [Fact]
    public void CausalConvMatchesGgmlSsmConv()
    {
        var (_, _, convs) = Load();
        Assert.Equal(3, convs.Count);
        foreach (var c in convs)
        {
            var tg = new TG(); var xin = tg.InF("x", TG.T, c.C); var wv = tg.WF("w", c.W, c.C, c.K);
            var state = tg.State("cs", DataType.F32, StateKind.Recurrent, c.K - 1, c.C);
            tg.B.Output("y", tg.B.Emit("CausalConv1d", [xin, wv, state], Attributes.From(("activation", "none")), writes: [state])[0]);
            using var m = Kit.Compile(tg.B.Build());
            ((CpuCompiledSegment)m.Segments[0]).TestSetState(state, c.State);
            Kit.Set(m, xin, c.X);
            var p = Kit.P(c.T); m.Run(p);
            Kit.AssertClose(c.Out, Kit.Out(m, "y", p), 1e-4, 1e-4, "ssm_conv");
        }
    }

    internal static float[] RunDelta(float[] q, float[] k, float[] v, float[] g, float[] beta, float[] state0, int T, int Hk, int Hv, int Dk, int Dv, float scale, int[] chunks, out float[] finalState)
    {
        var tg = new TG(); var D = TG.T;
        var qi = tg.InF("q", D, Hk, Dk); var ki = tg.InF("k", D, Hk, Dk); var vi = tg.InF("v", D, Hv, Dv); var gi = tg.InF("g", D, Hv); var bi = tg.InF("b", D, Hv);
        var st = tg.State("ssm", DataType.F32, StateKind.Recurrent, Hv, Dv, Dk);
        tg.B.Output("o", tg.B.Emit("GatedDeltaRule", [qi, ki, vi, gi, bi, st], Attributes.From(("scale", scale)), writes: [st])[0]);
        using var m = Kit.Compile(tg.B.Build(), maxTokens: Math.Max(8, T));
        var o = new float[T * Hv * Dv];
        int t0 = 0;
        if (state0 != null)
        {
            // Load an initial state: with zero state and a single decayed token we cannot inject arbitrary state, so
            // use the internal API through the compiled segment instead.
            var seg = (CpuCompiledSegment)m.Segments[0];
            seg.TestSetState(st, state0);
        }
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(t0 * Hk * Dk)..((t0 + c) * Hk * Dk)]); Kit.Set(m, ki, k[(t0 * Hk * Dk)..((t0 + c) * Hk * Dk)]);
            Kit.Set(m, vi, v[(t0 * Hv * Dv)..((t0 + c) * Hv * Dv)]); Kit.Set(m, gi, g[(t0 * Hv)..((t0 + c) * Hv)]); Kit.Set(m, bi, beta[(t0 * Hv)..((t0 + c) * Hv)]);
            var p = Kit.P(c); m.Run(p);
            Kit.Out(m, "o", p).CopyTo(o, t0 * Hv * Dv);
            t0 += c;
        }
        finalState = ((CpuCompiledSegment)m.Segments[0]).TestGetState(st);
        return o;
    }
}

public class StatefulOpTests
{
    [Fact]
    public void GatedDeltaRuleCarriesStateAcrossCalls()
    {
        int T = 4, Hk = 2, Hv = 4, Dk = 16, Dv = 8; float scale = 0.25f;
        var q = Kit.Rand(T * Hk * Dk, 1); var k = Kit.Rand(T * Hk * Dk, 2); var v = Kit.Rand(T * Hv * Dv, 3);
        var g = Kit.Rand(T * Hv, 4).Select(x => -Math.Abs(x)).ToArray(); var b = Kit.Rand(T * Hv, 5).Select(x => Math.Abs(x)).ToArray();
        var S = new float[Hv * Dv * Dk];
        var exp = Ref.Delta(q, k, v, g, b, S, T, Hk, Hv, Dk, Dv, scale);
        var once = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [4], out var s1);
        var single = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [1, 1, 1, 1], out var s2);
        var split = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [3, 1], out var s3);
        Kit.AssertClose(exp, once, 1e-5, 1e-4, "T=4"); Kit.AssertClose(exp, single, 1e-5, 1e-4, "4xT=1"); Kit.AssertClose(exp, split, 1e-5, 1e-4, "3+1");
        Kit.AssertClose(S, s1, 1e-5, 1e-4); Kit.AssertClose(S, s2, 1e-5, 1e-4); Kit.AssertClose(S, s3, 1e-5, 1e-4);
    }

    [Fact]
    public void GatedDeltaRuleLargeHeadDimParallelRows()
    {
        int T = 3, Hk = 1, Hv = 3, Dk = 128, Dv = 128; float scale = 1f / MathF.Sqrt(Dk);
        var q = Kit.Rand(T * Hk * Dk, 1); var k = Kit.Rand(T * Hk * Dk, 2, 0.2f); var v = Kit.Rand(T * Hv * Dv, 3);
        var g = Kit.Rand(T * Hv, 4).Select(x => -Math.Abs(x)).ToArray(); var b = Kit.Rand(T * Hv, 5).Select(x => Math.Abs(x)).ToArray();
        var S = new float[Hv * Dv * Dk];
        var exp = Ref.Delta(q, k, v, g, b, S, T, Hk, Hv, Dk, Dv, scale);
        var got = GgmlCrossCheckTests.RunDelta(q, k, v, g, b, null!, T, Hk, Hv, Dk, Dv, scale, [2, 1], out var st);
        Kit.AssertClose(exp, got, 1e-4, 1e-3); Kit.AssertClose(S, st, 1e-4, 1e-3);
    }

    [Theory]
    [InlineData(20, 4, true), InlineData(20, 4, false), InlineData(600, 4, true), InlineData(33, 2, true), InlineData(16, 6, false)]
    public void CausalConvStateCarryOver(int C, int K, bool silu)
    {
        int T = 7;
        var x = Kit.Rand(T * C, 1); var w = Kit.Rand(C * K, 2);
        var tg = new TG(); var xin = tg.InF("x", TG.T, C); var wv = tg.WF("w", w, C, K);
        var state = tg.State("cs", DataType.F32, StateKind.Recurrent, K - 1, C);
        tg.B.Output("y", tg.B.Emit("CausalConv1d", [xin, wv, state], Attributes.From(("activation", silu ? "silu" : "none")), writes: [state])[0]);
        var graph = tg.B.Build();
        var refState = new float[(K - 1) * C];
        var exp = Ref.Conv(x, T, C, w, K, refState, silu);
        foreach (var chunks in new[] { new[] { 7 }, [1, 1, 1, 1, 1, 1, 1], [2, 5], [1, 2, 4], [3, 3, 1] })
        {
            using var m = Kit.Compile(graph, maxTokens: 16);
            var got = new List<float>(); int t0 = 0;
            foreach (int c in chunks)
            {
                Kit.Set(m, xin, x[(t0 * C)..((t0 + c) * C)]);
                var p = Kit.P(c); m.Run(p); got.AddRange(Kit.Out(m, "y", p)); t0 += c;
            }
            Kit.AssertClose(exp, got.ToArray(), 1e-5, 1e-4, $"chunks {string.Join("+", chunks)}");
            // ResetState restarts from zero history
            m.ResetState();
            Kit.Set(m, xin, x[..(3 * C)]); var p3 = Kit.P(3); m.Run(p3);
            var again = Kit.Out(m, "y", p3);
            Kit.AssertClose(exp[..(3 * C)], again, 1e-5, 1e-4, "after reset");
        }
    }

    /// <summary>KV-cache attention across two Execute calls (n_past advancing), GQA, f16 and f32 caches.</summary>
    [Theory]
    [InlineData("f16", 16, 16), InlineData("f32", 16, 16), InlineData("f16", 24, 32)]
    public void CachedAttentionAcrossCalls(string cacheType, int D, int Dv)
    {
        int Hq = 6, Hkv = 2, ctx = 40; float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [5, 3, 1, 1];
        int total = chunks.Sum();
        var q = Kit.Rand(total * Hq * D, 1); var k = Kit.Rand(total * Hkv * D, 2); var v = Kit.Rand(total * Hkv * Dv, 3);
        var ct = DataType.Get(cacheType);
        var tg = new TG(); var T = TG.T;
        var qi = tg.InF("q", T, Hq, D); var ki = tg.InF("k", T, Hkv, D); var vi = tg.InF("v", T, Hkv, Dv);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D); var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, Dv);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
        tg.B.Output("o", tg.B.CachedAttention(qi, kc, vc, scale));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        // reference keys/values rounded like the cache
        Func<float, float> rnd = cacheType == "f16" ? (x => (float)(Half)x) : (x => x);
        var kr = k.Select(rnd).ToArray(); var vr = v.Select(rnd).ToArray();
        int past = 0;
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(past * Hq * D)..((past + c) * Hq * D)]); Kit.Set(m, ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]); Kit.Set(m, vi, v[(past * Hkv * Dv)..((past + c) * Hkv * Dv)]);
            var p = Kit.P(c, past); m.Run(p);
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
        int Hq = 4, Hkv = 2, D = 16, ctx = 40; float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1];
        int total = chunks.Sum();
        var q = Kit.Rand(total * Hq * D, 1); var k = Kit.Rand(total * Hkv * D, 2); var v = Kit.Rand(total * Hkv * D, 3);
        for (int i = 0; i < q.Length; i++)
        {
            q[i] *= 4;   // large scores so that capping / windowing matter
        }

        var tg = new TG(); var T = TG.T;
        var qi = tg.InF("q", T, Hq, D); var ki = tg.InF("k", T, Hkv, D); var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", DataType.F32, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D); var vc = tg.State("vc", DataType.F32, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
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

        tg.B.RuntimeScalar("n_past");
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc], Attributes.From([.. attrs])));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        int past = 0;
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(past * Hq * D)..((past + c) * Hq * D)]); Kit.Set(m, ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]); Kit.Set(m, vi, v[(past * Hkv * D)..((past + c) * Hkv * D)]);
            var p = Kit.P(c, past); m.Run(p);
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
        int Hq = 4, Hkv = 2, D = 16, ctx = 40; float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1];
        int total = chunks.Sum();
        var q = Kit.Rand(total * Hq * D, 1); var k = Kit.Rand(total * Hkv * D, 2); var v = Kit.Rand(total * Hkv * D, 3);
        var sinks = new[] { 3.5f, -2f, 0.7f, 6f };
        var ct = DataType.Get(cacheType);
        var tg = new TG(); var T = TG.T;
        var qi = tg.InF("q", T, Hq, D); var ki = tg.InF("k", T, Hkv, D); var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D); var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        var sk = tg.WF("sinks", sinks, Hq);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
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
        var kr = k.Select(rnd).ToArray(); var vr = v.Select(rnd).ToArray();
        int past = 0;
        foreach (int c in chunks)
        {
            Kit.Set(m, qi, q[(past * Hq * D)..((past + c) * Hq * D)]); Kit.Set(m, ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]); Kit.Set(m, vi, v[(past * Hkv * D)..((past + c) * Hkv * D)]);
            var p = Kit.P(c, past); m.Run(p);
            var exp = Ref.Attention(q[(past * Hq * D)..((past + c) * Hq * D)], c, Hq, D, kr, vr, past + c, Hkv, D, scale, past, true, null, window, 0f, sinks);
            Kit.AssertClose(exp, Kit.Out(m, "o", p), 2e-5, 2e-4, $"sinks window {window} past {past} T {c}");
            past += c;
        }
    }

    [Fact]
    public void CacheOverflowThrowsAndResetStateClears()
    {
        int Hq = 2, Hkv = 1, D = 8, ctx = 4;
        var tg = new TG(); var T = TG.T;
        var qi = tg.InF("q", T, Hq, D); var ki = tg.InF("k", T, Hkv, D); var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D); var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
        tg.B.Output("o", tg.B.CachedAttention(qi, kc, vc, 0.3f));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8, ctx: ctx);
        Kit.Set(m, qi, Kit.Rand(3 * Hq * D, 1)); Kit.Set(m, ki, Kit.Rand(3 * D, 2)); Kit.Set(m, vi, Kit.Rand(3 * D, 3));
        m.Run(Kit.P(3, 0));
        Assert.Throws<InvalidOperationException>(() => m.Run(Kit.P(3, 3)));
        var first = Kit.Out(m, "o", Kit.P(3));
        m.ResetState();
        var p = Kit.P(3, 0); m.Run(p);
        Kit.AssertClose(first, Kit.Out(m, "o", p), 0, 0);
    }

    [Theory, InlineData(true, false), InlineData(false, false), InlineData(true, true)]
    public void PlainAttention(bool causal, bool useMask)
    {
        int T = 6, S = 6, Hq = 4, Hkv = 2, D = 16, Dv = 8; float scale = 0.2f;
        var q = Kit.Rand(T * Hq * D, 1); var k = Kit.Rand(S * Hkv * D, 2); var v = Kit.Rand(S * Hkv * Dv, 3);
        var mask = Kit.Rand(T * S, 4);
        var tg = new TG(); var Tm = TG.T;
        var qi = tg.InF("q", Tm, Hq, D); var ki = tg.InF("k", Tm, Hkv, D); var vi = tg.InF("v", Tm, Hkv, Dv); var mi = tg.InF("mask", Tm, S);
        var ins = useMask ? new[] { qi, ki, vi, mi } : [qi, ki, vi];
        tg.B.Output("o", tg.B.Emit1("Attention", ins, Attributes.From(("scale", scale), ("causal", causal))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, qi, q); Kit.Set(m, ki, k); Kit.Set(m, vi, v); Kit.Set(m, mi, mask);
        var p = Kit.P(T); m.Run(p);
        var exp = Ref.Attention(q, T, Hq, D, k, v, S, Hkv, Dv, scale, 0, causal, useMask ? mask : null);
        Kit.AssertClose(exp, Kit.Out(m, "o", p), 2e-5, 2e-4);
    }
}
