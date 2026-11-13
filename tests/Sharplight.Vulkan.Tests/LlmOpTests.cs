using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public class MatMulTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in new[] { "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k", "q5_k", "q6_k" })
            foreach (var t in new[] { 1, 2, 8, 9, 20, 64 })
                yield return [fmt, t];
    }

    // The T > 8 path may use reduced-precision (tensor core) products; allow ~1% there, tight bounds for the matvec path.
    static double Atol(int t) => t > 8 ? 1.5e-1 : 2e-3;
    static double Rtol(int t) => t > 8 ? 1e-2 : 1e-3;

    [Theory]
    [MemberData(nameof(Cases))]
    public void WeightFormat(string fmt, int t)
    {
        Gpu.Require();
        int K = 512, N = 96;
        var tg = new TG(); var x = tg.InF("x", TG.T, K);
        var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
        tg.B.Output("y", tg.B.MatMul(x, w));
        Diff.Check("MatMul." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(t), rtol: Rtol(t), what: $"T={t}");
    }

    [Theory]
    [InlineData("f32", 288, 100)] [InlineData("f16", 160, 33)] [InlineData("q8_0", 96, 130)] [InlineData("q4_0", 64, 7)] [InlineData("bf16", 40, 17)] [InlineData("f32", 16, 8)] [InlineData("f16", 64, 8)] [InlineData("f32", 24, 40)]
    public void OddShapes(string fmt, int K, int N)
    {
        Gpu.Require();
        foreach (var t in new[] { 1, 5, 17 })
        {
            var tg = new TG(); var x = tg.InF("x", TG.T, K);
            var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
            tg.B.Output("y", tg.B.MatMul(x, w));
            Diff.Check("MatMul.odd." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: Atol(t), rtol: Rtol(t), what: $"T={t} K={K} N={N}");
        }
    }

    [Theory] [InlineData(1)] [InlineData(12)]
    public void StridedAndHigherRankInputs(int t)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 8, 72);
        var xs = tg.B.Slice(x, 2, 8, 64);                      // [T,8,64] strided view, rows are not dense
        var w = tg.RandW("w", DataType.F32, 0.5f, 40, 64);
        tg.B.Output("y", tg.B.MatMul(xs, w));
        var xp = tg.B.Permute(x, 0, 2, 1);                     // [T,72,8]: inner axis not contiguous -> dense temp
        var w2 = tg.RandW("w2", DataType.F16, 0.5f, 24, 8);
        tg.B.Output("y2", tg.B.MatMul(xp, w2));
        Diff.Check("MatMul.strided", tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * 576, 3))), atol: Atol(t * 8), rtol: Rtol(t * 8), what: $"T={t}");
    }
}

public class NormTests
{
    [Theory]
    [InlineData("RmsNorm", 1, 64, true)] [InlineData("RmsNorm", 6, 2048, true)] [InlineData("RmsNorm", 5, 100, false)] [InlineData("RmsNorm", 3, 33, true)]
    [InlineData("L2Norm", 1, 64, false)] [InlineData("L2Norm", 6, 128, false)] [InlineData("L2Norm", 4, 50, false)]
    public void Dense(string op, int t, int d, bool weight)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, d);
        var args = new List<Value> { x };
        if (weight) args.Add(tg.RandW("w", DataType.F32, 1f, d));
        tg.B.Output("y", tg.B.Emit1(op, args, Attributes.From(("eps", 1e-6f))));
        Diff.Check(op, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * d, 3))), atol: 2e-5, rtol: 1e-4);
    }

    [Theory]
    [InlineData("RmsNorm", true)] [InlineData("RmsNorm", false)] [InlineData("L2Norm", false)]
    public void ThreeAndFourDimStrided(string op, bool weight)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 6, 48);
        var s = tg.B.Slice(x, 2, 8, 32);                        // [T,6,32], row stride 48
        var args = new List<Value> { s };
        if (weight) args.Add(tg.RandW("w", DataType.F32, 1f, 32));
        tg.B.Output("y", tg.B.Emit1(op, args, Attributes.From(("eps", 1e-5f))));
        var x4 = tg.InF("x4", TG.T, 3, 5, 24);
        tg.B.Output("y4", tg.B.Emit1(op, [x4], Attributes.From(("eps", 1e-5f))));
        var p = tg.B.Permute(x4, 0, 2, 1, 3);                   // [T,5,3,24], middle dims not collapsible -> dense temp
        tg.B.Output("yp", tg.B.Emit1(op, [p], Attributes.From(("eps", 1e-5f))));
        Diff.Check(op + ".strided", tg.B.Build(), new Step(5, 0, (x, Rnd.F(5 * 288, 3)), (x4, Rnd.F(5 * 360, 4))), atol: 2e-5, rtol: 1e-4);
    }
}

public class RopeTests
{
    public static IEnumerable<object[]> Cases() =>
    [
        ["neox", 16, 1, null!, false, 1f],
        ["gptj", 16, 1, null!, false, 1f],
        ["neox", 8, 1, null!, false, 0.5f],
        ["gptj", 8, 1, null!, false, 1f],
        ["neox", 16, 4, new[] { 2, 3, 3, 0 }, false, 1f],
        ["neox", 16, 4, new[] { 3, 2, 2, 1 }, true, 1f],
        ["neox", 16, 4, new[] { 4, 2, 2, 0 }, true, 1f],
        ["neox", 8, 4, new[] { 1, 1, 1, 1 }, true, 1f],
    ];

    [Theory]
    [MemberData(nameof(Cases))]
    public void Variants(string style, int dims, int axes, int[]? sections, bool interleaved, float freqScale)
    {
        Gpu.Require();
        int T = 6, H = 3, D = 16; float @base = 10000f;
        var tg = new TG(); var x = tg.InF("x", TG.T, H, D); var pos = tg.In("pos", DataType.I32, TG.T, axes);
        var attrs = Attributes.From(("dims", dims), ("base", @base), ("style", style), ("interleaved", interleaved), ("freq_scale", freqScale))
            .With("sections", sections ?? [0, 0, 0, 0]);
        tg.B.Output("y", tg.B.Emit1("Rope", [x, pos], attrs));
        Diff.Check("Rope", tg.B.Build(), new Step(T, 0, (x, Rnd.F(T * H * D, 1)), (pos, Rnd.I(T * axes, 5, 300))), atol: 1e-4, rtol: 1e-4, what: $"{style} dims={dims} axes={axes} il={interleaved}");
    }

    [Fact]
    public void FrequencyFactors()
    {
        Gpu.Require();
        int T = 5, H = 2, D = 16; float @base = 500000f;
        var tg = new TG(); var x = tg.InF("x", TG.T, H, D); var pos = tg.In("pos", DataType.I32, TG.T, 1); var ff = tg.InF("ff", D / 2);
        tg.B.Output("y", tg.B.Emit1("Rope", [x, pos, ff], Attributes.From(("dims", D), ("base", @base), ("style", "neox"))));
        var factors = Sharplight.Models.RopeScaling.Llama3(D, @base, 32f, 1f, 4f, 8192).Factors;
        Diff.Check("Rope.ff", tg.B.Build(), new Step(T, 0, (x, Rnd.F(T * H * D, 1)), (pos, Enumerable.Range(100, T).ToArray()), (ff, factors)), atol: 1e-4, rtol: 1e-4);
    }

    [Fact]
    public void OnStridedSlice()
    {
        Gpu.Require();
        int T = 4, H = 2, D = 8;
        var tg = new TG(); var x = tg.InF("x", TG.T, H * 2 * D); var pos = tg.In("pos", DataType.I32, TG.T, 1);
        var q = tg.B.Slice(tg.B.Reshape(x, 0, H, 2 * D), 2, 0, D);
        tg.B.Output("y", tg.B.Emit1("Rope", [q, pos], Attributes.From(("dims", D), ("base", 1000f), ("style", "neox"))));
        Diff.Check("Rope.strided", tg.B.Build(), new Step(T, 0, (x, Rnd.F(T * H * 2 * D, 3)), (pos, Enumerable.Range(10, T).ToArray())), atol: 1e-5, rtol: 1e-4);
    }
}

public class GatherTests
{
    [Theory]
    [InlineData("f32", 128)] [InlineData("f16", 128)] [InlineData("bf16", 64)] [InlineData("q8_0", 128)] [InlineData("q4_0", 96)] [InlineData("q4_k", 256)] [InlineData("q5_k", 512)] [InlineData("q6_k", 256)]
    public void Table(string fmt, int d)
    {
        Gpu.Require();
        int V = 50;
        var tg = new TG(); var ids = tg.In("ids", DataType.I32, TG.T);
        var table = tg.RandW("emb", DataType.Get(fmt), 1f, V, d);
        tg.B.Output("y", tg.B.Gather(table, ids));
        Diff.Check("Gather." + fmt, tg.B.Build(), new Step(9, 0, (ids, Rnd.I(9, 4, V))), atol: 1e-6, rtol: 1e-6);
    }

    [Fact]
    public void ActivationTableWithSlicedRows()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 32); var rows = tg.In("rows", DataType.I32, Dim.Sym("R"));
        var s = tg.B.Slice(x, 1, 8, 16);                       // rows have stride 32, 16 wide
        tg.B.Output("y", tg.B.Gather(s, rows));
        var g = tg.B.Build();
        Diff.Check("Gather.view", g, new Step(6, 0, (x, Rnd.F(6 * 32, 1)), (rows, new[] { 5, 1, 3 })) { R = 3 }, atol: 1e-6);
    }
}

public class StatefulTests
{
    [Theory]
    [InlineData("f16", 16, 16, 6, 2)] [InlineData("f32", 16, 16, 6, 2)] [InlineData("f16", 24, 32, 4, 4)] [InlineData("f32", 64, 64, 8, 1)] [InlineData("f16", 128, 128, 4, 2)]
    public void CachedAttentionAcrossSteps(string cacheType, int D, int Dv, int Hq, int Hkv)
        => CachedAttention(cacheType, D, Dv, Hq, Hkv, [5, 3, 1, 1, 7, 1], 40);

    // Long prefill chunks: several key blocks, partial query blocks, GQA (the tensor-core flash-attention path when available).
    [Theory]
    [InlineData("f16", 128, 128, 8, 2)] [InlineData("f16", 256, 256, 4, 1)] [InlineData("f16", 64, 128, 6, 3)] [InlineData("f32", 128, 128, 4, 2)]
    public void CachedAttentionLongPrefill(string cacheType, int D, int Dv, int Hq, int Hkv)
        => CachedAttention(cacheType, D, Dv, Hq, Hkv, [70, 33, 1, 5], 120);

    static void CachedAttention(string cacheType, int D, int Dv, int Hq, int Hkv, int[] chunks, int ctx)
    {
        Gpu.Require();
        float scale = 1f / MathF.Sqrt(D);
        int total = chunks.Sum();
        var tg = new TG(); var T = TG.T;
        var qi = tg.InF("q", T, Hq, D); var ki = tg.InF("k", T, Hkv, D); var vi = tg.InF("v", T, Hkv, Dv);
        var ct = DataType.Get(cacheType);
        var kc = tg.State("kc", ct, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D); var vc = tg.State("vc", ct, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, Dv);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
        tg.B.Output("o", tg.B.CachedAttention(qi, kc, vc, scale));
        var q = Rnd.F(total * Hq * D, 1); var k = Rnd.F(total * Hkv * D, 2); var v = Rnd.F(total * Hkv * Dv, 3);
        var steps = new List<Step>(); int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]), (vi, v[(past * Hkv * Dv)..((past + c) * Hkv * Dv)])));
            past += c;
        }
        // f16 caches may run on tensor cores with f16 queries and probabilities: allow for their rounding.
        bool f16 = cacheType == "f16";
        Diff.Check("CachedAttention", tg.B.Build(), steps, atol: f16 ? 3e-4 : 3e-5, rtol: f16 ? 3e-3 : 2e-4, maxTokens: chunks.Max(), ctx: ctx, what: $"{cacheType} D={D} Hq={Hq} Hkv={Hkv}");
    }

    [Theory]
    [InlineData(4, 0f)] [InlineData(0, 20f)] [InlineData(5, 30f)] [InlineData(150, 0f)]
    public void CachedAttentionWindowAndSoftcap(int window, float softcap)
    {
        Gpu.Require();
        int Hq = 4, Hkv = 2, D = 32, ctx = 256; float scale = 1f / MathF.Sqrt(D);
        int[] chunks = [6, 3, 1, 1, 180, 1];
        int total = chunks.Sum();
        var tg = new TG(); var T = TG.T;
        var qi = tg.InF("q", T, Hq, D); var ki = tg.InF("k", T, Hkv, D); var vi = tg.InF("v", T, Hkv, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D); var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, ki); tg.B.CacheWrite(vc, vi);
        var attrs = new List<(string, object)> { ("scale", scale), ("offset", "n_past") };
        if (window > 0) attrs.Add(("window", window));
        if (softcap > 0) attrs.Add(("softcap", softcap));
        tg.B.RuntimeScalar("n_past");
        tg.B.Output("o", tg.B.Emit1("CachedAttention", [qi, kc, vc], Attributes.From([.. attrs])));
        var q = Rnd.F(total * Hq * D, 1).Select(x => x * 4).ToArray(); var k = Rnd.F(total * Hkv * D, 2); var v = Rnd.F(total * Hkv * D, 3);
        var steps = new List<Step>(); int past = 0;
        foreach (int c in chunks)
        {
            steps.Add(new Step(c, past, (qi, q[(past * Hq * D)..((past + c) * Hq * D)]), (ki, k[(past * Hkv * D)..((past + c) * Hkv * D)]), (vi, v[(past * Hkv * D)..((past + c) * Hkv * D)])));
            past += c;
        }
        // f16 caches may run on the tensor-core flash kernel (f16 Q and P).
        Diff.Check("CachedAttention.window", tg.B.Build(), steps, atol: 1e-3, rtol: 3e-3, maxTokens: 192, ctx: ctx, what: $"window={window} softcap={softcap}");
    }

    [Fact]
    public void CachedAttentionWithStridedQueryAndKv()
    {
        Gpu.Require();
        int Hq = 4, Hkv = 2, D = 16, ctx = 32;
        var tg = new TG(); var T = TG.T;
        var qf = tg.InF("qf", T, Hq, 2 * D);                      // q and gate interleaved per head (Qwen3.5 style)
        var q = tg.B.Slice(qf, 2, 0, D);
        var kf = tg.InF("kf", T, Hkv, 2 * D);
        var k = tg.B.Slice(kf, 2, D, D);
        var vf = tg.InF("vf", T, Hkv, D);
        var kc = tg.State("kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D); var vc = tg.State("vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
        tg.B.CacheWrite(kc, k); tg.B.CacheWrite(vc, vf);
        tg.B.Output("o", tg.B.CachedAttention(q, kc, vc, 0.25f));
        int[] chunks = [4, 1, 2]; int[] pasts = [0, 4, 5];
        var steps = chunks.Select((c, i) => new Step(c, pasts[i], (qf, Rnd.F(c * Hq * 2 * D, 10 + i)), (kf, Rnd.F(c * Hkv * 2 * D, 20 + i)), (vf, Rnd.F(c * Hkv * D, 30 + i)))).ToList();
        Diff.Check("CachedAttention.strided", tg.B.Build(), steps, atol: 3e-5, rtol: 2e-4, maxTokens: 8, ctx: ctx);
    }

    [Theory]
    [InlineData(20, 4, true)] [InlineData(20, 4, false)] [InlineData(600, 4, true)] [InlineData(33, 2, true)] [InlineData(16, 6, false)]
    public void CausalConvAcrossSteps(int C, int K, bool silu)
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, C);
        var w = tg.RandW("w", DataType.F32, 1f, C, K);
        var state = tg.State("cs", DataType.F32, StateKind.Recurrent, K - 1, C);
        tg.B.Output("y", tg.B.Emit("CausalConv1d", [x, w, state], Attributes.From(("activation", silu ? "silu" : "none")), writes: [state])[0]);
        int[] chunks = [3, 1, 1, 5, 2];
        var steps = chunks.Select((c, i) => new Step(c, 0, (x, Rnd.F((long)c * C, 100 + i)))).ToList();
        Diff.Check("CausalConv1d", tg.B.Build(), steps, atol: 2e-5, rtol: 1e-4, maxTokens: 8, what: $"C={C} K={K} silu={silu}");
    }

    [Theory]
    [InlineData(2, 4, 16, 8)] [InlineData(1, 3, 128, 128)] [InlineData(4, 4, 32, 64)] [InlineData(2, 2, 64, 32)]
    public void GatedDeltaRuleAcrossSteps(int Hk, int Hv, int Dk, int Dv)
    {
        Gpu.Require();
        var tg = new TG(); var D = TG.T;
        var qi = tg.InF("q", D, Hk, Dk); var ki = tg.InF("k", D, Hk, Dk); var vi = tg.InF("v", D, Hv, Dv); var gi = tg.InF("g", D, Hv); var bi = tg.InF("b", D, Hv);
        var st = tg.State("ssm", DataType.F32, StateKind.Recurrent, Hv, Dv, Dk);
        tg.B.Output("o", tg.B.Emit("GatedDeltaRule", [qi, ki, vi, gi, bi, st], Attributes.From(("scale", 1f / MathF.Sqrt(Dk))), writes: [st])[0]);
        int[] chunks = [3, 1, 2, 1];
        var steps = chunks.Select((c, i) => new Step(c, 0,
            (qi, Rnd.F(c * Hk * Dk, 1 + i, 1f)), (ki, Rnd.F(c * Hk * Dk, 11 + i, 0.3f)), (vi, Rnd.F(c * Hv * Dv, 21 + i)),
            (gi, Rnd.F(c * Hv, 31 + i).Select(v => -Math.Abs(v)).ToArray()), (bi, Rnd.F(c * Hv, 41 + i).Select(Math.Abs).ToArray()))).ToList();
        Diff.Check("GatedDeltaRule", tg.B.Build(), steps, atol: 2e-4, rtol: 2e-3, maxTokens: 8, what: $"Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}");
    }
}

public class ViewTests
{
    [Fact]
    public void SliceConcatAndContiguous()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 4, 10);
        var a = tg.B.Slice(x, 2, 0, 4); var b = tg.B.Slice(x, 2, 4, 6);
        tg.B.Output("cat", tg.B.Concat(2, b, a));
        tg.B.Output("cat1", tg.B.Concat(1, tg.B.Slice(x, 1, 0, 1), tg.B.Slice(x, 1, 2, 2)));
        tg.B.Output("contig", tg.B.Contiguous(tg.B.Permute(x, 0, 2, 1)));
        Diff.Check("Views.concat", tg.B.Build(), new Step(5, 0, (x, Rnd.F(5 * 40, 1))), atol: 0, rtol: 0);
    }

    [Fact]
    public void ReshapeOfNonContiguousViewFallsBackToCopy()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 6, 8);
        var p = tg.B.Permute(x, 0, 2, 1);              // [T,8,6] strided
        var r = tg.B.Reshape(p, 0, 48);                // cannot be a view of the permuted strides
        tg.B.Output("y", tg.B.Scale(r, 2f));
        var s = tg.B.Slice(x, 2, 2, 4);                // [T,6,4], row stride 8
        tg.B.Output("z", tg.B.Add(tg.B.Reshape(s, 0, 24), tg.B.Reshape(s, 0, 24)));
        Diff.Check("Views.reshape", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 48, 1))), atol: 1e-6);
    }

    [Fact]
    public void PermuteHeadsThenMatMulAndBack()
    {
        Gpu.Require();
        int H = 4, D = 16;
        var tg = new TG(); var x = tg.InF("x", TG.T, H * D);
        var heads = tg.B.Reshape(x, 0, H, D);
        var w = tg.RandW("w", DataType.F32, 0.5f, 8, D);
        tg.B.Output("y", tg.B.MatMul(heads, w));
        tg.B.Output("sum", tg.B.Add(tg.B.Slice(x, 1, 0, 16), tg.B.Slice(x, 1, 16, 16)));
        Diff.Check("Views.heads", tg.B.Build(), new Step(7, 0, (x, Rnd.F(7 * H * D, 1))), atol: 2e-2, rtol: 1e-2);
    }

    [Fact]
    public void GraphOutputThatIsAView()
    {
        Gpu.Require();
        var tg = new TG(); var x = tg.InF("x", TG.T, 4, 10);
        tg.B.Output("v", tg.B.Permute(tg.B.Slice(x, 2, 2, 5), 0, 2, 1));
        Diff.Check("Views.output", tg.B.Build(), new Step(3, 0, (x, Rnd.F(3 * 40, 1))), atol: 0, rtol: 0);
    }
}

[CollectionDefinition("WorkgroupMatrixGemm", DisableParallelization = true)]
public class WorkgroupMatrixGemmCollection;

/// <summary>The opt-in workgroup-scope matrix GEMM (DeviceFeatures.MatrixWorkgroup, VK_NV_cooperative_matrix2).</summary>
[Collection("WorkgroupMatrixGemm")]
public class WorkgroupMatrixGemmTests
{
    [Theory]
    [InlineData("f16", 512, 96)] [InlineData("q8_0", 512, 300)] [InlineData("q8_0", 192, 130)] [InlineData("f16", 4096, 64)] [InlineData("q8_0", 4096, 520)]
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
                var tg = new TG(); var x = tg.InF("x", TG.T, K);
                var w = tg.RandW("w", DataType.Get(fmt), 1f, N, K);
                tg.B.Output("y", tg.B.MatMul(x, w));
                Diff.Check("MatMul.nv2." + fmt, tg.B.Build(), new Step(t, 0, (x, Rnd.F((long)t * K, 3))), atol: 1.5e-1, rtol: 1e-2,
                    maxTokens: 300, what: $"T={t} K={K} N={N}");
            }
        }
        finally { Shaders.Lowering.GraphLowerer.PreferWorkgroupMatrixGemm = old; }
    }
}
