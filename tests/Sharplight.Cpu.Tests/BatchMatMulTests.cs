using Sharplight.Graph;

namespace Sharplight.Cpu.Tests;

public class BatchMatMulTests
{
    public static IEnumerable<object[]> Cases()
    {
        (long[] a, long[] b, bool ta, bool tb, float alpha)[] c =
        [
            ([2, 3, 5, 7], [2, 3, 7, 4], false, false, 1f),
            ([2, 3, 5, 7], [2, 3, 4, 7], false, true, 0.125f),
            ([2, 3, 7, 5], [2, 3, 7, 4], true, false, 1f),
            ([2, 3, 7, 5], [2, 3, 4, 7], true, true, 2f),
            ([4, 1, 6, 9], [1, 3, 9, 5], false, false, 1f),
            ([3, 6, 9], [9, 5], false, false, 1f),
            ([6, 9], [3, 9, 5], false, false, 1f),
            ([4, 1, 6, 9], [3, 5, 9], false, true, 1f),
            ([2, 5, 7], [2, 7, 3], false, false, 1f),
            ([1, 70, 300], [1, 300, 75], false, false, 1f),     // crosses KC and tile edges
            ([2, 70, 300], [2, 75, 300], false, true, 0.5f),
            ([5, 33], [33, 40], false, false, 1f),
            ([1, 1, 1, 64], [1, 1, 64, 130], false, false, 1f),  // M=1 (decode-like)
            ([2, 12, 197, 64], [2, 12, 64, 197], false, false, 0.125f), // attention-like
        ];
        foreach (var t in c)
        {
            yield return [t.a, t.b, t.ta, t.tb, t.alpha];
        }
    }

    [Theory, MemberData(nameof(Cases))]
    public void Bmm(long[] ad, long[] bd, bool ta, bool tb, float alpha)
    {
        var a = Kit.Rand((int)SRef.Prod(ad), 51);
        var b = Kit.Rand((int)SRef.Prod(bd), 52);
        var tg = new TestGraph();
        var ain = tg.InF("a", SpatialKit.D(ad));
        var bin = tg.InF("b", SpatialKit.D(bd));
        tg.B.Output("y", tg.B.Emit1("BatchMatMul", [ain, bin], Attributes.From(("trans_a", ta), ("trans_b", tb), ("alpha", alpha))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, ain, a);
        Kit.Set(m, bin, b);
        var p = Kit.P(1);
        m.Run(p);
        var exp = SRef.Bmm(a, ad, b, bd, ta, tb, alpha, out var od);
        Assert.Equal(od, m.Graph.Outputs["y"].Shape.Dims.Select(d => d.Fixed).ToArray());
        Kit.AssertClose(exp, Kit.Out(m, "y", p), 1e-4, 1e-4);
    }

    /// <summary>Attention-style on permuted (strided) views: q [B,T,H,D] -> [B,H,T,D]; k [B,S,H,D] -> [B,H,D,S].</summary>
    [Fact]
    public void AttentionOnStridedViews()
    {
        int B = 2;
        int T = 13;
        int S = 21;
        int H = 3;
        int D = 17;
        var q = Kit.Rand(B * T * H * D, 53);
        var k = Kit.Rand(B * S * H * D, 54);
        var tg = new TestGraph();
        var qin = tg.InF("q", SpatialKit.D(B, T, H, D));
        var kin = tg.InF("k", SpatialKit.D(B, S, H, D));
        var qp = tg.B.Permute(qin, 0, 2, 1, 3);
        var kp = tg.B.Permute(kin, 0, 2, 3, 1);
        tg.B.Output("s", tg.B.Emit1("BatchMatMul", [qp, kp], Attributes.From(("alpha", 0.25f))));
        // transposed-flag variant of the same product: k permuted to [B,H,S,D] with trans_b
        var kp2 = tg.B.Permute(kin, 0, 2, 1, 3);
        tg.B.Output("s2", tg.B.Emit1("BatchMatMul", [qp, kp2], Attributes.From(("trans_b", true), ("alpha", 0.25f))));
        using var m = Kit.Compile(tg.B.Build());
        Kit.Set(m, qin, q);
        Kit.Set(m, kin, k);
        var p = Kit.P(1);
        m.Run(p);
        var ql = new float[B * H * T * D];
        var kl = new float[B * H * D * S];
        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int t = 0; t < T; t++)
                {
                    for (int d = 0; d < D; d++)
                    {
                        ql[((b * H + h) * T + t) * D + d] = q[((b * T + t) * H + h) * D + d];
                    }
                }
            }
        }

        for (int b = 0; b < B; b++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int s = 0; s < S; s++)
                {
                    for (int d = 0; d < D; d++)
                    {
                        kl[((b * H + h) * D + d) * S + s] = k[((b * S + s) * H + h) * D + d];
                    }
                }
            }
        }

        var exp = SRef.Bmm(ql, [B, H, T, D], kl, [B, H, D, S], false, false, 0.25f, out _);
        Kit.AssertClose(exp, Kit.Out(m, "s", p), 1e-4, 1e-4);
        Kit.AssertClose(exp, Kit.Out(m, "s2", p), 1e-4, 1e-4);
    }

    [Fact]
    public void SymbolicM()
    {
        int T = 5;
        int K = 19;
        int N = 23;
        var a = Kit.Rand(T * K, 55);
        var b = Kit.Rand(K * N, 56);
        var tg = new TestGraph();
        var ain = tg.InF("a", SpatialKit.DSym(K));
        var bin = tg.InF("b", SpatialKit.D(K, N));
        tg.B.Output("y", tg.B.Emit1("BatchMatMul", [ain, bin]));
        using var m = Kit.Compile(tg.B.Build(), maxTokens: 8);
        Kit.Set(m, ain, a);
        Kit.Set(m, bin, b);
        var p = Kit.P(T);
        m.Run(p);
        Kit.AssertClose(SRef.Bmm(a, [T, K], b, [K, N], false, false, 1f, out _), Kit.Out(m, "y", p), 1e-4, 1e-4);
    }
}
