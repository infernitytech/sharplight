using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Cross-checks against outputs recorded from llama.cpp's ggml CPU backend (rope, ssm_conv, gated_delta_net).</summary>
public class GgmlCrossCheckTests
{
    private static float[] Floats(BinaryReader r, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = r.ReadSingle();
        }
        return a;
    }
    private static int[] Ints(BinaryReader r, int n)
    {
        var a = new int[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = r.ReadInt32();
        }
        return a;
    }

    private record RopeRec(int Mode, int Dims, int[] Sec, float FreqScale, int T, int H, int D, float[] X, int[] Pos, float[] Out);
    private record DeltaRec(int T, int Hk, int Hv, int S, float[] Q, float[] K, float[] V, float[] G, float[] B, float[] State, float[] Out);
    private record ConvRec(int T, int C, int K, float[] State, float[] X, float[] W, float[] Out);

    private static (List<RopeRec> Rope, List<DeltaRec> Delta, List<ConvRec> Conv) Load()
    {
        var r = new BinaryReader(File.OpenRead(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml_ops.bin")));
        r.ReadInt32();
        List<RopeRec> rope = [];
        List<DeltaRec> delta = [];
        List<ConvRec> conv = [];
        while (r.BaseStream.Position < r.BaseStream.Length)
        {
            int tag = r.ReadInt32();
            if (tag == 1)
            {
                int mode = r.ReadInt32();
                int dims = r.ReadInt32();
                var sec = Ints(r, 4);
                float fs = r.ReadSingle();
                int T = r.ReadInt32();
                int H = r.ReadInt32();
                int D = r.ReadInt32();
                rope.Add(new(mode, dims, sec, fs, T, H, D, Floats(r, T * H * D), Ints(r, T * 4), Floats(r, T * H * D)));
            }
            else if (tag == 2)
            {
                int T = r.ReadInt32();
                int Hk = r.ReadInt32();
                int Hv = r.ReadInt32();
                int S = r.ReadInt32();
                delta.Add(new(T, Hk, Hv, S, Floats(r, S * Hk * T), Floats(r, S * Hk * T), Floats(r, S * Hv * T), Floats(r, Hv * T), Floats(r, Hv * T),
                    Floats(r, S * S * Hv), Floats(r, S * Hv * T + S * S * Hv)));
            }
            else
            {
                int T = r.ReadInt32();
                int C = r.ReadInt32();
                int K = r.ReadInt32();
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
            var tg = new TestGraph();
            var xin = tg.InF("x", TestGraph.T, c.H, c.D);
            var pin = tg.In("pos", DataType.I32, TestGraph.T, axes);
            var attrs = Attributes.From(("dims", c.Dims), ("base", 10000f), ("style", style), ("interleaved", c.Mode == 40), ("freq_scale", c.FreqScale)).With("sections", multi ? c.Sec : new int[4]);
            tg.B.Output("y", tg.B.Emit1("Rope", [xin, pin], attrs));
            using var m = Kit.Compile(tg.B.Build());
            Kit.Set(m, xin, c.X);
            Kit.Set(m, pin, posIn);
            var p = Kit.P(c.T);
            m.Run(p);
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
            var tg = new TestGraph();
            var xin = tg.InF("x", TestGraph.T, c.C);
            var wv = tg.WF("w", c.W, c.C, c.K);
            var state = tg.State("cs", DataType.F32, StateKind.Recurrent, c.K - 1, c.C);
            tg.B.Output("y", tg.B.Emit("CausalConv1d", [xin, wv, state], Attributes.From(("activation", "none")), writes: [state])[0]);
            using var m = Kit.Compile(tg.B.Build());
            ((CpuCompiledSegment)m.Segments[0]).TestSetState(state, c.State);
            Kit.Set(m, xin, c.X);
            var p = Kit.P(c.T);
            m.Run(p);
            Kit.AssertClose(c.Out, Kit.Out(m, "y", p), 1e-4, 1e-4, "ssm_conv");
        }
    }

    internal static float[] RunDelta(float[] q, float[] k, float[] v, float[] g, float[] beta, float[] state0, int T, int Hk, int Hv, int Dk, int Dv, float scale, int[] chunks, out float[] finalState,
        DataType? stateType = null)
    {
        var tg = new TestGraph();
        var D = TestGraph.T;
        var qi = tg.InF("q", D, Hk, Dk);
        var ki = tg.InF("k", D, Hk, Dk);
        var vi = tg.InF("v", D, Hv, Dv);
        var gi = tg.InF("g", D, Hv);
        var bi = tg.InF("b", D, Hv);
        var st = tg.State("ssm", stateType ?? DataType.F32, StateKind.Recurrent, Hv, Dv, Dk);
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
            Kit.Set(m, qi, q[(t0 * Hk * Dk)..((t0 + c) * Hk * Dk)]);
            Kit.Set(m, ki, k[(t0 * Hk * Dk)..((t0 + c) * Hk * Dk)]);
            Kit.Set(m, vi, v[(t0 * Hv * Dv)..((t0 + c) * Hv * Dv)]);
            Kit.Set(m, gi, g[(t0 * Hv)..((t0 + c) * Hv)]);
            Kit.Set(m, bi, beta[(t0 * Hv)..((t0 + c) * Hv)]);
            var p = Kit.P(c);
            m.Run(p);
            Kit.Out(m, "o", p).CopyTo(o, t0 * Hv * Dv);
            t0 += c;
        }
        finalState = ((CpuCompiledSegment)m.Segments[0]).TestGetState(st);
        return o;
    }
}
