using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// A tiny 2-layer "Qwen3.5-like" model (full attention layer + gated-delta linear layer, SwiGLU MLPs, mixed weight types)
/// built with GraphBuilder, compiled through CompiledModel with the CPU device, compared with a scalar reference.
/// </summary>
public class EndToEndTests
{
    private const int V = 50;
    private const int E = 32;
    private const int H = 4;
    private const int Hkv = 2;
    private const int D = 8;
    private const int FF = 64;
    private const int Hk = 2;
    private const int Dk = 8;
    private const int Hv = 4;
    private const int Dv = 8;
    private const int ConvK = 4;
    private const float Eps = 1e-6f;
    private static readonly int[] Sections = [2, 1, 1, 0];

    private sealed class Model
    {
        public readonly Dictionary<string, float[]> Deq = [];     // dequantized values for the reference
        public TestGraph G = new("tiny");
        public Value W(string name, string type, int n, int k, int seed, float scale = 0.3f)
        {
            var w = Kit.Rand(n * k, seed, scale);
            byte[] data;
            float[] deq;
            var dt = DataType.Get(type);
            switch (type)
            {
                case "f32":
                    data = MemoryMarshal.AsBytes(w.AsSpan()).ToArray();
                    deq = w;
                    break;
                case "f16":
                    data = new byte[n * k * 2];
                    deq = new float[n * k];
                    for (int i = 0; i < w.Length; i++)
                    {
                        var h = (Half)w[i];
                        BitConverter.TryWriteBytes(data.AsSpan(i * 2), h);
                        deq[i] = (float)h;
                    }
                    break;
                case "q8_0":
                    data = Quantizers.Q8_0(w);
                    deq = Quantizers.DeqQ8_0(data);
                    break;
                case "q4_0":
                    data = Quantizers.Q4_0(w);
                    deq = Quantizers.DeqQ4_0(data);
                    break;
                default:
                    throw new NotSupportedException(type);
            }
            Deq[name] = deq;
            return G.W(name, dt, k == 0 ? new Shape(n) : new Shape(n, k), data);
        }
        public Value Vec(string name, int n, int seed, float scale = 1f, float offset = 0f)
        {
            var w = Kit.Rand(n, seed, scale).Select(x => x + offset).ToArray();
            Deq[name] = w;
            return G.WF(name, w, n);
        }
    }

    private (Model m, Value tokens, Value pos, Value rows, ComputeGraph graph) Build()
    {
        var m = new Model();
        var g = m.G;
        var b = g.B;
        var T = TestGraph.T;
        var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
        var pos = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 4));
        var rows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        var embd = m.W("token_embd.weight", "q8_0", V, E, 1);
        Value x;
        using (b.Group("input"))
        {
            x = b.Gather(embd, tokens, "inp_embd");
        }

        // ---- layer 0: full attention ----
        using (b.Group("blk.0"))
        {
            var residual = x;
            var h = b.RmsNorm(x, m.Vec("l0.attn_norm", E, 10, 0.5f, 1f), Eps);
            var qFull = b.Reshape(b.MatMul(h, m.W("l0.q", "q8_0", 2 * H * D, E, 11)), 0, H, 2 * D);
            var q = b.Slice(qFull, 2, 0, D);
            var gate = b.Slice(qFull, 2, D, D);
            q = b.RmsNorm(q, m.Vec("l0.q_norm", D, 12, 0.5f, 1f), Eps);
            var k = b.Reshape(b.MatMul(h, m.W("l0.k", "f32", Hkv * D, E, 13)), 0, Hkv, D);
            k = b.RmsNorm(k, m.Vec("l0.k_norm", D, 14, 0.5f, 1f), Eps);
            var v = b.Reshape(b.MatMul(h, m.W("l0.v", "f16", Hkv * D, E, 15)), 0, Hkv, D);
            var rope = Attributes.From(("dims", D), ("base", 10000f), ("style", "neox"), ("interleaved", true)).With("sections", Sections);
            q = b.Emit1("Rope", [q, pos], rope);
            k = b.Emit1("Rope", [k, pos], rope);
            var kc = g.State("l0.kc", DataType.F16, StateKind.KeyCache, Dim.Sym("ctx"), Hkv, D);
            var vc = g.State("l0.vc", DataType.F16, StateKind.ValueCache, Dim.Sym("ctx"), Hkv, D);
            b.CacheWrite(kc, k);
            b.CacheWrite(vc, v);
            var attn = b.CachedAttention(q, kc, vc, 1f / MathF.Sqrt(D));
            attn = b.Mul(attn, b.Sigmoid(gate));
            var o = b.MatMul(b.Reshape(attn, 0, H * D), m.W("l0.o", "q4_0", E, H * D, 16));
            x = b.Add(o, residual);
            x = Mlp(m, b, x, "l0", 20);
        }

        // ---- layer 1: gated delta net ----
        using (b.Group("blk.1"))
        {
            var residual = x;
            var h = b.RmsNorm(x, m.Vec("l1.attn_norm", E, 30, 0.5f, 1f), Eps);
            int qkvDim = 2 * Hk * Dk + Hv * Dv;
            var qkv = b.MatMul(h, m.W("l1.qkv", "q8_0", qkvDim, E, 31));
            var z = b.MatMul(h, m.W("l1.zgate", "q8_0", Hv * Dv, E, 32));
            var beta = b.Sigmoid(b.MatMul(h, m.W("l1.beta", "f32", Hv, E, 33)));
            var alpha = b.MatMul(h, m.W("l1.alpha", "f32", Hv, E, 34));
            var gdec = b.Mul(b.Softplus(b.Add(alpha, m.Vec("l1.dt_bias", Hv, 35))), m.Vec("l1.ssm_a", Hv, 36, 0.5f, -1f));
            var convState = g.State("l1.conv", DataType.F32, StateKind.Recurrent, ConvK - 1, qkvDim);
            var ssmState = g.State("l1.ssm", DataType.F32, StateKind.Recurrent, Hv, Dv, Dk);
            var conv = b.Emit("CausalConv1d", [qkv, m.W("l1.conv1d", "f32", qkvDim, ConvK, 37), convState], Attributes.From(("activation", "silu")), writes: [convState])[0];
            var q = b.L2Norm(b.Reshape(b.Slice(conv, 1, 0, Hk * Dk), 0, Hk, Dk), Eps);
            var k = b.L2Norm(b.Reshape(b.Slice(conv, 1, Hk * Dk, Hk * Dk), 0, Hk, Dk), Eps);
            var v = b.Reshape(b.Slice(conv, 1, 2 * Hk * Dk, Hv * Dv), 0, Hv, Dv);
            var dr = b.Emit("GatedDeltaRule", [q, k, v, gdec, beta, ssmState], Attributes.From(("scale", 1f / MathF.Sqrt(Dk))), writes: [ssmState])[0];
            var on = b.RmsNorm(dr, m.Vec("l1.ssm_norm", Dv, 38, 0.5f, 1f), Eps);
            on = b.Mul(on, b.Silu(b.Reshape(z, 0, Hv, Dv)));
            var o = b.MatMul(b.Reshape(on, 0, Hv * Dv), m.W("l1.out", "f16", E, Hv * Dv, 39));
            x = b.Add(o, residual);
            x = Mlp(m, b, x, "l1", 40);
        }

        using (b.Group("output"))
        {
            var hidden = b.RmsNorm(x, m.Vec("output_norm", E, 50, 0.5f, 1f), Eps);
            var sel = b.Gather(hidden, rows);
            b.Output("logits", b.MatMul(sel, embd));
            b.Output("hidden", hidden);
        }
        return (m, tokens, pos, rows, b.Build());
    }

    private static Value Mlp(Model m, GraphBuilder b, Value x, string p, int seed)
    {
        var residual = x;
        var h = b.RmsNorm(x, m.Vec(p + ".ffn_norm", E, seed, 0.5f, 1f), Eps);
        var gate = b.MatMul(h, m.W(p + ".gate", "q8_0", FF, E, seed + 1));
        var up = b.MatMul(h, m.W(p + ".up", "q8_0", FF, E, seed + 2));
        var down = b.MatMul(b.Mul(b.Silu(gate), up), m.W(p + ".down", "f32", E, FF, seed + 3));
        return b.Add(down, residual);
    }

    // ---------------------------------------------------------------- scalar reference
    private static float[] RefForward(Model m, int[] tokens, int[] positions /* [T,4] */, out float[] hiddenOut)
    {
        int T = tokens.Length;
        var W = m.Deq;
        var x = new float[T * E];
        for (int t = 0; t < T; t++)
        {
            Array.Copy(W["token_embd.weight"], tokens[t] * E, x, t * E, E);
        }

        Func<float[], float[], float[]> add = (a, c) => a.Select((v, i) => v + c[i]).ToArray();

        float[] Mlp(float[] x, string p)
        {
            var h = Ref.RmsNorm(x, T, E, W[p + ".ffn_norm"], Eps);
            var gate = Ref.MatMul(h, T, E, W[p + ".gate"], FF);
            var up = Ref.MatMul(h, T, E, W[p + ".up"], FF);
            var act = gate.Select((v, i) => Ref.Silu(v) * up[i]).ToArray();
            return add(Ref.MatMul(act, T, FF, W[p + ".down"], E), x);
        }

        // layer 0
        {
            var h = Ref.RmsNorm(x, T, E, W["l0.attn_norm"], Eps);
            var qFull = Ref.MatMul(h, T, E, W["l0.q"], 2 * H * D);
            var q = new float[T * H * D];
            var gate = new float[T * H * D];
            for (int r = 0; r < T * H; r++)
            {
                for (int i = 0; i < D; i++)
                {
                    q[r * D + i] = qFull[r * 2 * D + i];
                    gate[r * D + i] = qFull[r * 2 * D + D + i];
                }
            }

            q = Ref.RmsNorm(q, T * H, D, W["l0.q_norm"], Eps);
            var k = Ref.RmsNorm(Ref.MatMul(h, T, E, W["l0.k"], Hkv * D), T * Hkv, D, W["l0.k_norm"], Eps);
            var v = Ref.MatMul(h, T, E, W["l0.v"], Hkv * D);
            q = Ref.Rope(q, T, H, D, positions, 4, D, 10000f, "neox", Sections, true, 1f);
            k = Ref.Rope(k, T, Hkv, D, positions, 4, D, 10000f, "neox", Sections, true, 1f);
            var kr = k.Select(f => (float)(Half)f).ToArray();
            var vr = v.Select(f => (float)(Half)f).ToArray();
            var attn = Ref.Attention(q, T, H, D, kr, vr, T, Hkv, D, 1f / MathF.Sqrt(D), 0, true);
            for (int i = 0; i < attn.Length; i++)
            {
                attn[i] *= Ref.Sigmoid(gate[i]);
            }

            x = Mlp(add(Ref.MatMul(attn, T, H * D, W["l0.o"], E), x), "l0");
        }
        // layer 1
        {
            var h = Ref.RmsNorm(x, T, E, W["l1.attn_norm"], Eps);
            int qkvDim = 2 * Hk * Dk + Hv * Dv;
            var qkv = Ref.MatMul(h, T, E, W["l1.qkv"], qkvDim);
            var z = Ref.MatMul(h, T, E, W["l1.zgate"], Hv * Dv);
            var beta = Ref.MatMul(h, T, E, W["l1.beta"], Hv).Select(Ref.Sigmoid).ToArray();
            var alpha = Ref.MatMul(h, T, E, W["l1.alpha"], Hv);
            var g = new float[T * Hv];
            for (int t = 0; t < T; t++)
            {
                for (int i = 0; i < Hv; i++)
                {
                    g[t * Hv + i] = Ref.Softplus(alpha[t * Hv + i] + W["l1.dt_bias"][i]) * W["l1.ssm_a"][i];
                }
            }

            var conv = Ref.Conv(qkv, T, qkvDim, W["l1.conv1d"], ConvK, new float[(ConvK - 1) * qkvDim], true);
            float[] Cut(int col, int n) { var r = new float[T * n]; for (int t = 0; t < T; t++) { Array.Copy(conv, t * qkvDim + col, r, t * n, n); } return r; }
            var q = Ref.L2Norm(Cut(0, Hk * Dk), T * Hk, Dk, Eps);
            var k = Ref.L2Norm(Cut(Hk * Dk, Hk * Dk), T * Hk, Dk, Eps);
            var v = Cut(2 * Hk * Dk, Hv * Dv);
            var o = Ref.Delta(q, k, v, g, beta, new float[Hv * Dv * Dk], T, Hk, Hv, Dk, Dv, 1f / MathF.Sqrt(Dk));
            var on = Ref.RmsNorm(o, T * Hv, Dv, W["l1.ssm_norm"], Eps);
            for (int i = 0; i < on.Length; i++)
            {
                on[i] *= Ref.Silu(z[i]);
            }

            x = Mlp(add(Ref.MatMul(on, T, Hv * Dv, W["l1.out"], E), x), "l1");
        }
        var hid = Ref.RmsNorm(x, T, E, W["output_norm"], Eps);
        hiddenOut = hid;
        return Ref.MatMul(hid, T, E, W["token_embd.weight"], V);
    }

    [Fact]
    public void TinyModelPrefillThenDecodeMatchesReference()
    {
        var (model, tokensIn, posIn, rowsIn, graph) = Build();
        int[] tokens = [3, 17, 42, 8, 29, 5];
        int total = tokens.Length;
        var positions = new int[total * 4];
        for (int t = 0; t < total; t++)
        {
            for (int a = 0; a < 4; a++)
            {
                positions[t * 4 + a] = t;
            }
        }

        var refLogits = RefForward(model, tokens, positions, out var refHidden);

        using var cm = Kit.Compile(graph, maxTokens: 16, ctx: 32);
        Assert.Single(cm.Segments);
        void Run(int start, int count, int[] rows, out float[] logits, out float[] hidden)
        {
            cm.SetInput(tokensIn, MemoryMarshal.AsBytes(tokens.AsSpan(start, count)));
            cm.SetInput(posIn, MemoryMarshal.AsBytes(positions.AsSpan(start * 4, count * 4)));
            cm.SetInput(rowsIn, MemoryMarshal.AsBytes(rows.AsSpan()));
            var p = Kit.P(count, start, ("R", rows.Length));
            cm.Run(p);
            logits = Kit.Out(cm, "logits", p);
            hidden = Kit.Out(cm, "hidden", p);
        }
        // prefill 5 tokens (ask for logits of the last one), then decode token 6
        Run(0, 5, [4], out var l1, out var h1);
        Kit.AssertClose(refLogits[(4 * V)..(5 * V)], l1, 2e-3, 2e-3, "prefill logits");
        Kit.AssertClose(refHidden[..(5 * E)], h1, 2e-3, 2e-3, "prefill hidden");
        Run(5, 1, [0], out var l2, out var h2);
        Kit.AssertClose(refLogits[(5 * V)..(6 * V)], l2, 2e-3, 2e-3, "decode logits");
        Kit.AssertClose(refHidden[(5 * E)..], h2, 2e-3, 2e-3, "decode hidden");

        // reset and run everything at once, requesting two rows
        cm.ResetState();
        Run(0, 6, [2, 5], out var l3, out _);
        Kit.AssertClose(refLogits[(2 * V)..(3 * V)], l3[..V], 2e-3, 2e-3, "T=6 row 2");
        Kit.AssertClose(refLogits[(5 * V)..], l3[V..], 2e-3, 2e-3, "T=6 row 5");
    }

    [Fact]
    public void SegmentUsesArenaReuseAndReportsMemory()
    {
        var (_, _, _, _, graph) = Build();
        using var cm = Kit.Compile(graph, maxTokens: 64, ctx: 64);
        var seg = (CpuCompiledSegment)cm.Segments[0];
        Assert.True(seg.DeviceMemoryBytes > 0);
        // arena must be much smaller than the sum of all intermediates (liveness packing)
        long naive = graph.Nodes.Where(n => n.Op.Category != OpCategory.View).SelectMany(n => n.Outputs).Sum(o => (long)o.Type.ByteSize(o.Shape.Dims[0].IsSymbolic ? o.Shape.InnerElements * 64 : o.Shape.Elements()));
        Assert.True(seg.ArenaBytes < naive / 2, $"arena {seg.ArenaBytes} vs naive {naive}");
    }

    /// <summary>The CPU device can act as the pipeline fallback: unsupported ops are rejected, supported ones placed.</summary>
    [Fact]
    public void ProviderSurface()
    {
        var prov = new CpuExecutionProvider();
        Assert.Equal("cpu", prov.Name);
        Assert.True(prov.CheckAvailability().IsAvailable);
        var info = Assert.Single(prov.GetDevices());
        Assert.Equal(Sharplight.Runtime.DeviceKind.Cpu, info.Kind);
        Assert.False(string.IsNullOrWhiteSpace(info.Name));
        Assert.True(info.TotalMemory > (1L << 28));
        Assert.Equal(Environment.ProcessorCount.ToString(), info.Capabilities["threads"]);
        Assert.Contains("avx512", info.Capabilities.Keys);
        using var dev = prov.OpenDevice(info);
        Assert.True(dev.AvailableMemory > 0);
        CpuExecutionProvider.Register();
        Assert.Contains(Sharplight.Runtime.ProviderRegistry.Discover(), d => d.Provider.Name == "cpu" && d.Availability.IsAvailable);
    }

    [Fact]
    public void UnknownOpIsUnsupported()
    {
        var tg = new TestGraph();
        var x = tg.InF("x", TestGraph.T, 4);
        var op = new OpSchema("MyCustomOp", OpCategory.Composite, 1, 1, (i, a) => [new TensorType(DataType.F32, i[0].Shape)]);
        OpRegistry.Register(op);
        var y = tg.B.Emit(op, [x])[0];
        tg.B.Output("y", y);
        var sup = Kit.Device.Supports(tg.B.Build().Nodes[0]);
        Assert.False(sup.IsSupported);
        Assert.Contains("MyCustomOp", sup.Reason);
    }
}
