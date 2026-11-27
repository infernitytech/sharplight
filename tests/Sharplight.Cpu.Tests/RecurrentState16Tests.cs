using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>
/// The selective scan (Mamba-2, and Mamba-1 with a decay per state element) and the RWKV-6 / RWKV-7 recurrences with a bf16 / f16
/// state: the state is rounded after every token, so the outputs do not depend on how the tokens are split into calls (a speculative
/// verification run of several rows equals token-by-token decoding), and they stay close to the f32 state's.
/// </summary>
public class RecurrentState16Tests
{
    private const int Tokens = 9;

    [Theory]
    [InlineData("SelectiveScan", "bf16", 2e-2)]
    [InlineData("SelectiveScan", "f16", 3e-3)]
    [InlineData("SelectiveScan1", "bf16", 2e-2)]
    [InlineData("SelectiveScan1", "f16", 3e-3)]
    [InlineData("Wkv7", "bf16", 3e-2)]
    [InlineData("Wkv7", "f16", 5e-3)]
    [InlineData("Wkv6", "bf16", 3e-2)]
    [InlineData("Wkv6", "f16", 5e-3)]
    public void SixteenBitStateRoundsEveryToken(string op, string stateType, double vsF32)
    {
        var whole = Run(op, DataType.Get(stateType), [Tokens]);
        Assert.Equal(whole, Run(op, DataType.Get(stateType), [1, 1, 1, 1, 1, 1, 1, 1, 1]));
        Assert.Equal(whole, Run(op, DataType.Get(stateType), [4, 1, 3, 1]));

        var f32 = Run(op, DataType.F32, [Tokens]);
        Assert.NotEqual(f32, whole);
        Kit.AssertClose(f32, whole, vsF32, vsF32 * 10, $"{op} {stateType} vs f32 state");
    }

    /// <summary>Runs the op over <see cref="Tokens"/> tokens split into calls of <paramref name="chunks"/> rows; all outputs concatenated.</summary>
    private static float[] Run(string op, DataType stateType, int[] chunks)
    {
        var (graph, inputs, widths) = Build(op, stateType);
        var data = inputs.Select((v, i) => Data(op, i, widths[i])).ToArray();
        using var m = Kit.Compile(graph, maxTokens: Tokens);
        var got = new List<float>();
        int t0 = 0;
        foreach (int c in chunks)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                Kit.Set(m, inputs[i], data[i][(t0 * widths[i])..((t0 + c) * widths[i])]);
            }

            var p = Kit.P(c);
            m.Run(p);
            got.AddRange(Kit.Out(m, "y", p));
            t0 += c;
        }

        return [.. got];
    }

    // Selective scan: H = 3 heads of P = 8 rows, N = 16, G = 1 (Mamba-1: 12 channels, N = 8); RWKV: H = 2 heads of N = 16.
    private static (ComputeGraph Graph, Value[] Inputs, int[] Widths) Build(string op, DataType stateType)
    {
        var tg = new TestGraph();
        var t = TestGraph.T;
        switch (op)
        {
            case "SelectiveScan":
            case "SelectiveScan1":
            {
                bool mamba1 = op == "SelectiveScan1";
                int h = mamba1 ? 12 : 3;
                int p = mamba1 ? 1 : 8;
                int n = mamba1 ? 8 : 16;
                var x = tg.InF("x", t, h, p);
                var dt = tg.InF("dt", t, h);
                var b = tg.InF("b", t, 1, n);
                var c = tg.InF("c", t, 1, n);
                var a = mamba1 ? tg.WF("A", [.. Kit.Rand(h * n, 7).Select(v => -0.1f - Math.Abs(v))], h, n)
                    : tg.WF("A", [.. Kit.Rand(h, 7).Select(v => -0.1f - Math.Abs(v))], h);
                var st = tg.State("ssm", stateType, StateKind.Recurrent, h, p, n);
                tg.B.Output("y", tg.B.Emit("SelectiveScan", [x, dt, tg.WF("dtb", Kit.Rand(h, 8), h), a, b, c, tg.WF("D", Kit.Rand(h, 9), h), st],
                    Attributes.From(("dt_min", 0.001f)), writes: [st])[0]);
                return (tg.B.Build(), [x, dt, b, c], [h * p, h, n, n]);
            }
            case "Wkv7":
            {
                Value[] ins = [.. new[] { "r", "w", "k", "v", "a", "b" }.Select(name => tg.InF(name, t, 2, 16))];
                var st = tg.State("wkv", stateType, StateKind.Recurrent, 2, 16, 16);
                tg.B.Output("y", tg.B.Emit("Wkv7", [.. ins, st], null, writes: [st])[0]);
                return (tg.B.Build(), ins, [.. ins.Select(_ => 32)]);
            }
            case "Wkv6":
            {
                Value[] ins = [.. new[] { "r", "k", "v", "w" }.Select(name => tg.InF(name, t, 2, 16))];
                var st = tg.State("wkv", stateType, StateKind.Recurrent, 2, 16, 16);
                tg.B.Output("y", tg.B.Emit("Wkv6", [.. ins, tg.WF("u", Kit.Rand(32, 9, 0.5f), 2, 16), st], null, writes: [st])[0]);
                return (tg.B.Build(), ins, [.. ins.Select(_ => 32)]);
            }
            default:
                throw new ArgumentException(op);
        }
    }

    /// <summary>
    /// Input <paramref name="index"/> of the op for all tokens: decays in (0.8, 1) for RWKV's w; RWKV-7's a = -kk and b = kk·gate of
    /// a unit vector kk per head (as the models build them, which keeps the state bounded).
    /// </summary>
    private static float[] Data(string op, int index, int width)
    {
        int n = Tokens * width;
        if (op == "Wkv7" && index is 4 or 5)
        {
            var kk = Kit.Rand(n, 40);
            for (int i = 0; i < n; i += 16)
            {
                float norm = MathF.Sqrt(kk.Skip(i).Take(16).Sum(x => x * x));
                for (int j = i; j < i + 16; j++)
                {
                    kk[j] /= norm;
                }
            }

            var gate = Kit.Rand(n, 41).Select(x => 0.5f + 0.5f * x).ToArray();
            return index == 4 ? [.. kk.Select(x => -x)] : [.. kk.Select((x, i) => x * gate[i])];
        }

        bool decay = (op == "Wkv7" && index == 1) || (op == "Wkv6" && index == 3);
        return decay ? [.. Kit.Rand(n, 30 + index).Select(x => 0.9f + 0.099f * x)] : Kit.Rand(n, 30 + index, op.StartsWith("Selective", StringComparison.Ordinal) && index == 1 ? 2f : 0.7f);
    }
}
