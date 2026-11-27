using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The selective scan (Mamba-2 with a decay per head, Mamba-1 with a decay per state element) and the RWKV-6 / RWKV-7 recurrences
/// with their state in f32, bf16 or f16 against the CPU oracle, over steps of several lengths and in the multi-sequence form (long
/// and short runs in one step, a slot that starts over). Both providers round a 16-bit state after every token, so they follow the
/// same rounded trajectory; the wider 16-bit tolerance only covers the rare rounding ties that the two providers' f32 sums break
/// differently.
/// </summary>
public sealed class RecurrentState16Tests
{
    private const int Slots = 3;

    [Theory]
    [InlineData("SelectiveScan", "f32")]
    [InlineData("SelectiveScan", "bf16")]
    [InlineData("SelectiveScan", "f16")]
    [InlineData("SelectiveScan1", "bf16")]
    [InlineData("SelectiveScan1", "f16")]
    [InlineData("Wkv7", "f32")]
    [InlineData("Wkv7", "bf16")]
    [InlineData("Wkv7", "f16")]
    [InlineData("Wkv6", "f32")]
    [InlineData("Wkv6", "bf16")]
    [InlineData("Wkv6", "f16")]
    public void MatchesCpuAcrossSteps(string op, string stateType)
    {
        Gpu.Require();
        var (g, inputs) = Build(op, DataType.Get(stateType));
        int[] chunks = [3, 1, 40, 2, 1, 17];
        var steps = chunks.Select((c, i) => new Step(c, 0, Inputs(op, inputs, c, 100 * i))).ToList();
        var (atol, rtol) = Tolerance(stateType);
        Diff.Check(op, g, steps, atol: atol, rtol: rtol, maxTokens: 64, what: $"{op} state={stateType}");
    }

    [Theory]
    [InlineData("SelectiveScan", "bf16")]
    [InlineData("SelectiveScan1", "f16")]
    [InlineData("Wkv7", "bf16")]
    [InlineData("Wkv6", "f16")]
    public void MultiSequenceMatchesCpu(string op, string stateType)
    {
        Gpu.Require();
        var (single, inputs) = Build(op, DataType.Get(stateType));
        var g = MultiSequence.Transform(single, Slots).Graph;
        Value In(string name) => g.Inputs.Single(v => v.Name == name);

        // Slot 0 starts over at step 3.
        var script = new List<(int Slot, int Tokens)[]>
        {
            new[] { (0, 20), (1, 5) },
            new[] { (0, 1), (1, 1), (2, 30) },
            new[] { (0, 9), (1, 12), (2, 1) },
            new[] { (0, 7), (2, 2) },
            new[] { (0, 1), (1, 1), (2, 1) },
        };
        const int ResetStep = 3;
        var length = new int[Slots];
        var steps = new List<Step>();
        foreach (var (step, si) in script.Select((x, i) => (x, i)))
        {
            if (si == ResetStep)
            {
                length[0] = 0;
            }

            int rows = step.Sum(x => x.Tokens);
            var seqRows = new int[rows * 4];
            var runs = new int[step.Length * 4];
            int r = 0;
            for (int u = 0; u < step.Length; u++)
            {
                var (slot, n) = step[u];
                for (int i = 0; i < n; i++, r++)
                {
                    seqRows[r * 4] = slot;
                    seqRows[r * 4 + 1] = length[slot] + i;
                    seqRows[r * 4 + 2] = n;
                    seqRows[r * 4 + 3] = u;
                }

                runs[u * 4] = slot;
                runs[u * 4 + 1] = r - n;
                runs[u * 4 + 2] = n;
                runs[u * 4 + 3] = length[slot];
                length[slot] += n;
            }

            var data = Inputs(op, inputs, rows, 100 * si)
                .Select(x => (In(x.V.Name), x.Data))
                .Append((In(MultiSequence.RowsInput), seqRows))
                .Append((In("sequence_runs"), runs));
            steps.Add(new Step(rows, 0, [.. data]) { Symbols = new Dictionary<string, long> { [WellKnown.RunSymbol] = step.Length } });
        }

        var (atol, rtol) = Tolerance(stateType);
        Diff.Check(op + ".multi", g, steps, atol: atol, rtol: rtol, maxTokens: 64, what: $"{op} state={stateType}");
    }

    private static (double Atol, double Rtol) Tolerance(string stateType) => stateType == "f32" ? (2e-4, 2e-3) : (5e-3, 3e-2);

    // Head shapes: selective scan H = 4 heads of P = 16 rows, N = 32, G = 2 groups (Mamba-1: H = 48 channels, P = 1, N = 16);
    // RWKV H = 2 heads of N = 64.
    private const int ScanHeads = 4;
    private const int ScanRows = 16;
    private const int ScanState = 32;
    private const int ScanGroups = 2;
    private const int Mamba1Channels = 48;
    private const int Mamba1State = 16;
    private const int WkvHeads = 2;
    private const int WkvSize = 64;

    private static (ComputeGraph Graph, Value[] Inputs) Build(string op, DataType stateType)
    {
        var tg = new TG();
        var t = TG.T;
        switch (op)
        {
            case "SelectiveScan":
            case "SelectiveScan1":
            {
                bool mamba1 = op == "SelectiveScan1";
                int h = mamba1 ? Mamba1Channels : ScanHeads;
                int p = mamba1 ? 1 : ScanRows;
                int n = mamba1 ? Mamba1State : ScanState;
                int groups = mamba1 ? 1 : ScanGroups;
                var x = tg.InF("x", t, h, p);
                var dt = tg.InF("dt", t, h);
                var b = tg.InF("b", t, groups, n);
                var c = tg.InF("c", t, groups, n);
                var dtb = tg.RandW("dtb", DataType.F32, 1f, h);
                var a = mamba1 ? tg.RandW("A", DataType.F32, 1f, h, n) : tg.RandW("A", DataType.F32, 1f, h);
                var d = tg.RandW("D", DataType.F32, 1f, h);
                var st = tg.State("ssm", stateType, StateKind.Recurrent, h, p, n);
                var negative = tg.B.Scale(tg.B.Unary("Abs", a), -1f, -0.1f);
                tg.B.Output("y", tg.B.Emit("SelectiveScan", [x, dt, dtb, negative, b, c, d, st], Attributes.From(("dt_min", 0.001f)), writes: [st])[0]);
                return (tg.B.Build(), [x, dt, b, c]);
            }
            case "Wkv7":
            {
                Value[] ins = [.. new[] { "r", "w", "k", "v", "a", "b" }.Select(name => tg.InF(name, t, WkvHeads, WkvSize))];
                var st = tg.State("wkv", stateType, StateKind.Recurrent, WkvHeads, WkvSize, WkvSize);
                tg.B.Output("y", tg.B.Emit("Wkv7", [.. ins, st], null, writes: [st])[0]);
                return (tg.B.Build(), ins);
            }
            case "Wkv6":
            {
                Value[] ins = [.. new[] { "r", "k", "v", "w" }.Select(name => tg.InF(name, t, WkvHeads, WkvSize))];
                var u = tg.RandW("u", DataType.F32, 0.5f, WkvHeads, WkvSize);
                var st = tg.State("wkv", stateType, StateKind.Recurrent, WkvHeads, WkvSize, WkvSize);
                tg.B.Output("y", tg.B.Emit("Wkv6", [.. ins, u, st], null, writes: [st])[0]);
                return (tg.B.Build(), ins);
            }
            default:
                throw new ArgumentException(op);
        }
    }

    /// <summary>
    /// Random inputs for <paramref name="rows"/> rows. RWKV: decays w in [0.8, 1), and for RWKV-7 the in-context learning-rate
    /// vectors a = -kk, b = kk·gate of a unit vector kk per head (as the models build them, which keeps the state bounded).
    /// </summary>
    private static (Value V, Array Data)[] Inputs(string op, Value[] inputs, int rows, int seed)
    {
        float[] Decay(int s) => [.. Rnd.F((long)rows * WkvHeads * WkvSize, s).Select(x => 0.9f + 0.1f * x * 0.99f)];
        switch (op)
        {
            case "SelectiveScan":
            case "SelectiveScan1":
            {
                bool mamba1 = op == "SelectiveScan1";
                int h = mamba1 ? Mamba1Channels : ScanHeads;
                int p = mamba1 ? 1 : ScanRows;
                int n = mamba1 ? Mamba1State : ScanState;
                int groups = mamba1 ? 1 : ScanGroups;
                return [(inputs[0], Rnd.F((long)rows * h * p, seed + 1, 0.5f)), (inputs[1], Rnd.F((long)rows * h, seed + 2, 2f)),
                        (inputs[2], Rnd.F((long)rows * groups * n, seed + 3, 0.5f)), (inputs[3], Rnd.F((long)rows * groups * n, seed + 4, 0.5f))];
            }
            case "Wkv7":
            {
                int width = WkvHeads * WkvSize;
                var kk = Rnd.F((long)rows * width, seed + 5);
                for (int i = 0; i < rows * WkvHeads; i++)
                {
                    var head = kk.AsSpan(i * WkvSize, WkvSize);
                    float norm = MathF.Sqrt(head.ToArray().Sum(x => x * x));
                    foreach (ref float x in head)
                    {
                        x /= norm;
                    }
                }

                var gate = Rnd.F((long)rows * width, seed + 6).Select(x => 0.5f + 0.5f * x).ToArray();
                return [(inputs[0], Rnd.F((long)rows * width, seed + 1)), (inputs[1], Decay(seed + 2)),
                        (inputs[2], Rnd.F((long)rows * width, seed + 3, 0.5f)), (inputs[3], Rnd.F((long)rows * width, seed + 4)),
                        (inputs[4], kk.Select(x => -x).ToArray()), (inputs[5], kk.Select((x, i) => x * gate[i]).ToArray())];
            }
            case "Wkv6":
            {
                int width = WkvHeads * WkvSize;
                return [(inputs[0], Rnd.F((long)rows * width, seed + 1)), (inputs[1], Rnd.F((long)rows * width, seed + 2, 0.5f)),
                        (inputs[2], Rnd.F((long)rows * width, seed + 3)), (inputs[3], Decay(seed + 4))];
            }
            default:
                throw new ArgumentException(op);
        }
    }
}
