using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The chunked (tensor-core) gated delta rule against the CPU oracle: batches long enough for the chunked form (64-row chunks,
/// partial last chunks, several chunks per step) interleaved with short batches of the token-by-token kernel, so the state is
/// handed back and forth between the two; single and multi-sequence (runs of both kinds in one step, a slot that starts
/// over). q and k are L2-normalized, as in every model using the op, either inside the graph (absorbed into the kernels) or
/// beforehand. <c>reduced</c> selects the one-product f16 form (reduced precision), otherwise the two-part split. The graphs
/// also carry the causal conv1d that precedes the op in the models, so its token-parallel prefill form (and the state
/// hand-over to the per-channel walk of short batches) is checked on the same batches.
/// </summary>
public sealed class ChunkedDeltaRuleTests
{
    private const int Slots = 3;
    private const int ConvChannels = 136;
    private const int ConvKernel = 4;

    [Theory]
    [InlineData(1, 3, 128, 128, true, false, 0.5f)]
    [InlineData(1, 3, 128, 128, true, true, 0.5f)]
    [InlineData(2, 4, 64, 32, false, false, 0.05f)]
    [InlineData(4, 4, 32, 64, true, false, 0.02f)]
    [InlineData(2, 2, 16, 48, false, false, 0.5f)]
    [InlineData(2, 6, 128, 128, true, false, 0.01f)]
    public void ChunkedMatchesCpu(int hk, int hv, int dk, int dv, bool normInGraph, bool reduced, float decay)
    {
        Gpu.Require();
        var (tg, inputs) = Build(hk, hv, dk, dv, normInGraph);
        int[] chunks = [150, 1, 64, 3, 63, 200, 1];
        var steps = chunks.Select((c, i) => new Step(c, 0, Inputs(inputs, c, hk, hv, dk, dv, normInGraph, decay, 100 * i))).ToList();
        var (atol, rtol) = reduced ? (3e-3, 3e-2) : (2e-5, 2e-4);
        Diff.Check("GatedDeltaRule.chunked", tg.B.Build(), steps, atol: atol, rtol: rtol, maxTokens: 256,
            what: $"Hk={hk} Hv={hv} Dk={dk} Dv={dv} norm={normInGraph} reduced={reduced} decay={decay}", allowReducedPrecision: reduced);
    }

    [Theory]
    [InlineData(2, 4, 64, 64, false)]
    [InlineData(1, 3, 128, 128, true)]
    public void ChunkedMultiSequenceMatchesCpu(int hk, int hv, int dk, int dv, bool reduced)
    {
        Gpu.Require();
        var (tg, inputs) = Build(hk, hv, dk, dv, normInGraph: true);
        var g = MultiSequence.Transform(tg.B.Build(), Slots).Graph;
        Value In(string name) => g.Inputs.Single(v => v.Name == name);

        // Long runs (chunked form) next to short ones (token by token) in one step; slot 0 starts over at step 3.
        var script = new List<(int Slot, int Tokens)[]>
        {
            new[] { (0, 70), (1, 5) },
            new[] { (0, 1), (1, 1), (2, 130) },
            new[] { (0, 64), (1, 100), (2, 1) },
            new[] { (0, 65), (2, 2) },
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

            var data = Inputs(inputs, rows, hk, hv, dk, dv, normInGraph: true, decay: 0.1f, seed: 100 * si)
                .Select(x => (In(x.V.Name), x.Data))
                .Append((In(MultiSequence.RowsInput), seqRows))
                .Append((In("sequence_runs"), runs));
            steps.Add(new Step(rows, 0, [.. data]) { Symbols = new Dictionary<string, long> { [WellKnown.RunSymbol] = step.Length } });
        }

        var (atol, rtol) = reduced ? (3e-3, 3e-2) : (2e-5, 2e-4);
        Diff.Check("GatedDeltaRule.chunked.multi", g, steps, atol: atol, rtol: rtol, maxTokens: 256,
            what: $"Hk={hk} Hv={hv} Dk={dk} Dv={dv} reduced={reduced}", allowReducedPrecision: reduced);
    }

    /// <summary>The token-by-token kernel (no f16 matrices) on the same long batches, as the reference for the chunked form.</summary>
    [Fact]
    public void SequentialFallbackMatchesCpu()
    {
        Gpu.Require();
        var (tg, inputs) = Build(1, 3, 128, 128, normInGraph: true);
        int[] chunks = [150, 1, 64];
        var steps = chunks.Select((c, i) => new Step(c, 0, Inputs(inputs, c, 1, 3, 128, 128, true, 0.5f, 100 * i))).ToList();
        Diff.Check("GatedDeltaRule.sequential", tg.B.Build(), steps, atol: 2e-4, rtol: 2e-3, maxTokens: 256,
            disabledFeatures: new HashSet<string> { DeviceFeatures.MatrixF16 });
    }

    private static (TG Tg, Value[] Inputs) Build(int hk, int hv, int dk, int dv, bool normInGraph)
    {
        var tg = new TG();
        var t = TG.T;
        var qi = tg.InF("q", t, hk, dk);
        var ki = tg.InF("k", t, hk, dk);
        var vi = tg.InF("v", t, hv, dv);
        var gi = tg.InF("g", t, hv);
        var bi = tg.InF("b", t, hv);
        var st = tg.State("ssm", DataType.F32, StateKind.Recurrent, hv, dv, dk);
        Value q = normInGraph ? tg.B.L2Norm(qi, 1e-6f) : qi;
        Value k = normInGraph ? tg.B.L2Norm(ki, 1e-6f) : ki;
        tg.B.Output("o", tg.B.Emit("GatedDeltaRule", [q, k, vi, gi, bi, st], Attributes.From(("scale", 1f / MathF.Sqrt(dk))), writes: [st])[0]);
        var cx = tg.InF("cx", t, ConvChannels);
        var cw = tg.RandW("cw", DataType.F32, 1f, ConvChannels, ConvKernel);
        var cs = tg.State("cs", DataType.F32, StateKind.Recurrent, ConvKernel - 1, ConvChannels);
        tg.B.Output("cy", tg.B.Emit("CausalConv1d", [cx, cw, cs], Attributes.From(("activation", "silu")), writes: [cs])[0]);
        return (tg, [qi, ki, vi, gi, bi, cx]);
    }

    /// <summary>Random inputs for <paramref name="rows"/> rows: unit q/k rows (unless normalized in the graph), log decays in [-decay, 0].</summary>
    private static (Value V, Array Data)[] Inputs(Value[] inputs, int rows, int hk, int hv, int dk, int dv, bool normInGraph, float decay, int seed)
    {
        float[] Unit(float[] x)
        {
            if (normInGraph)
            {
                return x;
            }

            for (int r = 0; r < x.Length / dk; r++)
            {
                var row = x.AsSpan(r * dk, dk);
                float norm = MathF.Sqrt(row.ToArray().Sum(e => e * e));
                foreach (ref float e in row)
                {
                    e /= norm;
                }
            }

            return x;
        }

        return
        [
            (inputs[0], Unit(Rnd.F((long)rows * hk * dk, seed + 1))),
            (inputs[1], Unit(Rnd.F((long)rows * hk * dk, seed + 2))),
            (inputs[2], Rnd.F((long)rows * hv * dv, seed + 3)),
            (inputs[3], Rnd.F((long)rows * hv, seed + 4).Select(x => -Math.Abs(x) * decay).ToArray()),
            (inputs[4], Rnd.F((long)rows * hv, seed + 5).Select(Math.Abs).ToArray()),
            (inputs[5], Rnd.F((long)rows * ConvChannels, seed + 6)),
        ];
    }
}
