using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The device sampling kernel (<see cref="StandardOps.Sample"/>) against the CPU implementation of the op (same random stream, so the
/// same token up to rounding) and against the exact target distribution of the filters on synthetic logits.
/// </summary>
public sealed class SampleTests
{
    private const int MaxRows = 256;

    private sealed class Sampler : IDisposable
    {
        private readonly CompiledModel _model;
        private readonly Value _logits;
        private readonly Value _params;
        private readonly int _vocab;

        public Sampler(IComputeDevice device, int vocab)
        {
            _vocab = vocab;
            var b = new GraphBuilder("sample", _ => null);
            var r = Dim.Sym("R");
            _logits = b.Input("logits", InputRole.Other, DataType.F32, new Shape(r, Dim.Of(vocab)));
            _params = b.Input("params", InputRole.SamplingParameters, DataType.I32, new Shape(r, Dim.Of(SamplingRow.Width)));
            var outs = b.Emit(StandardOps.Sample, [_logits, _params]);
            b.Output("tokens", outs[0]);
            b.Output("logprobs", outs[1]);
            _model = CompiledModel.Compile(b.Build(), new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Compile = new CompileOptions { MaxTokens = MaxRows, ContextLength = 16, SymbolBounds = new Dictionary<string, long> { ["R"] = MaxRows } },
                ReservePerDevice = 0,
            });
        }

        /// <summary>Samples every row of <paramref name="logits"/> ([rows, vocab]) with its parameters.</summary>
        public (int[] Tokens, float[] Logprobs) Run(float[] logits, IReadOnlyList<SamplingRow?> rows)
        {
            int n = rows.Count;
            var encoded = new int[n * SamplingRow.Width];
            for (int i = 0; i < n; i++)
            {
                var span = encoded.AsSpan(i * SamplingRow.Width, SamplingRow.Width);
                if (rows[i] is { } row)
                {
                    row.WriteTo(span);
                }
                else
                {
                    SamplingRow.WriteSkip(span);
                }
            }

            _model.SetInput(_logits, MemoryMarshal.AsBytes(logits.AsSpan(0, n * _vocab)));
            _model.SetInput<int>(_params, encoded);
            var p = new RunParameters();
            p.Symbols["R"] = n;
            _model.Run(p);
            var tokens = new int[n];
            var logprobs = new float[n];
            _model.GetOutput<int>("tokens", tokens);
            _model.GetOutput<float>("logprobs", logprobs);
            return (tokens, logprobs);
        }

        public void Dispose() => _model.Dispose();
    }

    /// <summary>
    /// Logits shaped like a language model's: a few strong candidates on top of a broad tail, uniform in [-3, 3] or (when
    /// <paramref name="gaussian"/>) normal with deviation 1.5, whose sparse upper end puts the top-k into the kernel's
    /// candidate window instead of its radix select.
    /// </summary>
    private static float[] Logits(int rows, int vocab, int seed, bool gaussian = false)
    {
        var rng = new Random(seed);
        var x = new float[rows * vocab];
        for (int r = 0; r < rows; r++)
        {
            for (int i = 0; i < vocab; i++)
            {
                x[r * vocab + i] = gaussian
                    ? (float)(1.5 * Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()))
                    : (float)(rng.NextDouble() * 6 - 3);
            }

            for (int j = 0; j < 20; j++)
            {
                x[r * vocab + rng.Next(vocab)] = 4 + (float)rng.NextDouble() * 8;
            }
        }

        return x;
    }

    public static TheoryData<int> Vocabularies => [1000, 32768, 128256, 248320];

    [Theory]
    [MemberData(nameof(Vocabularies))]
    public void GreedyIsTheFirstArgMax(int vocab)
    {
        Gpu.Require();
        const int Rows = 9;
        var x = Logits(Rows, vocab, 1);
        // Ties: row 1 has two equal maxima, row 2 is constant, row 3 has its maximum in the last entry.
        x[vocab + 7] = 100;
        x[vocab + vocab - 3] = 100;
        x.AsSpan(2 * vocab, vocab).Fill(-1.5f);
        x[3 * vocab + vocab - 1] = 1000;
        using var gpu = new Sampler(Gpu.Vulkan!, vocab);
        var rows = Enumerable.Range(0, Rows).Select(i => (SamplingRow?)SamplingRow.Greedy(wantLogprob: i % 2 == 0)).ToList();
        rows[5] = null;
        var (tokens, logprobs) = gpu.Run(x, rows);
        for (int r = 0; r < Rows; r++)
        {
            if (r == 5)
            {
                Assert.Equal(-1, tokens[r]);
                continue;
            }

            var row = x.AsSpan(r * vocab, vocab);
            int expected = row.IndexOf(MaxOf(row));
            Assert.Equal(expected, tokens[r]);
            if (r % 2 == 0)
            {
                Assert.Equal(LogSoftmax(row, expected), logprobs[r], 3e-4);
            }
        }
    }

    public static TheoryData<string, float, int, float, float> Configurations => new()
    {
        { "temperature", 0.8f, 0, 1f, 0f },
        { "top-k 40", 1f, 40, 1f, 0f },
        { "top-k 1", 1f, 1, 1f, 0f },
        { "top-k 5000", 1.2f, 5000, 1f, 0f },
        { "top-p 0.9", 0.7f, 0, 0.9f, 0f },
        { "top-k 40 top-p 0.95", 0.7f, 40, 0.95f, 0f },
        { "min-p 0.05", 1f, 0, 1f, 0.05f },
        { "all", 1.3f, 64, 0.8f, 0.02f },
        { "hot top-p", 3f, 0, 0.99f, 0f },
    };

    [Theory]
    [MemberData(nameof(Configurations))]
    public void SampledRowsMatchTheCpuOp(string name, float temperature, int topK, float topP, float minP)
    {
        Gpu.Require();
        const int Rows = 64;
        foreach (var (vocab, gaussian) in new[] { (1000, false), (128256, false), (128256, true), (151935, true) })
        {
            var x = Logits(Rows, vocab, vocab + topK, gaussian);
            var rows = Enumerable.Range(0, Rows).Select(i => (SamplingRow?)new SamplingRow
            {
                Temperature = temperature, TopK = topK, TopP = topP, MinP = minP, Seed = 1234, Counter = (uint)i, WantLogprob = i % 3 == 0,
            }).ToList();
            using var gpu = new Sampler(Gpu.Vulkan!, vocab);
            using var cpu = new Sampler(Gpu.Cpu, vocab);
            var a = gpu.Run(x, rows);
            var b = cpu.Run(x, rows);
            int mismatches = Enumerable.Range(0, Rows).Count(i => a.Tokens[i] != b.Tokens[i]);
            Assert.True(mismatches <= 1, $"{name} V={vocab} gaussian={gaussian}: {mismatches} of {Rows} rows differ between Vulkan and CPU");
            for (int i = 0; i < Rows; i += 3)
            {
                if (a.Tokens[i] == b.Tokens[i])
                {
                    Assert.Equal(b.Logprobs[i], a.Logprobs[i], 1e-3);
                }
            }

            // Same parameters again: the choice is a pure function of logits and parameters.
            Assert.Equal(a.Tokens, gpu.Run(x, rows).Tokens);
        }
    }

    [Theory]
    [MemberData(nameof(Configurations))]
    public void SampledDistributionMatchesTheFilters(string name, float temperature, int topK, float topP, float minP)
    {
        Gpu.Require();
        const int Vocab = 300;
        const int Draws = 100 * MaxRows;
        var one = Logits(1, Vocab, 7);
        var target = TargetDistribution(one, temperature, topK, topP, minP);
        var x = new float[MaxRows * Vocab];
        for (int r = 0; r < MaxRows; r++)
        {
            one.CopyTo(x, r * Vocab);
        }

        using var gpu = new Sampler(Gpu.Vulkan!, Vocab);
        var counts = new int[Vocab];
        for (int batch = 0; batch < Draws / MaxRows; batch++)
        {
            var rows = Enumerable.Range(0, MaxRows).Select(i => (SamplingRow?)new SamplingRow
            {
                Temperature = temperature, TopK = topK, TopP = topP, MinP = minP, Seed = 99, Counter = (uint)(batch * MaxRows + i),
            }).ToList();
            foreach (var t in gpu.Run(x, rows).Tokens)
            {
                counts[t]++;
            }
        }

        // Total variation between drawn and target distribution, against its expectation under multinomial noise.
        double tv = 0;
        double expected = 0;
        for (int i = 0; i < Vocab; i++)
        {
            Assert.False(target[i] == 0 && counts[i] > 0, $"{name}: token {i} is outside the filtered set but was drawn {counts[i]} times");
            tv += Math.Abs(counts[i] / (double)Draws - target[i]) / 2;
            expected += Math.Sqrt(2 * target[i] * (1 - target[i]) / (Math.PI * Draws)) / 2;
        }

        Assert.True(tv < 1.5 * expected + 0.005, $"{name}: total variation {tv:F4} between drawn and target distribution (noise level {expected:F4})");
    }

    /// <summary>The distribution the filters define (independent of the kernel's algorithm), in double precision.</summary>
    private static double[] TargetDistribution(float[] x, float temperature, int topK, float topP, float minP)
    {
        int n = x.Length;
        var order = Enumerable.Range(0, n).OrderByDescending(i => x[i]).ThenBy(i => i).ToArray();
        double m = x[order[0]];
        var p = order.Select(i => Math.Exp((x[i] - m) / temperature)).ToArray();
        bool nucleus = topP > 0 && topP < 1;
        int k = nucleus && (topK == 0 || topK > SamplingHead.MaxNucleusCandidates) ? SamplingHead.MaxNucleusCandidates : topK;
        int kept = k > 0 && k < n ? k : n;
        double z = p.Take(kept).Sum();
        int cut = kept;
        for (int j = 1; j < kept && minP > 0; j++)
        {
            if (p[j] < minP)
            {
                cut = j;
                break;
            }
        }

        if (nucleus)
        {
            double acc = 0;
            for (int j = 0; j < cut; j++)
            {
                acc += p[j];
                if (acc >= topP * z)
                {
                    cut = j + 1;
                    break;
                }
            }
        }

        double total = p.Take(cut).Sum();
        var target = new double[n];
        for (int j = 0; j < cut; j++)
        {
            target[order[j]] = p[j] / total;
        }

        return target;
    }

    private static float MaxOf(ReadOnlySpan<float> x)
    {
        float m = float.NegativeInfinity;
        foreach (var v in x)
        {
            m = Math.Max(m, v);
        }

        return m;
    }

    private static float LogSoftmax(ReadOnlySpan<float> x, int i)
    {
        double m = MaxOf(x);
        double s = 0;
        foreach (var v in x)
        {
            s += Math.Exp(v - m);
        }

        return (float)(x[i] - m - Math.Log(s));
    }
}
