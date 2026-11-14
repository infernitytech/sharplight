using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>Differential harness: runs one graph on the Vulkan device and on the CPU oracle and compares outputs.</summary>
internal static class Diff
{
    public static readonly ConcurrentDictionary<string, (double MaxAbs, double MaxScaled, int Cases)> Results = new();

    /// <summary>Largest relative RMS error (sqrt(sum d^2 / sum ref^2) of one output in one step) per op and case.</summary>
    public static readonly ConcurrentDictionary<string, double> RelativeRms = new();

    /// <summary>Features the CPU oracle runs without, so that it computes in f32 (the integer-dot matmuls quantize activations).</summary>
    private static readonly HashSet<string> s_exactOracle = [DeviceFeatures.IntDot];

    static Diff()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try
            {
                var lines = Results.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key,-40} cases={kv.Value.Cases,3} maxAbs={kv.Value.MaxAbs:E2} maxErr/tol={kv.Value.MaxScaled:F3}")
                    .Concat(RelativeRms.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key,-60} relRms={kv.Value:E2}"));
                File.WriteAllLines(Path.Combine(AppContext.BaseDirectory, "diff-summary.txt"), lines);
            }
            catch { }
        };
    }

    private static CompiledModel Compile(IComputeDevice dev, ComputeGraph g, long maxTokens, long ctx, IReadOnlySet<string>? disabledFeatures = null,
        bool allowReducedPrecision = false)
        => CompiledModel.Compile(g, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = dev }],
            Compile = new CompileOptions
            {
                MaxTokens = maxTokens,
                ContextLength = ctx,
                SymbolBounds = new Dictionary<string, long> { ["R"] = maxTokens, [Models.WellKnown.RunSymbol] = maxTokens },
                DisabledFeatures = disabledFeatures ?? new HashSet<string>(),
                AllowReducedPrecision = allowReducedPrecision,
            },
            ReservePerDevice = 0,
        });

    private static RunParameters Params(Step s)
    {
        var p = new RunParameters();
        p.Symbols["T"] = s.T; p.Symbols["R"] = s.R ?? s.T;
        p.Scalars["n_past"] = s.Past;
        foreach (var (name, value) in s.Symbols ?? new Dictionary<string, long>())
        {
            p.Symbols[name] = value;
        }

        return p;
    }

    private static void Feed(CompiledModel m, Step s)
    {
        foreach (var (v, data) in s.Inputs)
        {
            switch (data)
            {
                case float[] f: m.SetInput(v, MemoryMarshal.AsBytes(f.AsSpan())); break;
                case int[] i: m.SetInput(v, MemoryMarshal.AsBytes(i.AsSpan())); break;
                case long[] l: m.SetInput(v, MemoryMarshal.AsBytes(l.AsSpan())); break;
                default: throw new ArgumentException("input data must be float[] or int[]");
            }
        }
    }

    /// <summary>
    /// Runs all steps on both devices (state persists across steps) and compares every graph output.
    /// Returns the largest |a-b| seen. <paramref name="disabledFeatures"/> are turned off for the Vulkan compilation
    /// (to exercise the portable fallback kernels on a device that has the features); <paramref name="allowReducedPrecision"/>
    /// lets the Vulkan compilation use reduced-precision products.
    /// </summary>
    public static double Check(string op, ComputeGraph g, IReadOnlyList<Step> steps, double atol = 1e-4, double rtol = 1e-4, long maxTokens = 64, long ctx = 64, string? what = null,
        IReadOnlySet<string>? disabledFeatures = null, bool allowReducedPrecision = false)
    {
        Gpu.Require();
        using var vk = Compile(Gpu.Vulkan!, g, maxTokens, ctx, disabledFeatures, allowReducedPrecision);
        CompiledModel cpu;
        try
        {
            // The oracle stays exact: no int8-quantized activations on the CPU's integer-dot kernels.
            cpu = Compile(Gpu.Cpu, g, maxTokens, ctx, s_exactOracle);
        }
        catch (NotSupportedException e)
        {
            Assert.Skip("CPU oracle does not support this graph yet: " + e.Message);
            return 0;
        }
        using var _ = cpu;
        double maxAbs = 0, maxScaled = 0;
        for (int si = 0; si < steps.Count; si++)
        {
            var s = steps[si];
            var p = Params(s);
            var vkParameters = Params(s);
            Feed(vk, s); Feed(cpu, s);
            vk.Run(vkParameters); cpu.Run(p);
            // Data-dependent ops (NonMaxSuppression, ...) publish their counts: both devices must find the same.
            foreach (var (symbol, value) in p.Symbols)
            {
                long found = vkParameters.Symbols.GetValueOrDefault(symbol, -1);
                Assert.True(found == value, $"{op} {what} step {si}: symbol {symbol} is {found} on vulkan, {value} on cpu");
            }
            foreach (var (name, v) in g.Outputs)
            {
                long n = v.Shape.Elements(p.Symbols);
                if (v.Type == DataType.I32)
                {
                    var a = new int[n];
                    var b = new int[n];
                    vk.GetOutput<int>(name, a); cpu.GetOutput<int>(name, b);
                    for (long i = 0; i < n; i++)
                    {
                        if (a[i] != b[i])
                        {
                            Assert.Fail($"{op} {what} step {si} output {name}[{i}]: vulkan {a[i]} != cpu {b[i]}");
                        }
                    }

                    continue;
                }
                if (v.Type == DataType.I64)
                {
                    var a = new long[n];
                    var b = new long[n];
                    vk.GetOutput<long>(name, a); cpu.GetOutput<long>(name, b);
                    for (long i = 0; i < n; i++)
                    {
                        if (a[i] != b[i])
                        {
                            Assert.Fail($"{op} {what} step {si} output {name}[{i}]: vulkan {a[i]} != cpu {b[i]}");
                        }
                    }

                    continue;
                }
                var x = new float[n];
                var y = new float[n];
                vk.GetOutput<float>(name, x); cpu.GetOutput<float>(name, y);
                double sumDiff = 0, sumRef = 0;
                for (long i = 0; i < n; i++)
                {
                    sumDiff += ((double)x[i] - y[i]) * ((double)x[i] - y[i]);
                    sumRef += (double)y[i] * y[i];
                    double d = Math.Abs((double)x[i] - y[i]);
                    double tol = atol + rtol * Math.Abs(y[i]);
                    if (double.IsNaN(x[i]) != double.IsNaN(y[i]) || (!double.IsNaN(y[i]) && d > tol))
                    {
                        Assert.Fail($"{op} {what} step {si} output {name}[{i}]: vulkan {x[i]:G9} vs cpu {y[i]:G9} (|d|={d:E2}, tol={tol:E2})");
                    }

                    if (!double.IsNaN(d))
                    {
                        maxAbs = Math.Max(maxAbs, d);
                        maxScaled = Math.Max(maxScaled, d / tol);
                    }
                }

                double rel = sumRef > 0 ? Math.Sqrt(sumDiff / sumRef) : Math.Sqrt(sumDiff);
                RelativeRms.AddOrUpdate($"{op} {what}", rel, (_, old) => Math.Max(old, rel));
            }
        }
        Results.AddOrUpdate(op, (maxAbs, maxScaled, 1), (_, o) => (Math.Max(o.MaxAbs, maxAbs), Math.Max(o.MaxScaled, maxScaled), o.Cases + 1));
        return maxAbs;
    }

    public static double Check(string op, ComputeGraph g, Step step, double atol = 1e-4, double rtol = 1e-4, long maxTokens = 64, long ctx = 64, string? what = null,
        IReadOnlySet<string>? disabledFeatures = null, bool allowReducedPrecision = false)
        => Check(op, g, [step], atol, rtol, maxTokens, ctx, what, disabledFeatures, allowReducedPrecision);
}
