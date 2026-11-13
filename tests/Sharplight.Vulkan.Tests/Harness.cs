using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan;

[assembly: Xunit.CollectionBehavior(CollectionBehavior.CollectionPerAssembly)]

namespace Sharplight.Vulkan.Tests;

/// <summary>Weight bytes in managed memory (pinned for the zero-copy upload path).</summary>
sealed class MemSource : IWeightSource
{
    readonly byte[] _data; readonly GCHandle _h;
    public MemSource(byte[] data) { _data = data; _h = GCHandle.Alloc(data, GCHandleType.Pinned); }
    public long ByteLength => _data.Length;
    public void Read(long offset, Span<byte> destination) => _data.AsSpan((int)offset, destination.Length).CopyTo(destination);
    public nint TryGetPointer() => _h.AddrOfPinnedObject();
}

/// <summary>Graph builder wrapper owning weights (random or explicit).</summary>
sealed class TG
{
    public readonly Dictionary<string, WeightInfo> Weights = [];
    public readonly GraphBuilder B;
    public static readonly Dim T = Dim.Sym("T");
    int _seed = 1000;
    public TG(string name = "t") { B = new GraphBuilder(name, n => Weights.GetValueOrDefault(n)); }

    public Value W(string name, DataType type, Shape shape, byte[] data)
    {
        Weights[name] = new WeightInfo { Name = name, Type = type, Shape = shape, Source = new MemSource(data) };
        return B.Weight(name);
    }

    public Value WF(string name, float[] data, params long[] dims)
        => W(name, DataType.F32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    /// <summary>Random weight in any float or quantized format.</summary>
    public Value RandW(string name, DataType type, float scale, params long[] dims)
    {
        long n = dims.Aggregate(1L, (a, b) => a * b);
        var bytes = Rnd.WeightBytes(type, n, _seed++, scale);
        return W(name, type, new Shape(dims.Select(d => (Dim)d)), bytes);
    }

    public Value Const(string name, float[] data, params long[] dims)
        => B.Constant(name, DataType.F32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    public Value In(string name, DataType type, params Dim[] dims) => B.Input(name, InputRole.Other, type, new Shape(dims));
    public Value InF(string name, params Dim[] dims) => In(name, DataType.F32, dims);
    public Value State(string name, DataType type, StateKind kind, params Dim[] dims)
        => B.State(new StateDefinition { Name = name, Type = type, Shape = new Shape(dims), Kind = kind });
}

static class Rnd
{
    public static float[] F(long n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (long i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return a;
    }

    public static float[] Pos(long n, int seed, float lo = 0.1f, float hi = 2f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (long i = 0; i < n; i++) a[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return a;
    }

    public static int[] I(long n, int seed, int max)
    {
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, (int)n).Select(_ => rng.Next(max))];
    }

    static void PutHalf(byte[] b, long off, float v) => BitConverter.TryWriteBytes(b.AsSpan((int)off, 2), (Half)v);

    /// <summary>Random but valid weight bytes (float formats: uniform values; quantized: random quants with sane f16 scales).</summary>
    public static byte[] WeightBytes(DataType t, long elements, int seed, float scale)
    {
        if (t == DataType.F32) return MemoryMarshal.AsBytes(F(elements, seed, scale).AsSpan()).ToArray();
        if (t == DataType.F16) return MemoryMarshal.AsBytes([.. F(elements, seed, scale).Select(v => (Half)v)]).ToArray();
        if (t == DataType.BF16)
            return MemoryMarshal.AsBytes([.. F(elements, seed, scale).Select(v => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16))]).ToArray();
        var bytes = new byte[t.ByteSize(elements)];
        new Random(seed).NextBytes(bytes);
        long blocks = elements / t.BlockSize;
        var rng = new Random(seed + 7);
        for (long i = 0; i < blocks; i++)
        {
            long o = i * t.BlockBytes;
            float d() => (0.002f + (float)rng.NextDouble() * 0.01f) * scale;
            if (t == DataType.Q8_0 || t == DataType.Q4_0) PutHalf(bytes, o, d());
            else if (t == DataType.Q4_K || t == DataType.Q5_K) { PutHalf(bytes, o, d()); PutHalf(bytes, o + 2, d()); }
            else if (t == DataType.Q6_K) PutHalf(bytes, o + 208, d());
            else if (t == DataType.Q1_0 || t == DataType.Q2_0 || t == DataType.Q2_0_G128) PutHalf(bytes, o, d());
            else if (t == DataType.Tern1_G128) PutHalf(bytes, o + 26, d());
            else throw new NotSupportedException(t.Name);
        }
        return bytes;
    }
}

/// <summary>Device access; tests skip when no Vulkan device is available.</summary>
static class Gpu
{
    static readonly Lazy<(IComputeDevice? Dev, string Reason)> s_vk = new(() =>
    {
        try
        {
            var p = new VulkanProvider();
            var av = p.CheckAvailability();
            if (!av.IsAvailable) return (null, av.Reason);
            // SHARPLIGHT_TEST_DEVICE=<index> selects another device (e.g. the iGPU or llvmpipe) to exercise feature fallbacks.
            var devs = p.GetDevices();
            var dev = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_DEVICE"), out int di) && di >= 0 && di < devs.Count ? devs[di]
                : devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
            if (dev is null) return (null, "no Vulkan device");
            return (p.OpenDevice(dev), "");
        }
        catch (Exception e) { return (null, e.Message); }
    });

    static readonly Lazy<IComputeDevice> s_cpu = new(() => { var p = new CpuExecutionProvider(); return p.OpenDevice(p.GetDevices()[0]); });

    public static IComputeDevice Cpu => s_cpu.Value;
    public static IComputeDevice? Vulkan => s_vk.Value.Dev;
    public static void Require() => Assert.SkipUnless(Vulkan is not null, "No usable Vulkan device: " + s_vk.Value.Reason);
}

/// <summary>One execution of a compiled graph: symbol bindings, runtime scalars and input data.</summary>
sealed record Step(long T, long Past, params (Value V, Array Data)[] Inputs)
{
    /// <summary>Value of the secondary symbol "R" (e.g. number of gathered rows); defaults to T.</summary>
    public long? R { get; init; }
    public Step With(params (Value, Array)[] more) => this with { Inputs = [.. Inputs, .. more] };
}

/// <summary>Differential harness: runs one graph on the Vulkan device and on the CPU oracle and compares outputs.</summary>
static class Diff
{
    public static readonly ConcurrentDictionary<string, (double MaxAbs, double MaxScaled, int Cases)> Results = new();

    static Diff()
    {
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try
            {
                var lines = Results.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key,-40} cases={kv.Value.Cases,3} maxAbs={kv.Value.MaxAbs:E2} maxErr/tol={kv.Value.MaxScaled:F3}");
                File.WriteAllLines(Path.Combine(AppContext.BaseDirectory, "diff-summary.txt"), lines);
            }
            catch { }
        };
    }

    static CompiledModel Compile(IComputeDevice dev, ComputeGraph g, long maxTokens, long ctx)
        => CompiledModel.Compile(g, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = dev }],
            Compile = new CompileOptions { MaxTokens = maxTokens, ContextLength = ctx, SymbolBounds = new Dictionary<string, long> { ["R"] = maxTokens } },
            ReservePerDevice = 0,
        });

    static RunParameters Params(Step s)
    {
        var p = new RunParameters();
        p.Symbols["T"] = s.T; p.Symbols["R"] = s.R ?? s.T;
        p.Scalars["n_past"] = s.Past;
        return p;
    }

    static void Feed(CompiledModel m, Step s)
    {
        foreach (var (v, data) in s.Inputs)
            switch (data)
            {
                case float[] f: m.SetInput(v, MemoryMarshal.AsBytes(f.AsSpan())); break;
                case int[] i: m.SetInput(v, MemoryMarshal.AsBytes(i.AsSpan())); break;
                case long[] l: m.SetInput(v, MemoryMarshal.AsBytes(l.AsSpan())); break;
                default: throw new ArgumentException("input data must be float[] or int[]");
            }
    }

    /// <summary>
    /// Runs all steps on both devices (state persists across steps) and compares every graph output.
    /// Returns the largest |a-b| seen.
    /// </summary>
    public static double Check(string op, ComputeGraph g, IReadOnlyList<Step> steps, double atol = 1e-4, double rtol = 1e-4, long maxTokens = 64, long ctx = 64, string? what = null)
    {
        Gpu.Require();
        using var vk = Compile(Gpu.Vulkan!, g, maxTokens, ctx);
        CompiledModel cpu;
        try { cpu = Compile(Gpu.Cpu, g, maxTokens, ctx); }
        catch (NotSupportedException e) { Assert.Skip("CPU oracle does not support this graph yet: " + e.Message); return 0; }
        using var _ = cpu;
        double maxAbs = 0, maxScaled = 0;
        for (int si = 0; si < steps.Count; si++)
        {
            var s = steps[si];
            var p = Params(s);
            Feed(vk, s); Feed(cpu, s);
            vk.Run(p); cpu.Run(p);
            foreach (var (name, v) in g.Outputs)
            {
                long n = v.Shape.Elements(p.Symbols);
                if (v.Type == DataType.I32)
                {
                    var a = new int[n]; var b = new int[n];
                    vk.GetOutput<int>(name, a); cpu.GetOutput<int>(name, b);
                    for (long i = 0; i < n; i++)
                        if (a[i] != b[i]) Assert.Fail($"{op} {what} step {si} output {name}[{i}]: vulkan {a[i]} != cpu {b[i]}");
                    continue;
                }
                if (v.Type == DataType.I64)
                {
                    var a = new long[n]; var b = new long[n];
                    vk.GetOutput<long>(name, a); cpu.GetOutput<long>(name, b);
                    for (long i = 0; i < n; i++)
                        if (a[i] != b[i]) Assert.Fail($"{op} {what} step {si} output {name}[{i}]: vulkan {a[i]} != cpu {b[i]}");
                    continue;
                }
                var x = new float[n]; var y = new float[n];
                vk.GetOutput<float>(name, x); cpu.GetOutput<float>(name, y);
                for (long i = 0; i < n; i++)
                {
                    double d = Math.Abs((double)x[i] - y[i]);
                    double tol = atol + rtol * Math.Abs(y[i]);
                    if (double.IsNaN(x[i]) != double.IsNaN(y[i]) || (!double.IsNaN(y[i]) && d > tol))
                        Assert.Fail($"{op} {what} step {si} output {name}[{i}]: vulkan {x[i]:G9} vs cpu {y[i]:G9} (|d|={d:E2}, tol={tol:E2})");
                    if (!double.IsNaN(d)) { maxAbs = Math.Max(maxAbs, d); maxScaled = Math.Max(maxScaled, d / tol); }
                }
            }
        }
        Results.AddOrUpdate(op, (maxAbs, maxScaled, 1), (_, o) => (Math.Max(o.MaxAbs, maxAbs), Math.Max(o.MaxScaled, maxScaled), o.Cases + 1));
        return maxAbs;
    }

    public static double Check(string op, ComputeGraph g, Step step, double atol = 1e-4, double rtol = 1e-4, long maxTokens = 64, long ctx = 64, string? what = null)
        => Check(op, g, [step], atol, rtol, maxTokens, ctx, what);
}
