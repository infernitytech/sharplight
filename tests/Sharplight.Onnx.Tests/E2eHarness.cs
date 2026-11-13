using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>Providers shared by the end-to-end tests (opened once, lazily).</summary>
public static class Providers
{
    static readonly Lazy<IComputeDevice> s_cpu = new(() => { var p = new CpuExecutionProvider(); return p.OpenDevice(p.GetDevices()[0]); });
    static readonly Lazy<(IComputeDevice? Device, string Reason)> s_vk = new(() =>
    {
        try
        {
            var p = new VulkanProvider();
            var a = p.CheckAvailability();
            if (!a.IsAvailable) return (null, a.Reason);
            var d = p.GetDevices();
            return d.Count == 0 ? (null, "no Vulkan devices") : (p.OpenDevice(d[0]), "");
        }
        catch (Exception e) { return (null, e.Message); }
    });

    public static IComputeDevice Cpu => s_cpu.Value;

    public static IComputeDevice VulkanOrSkip()
    {
        var (d, reason) = s_vk.Value;
        if (d is null) Assert.Skip("Vulkan unavailable: " + reason);
        return d;
    }

    public static IComputeDevice Get(string provider) => provider == "cpu" ? Cpu : VulkanOrSkip();
}

public sealed record RefInput(string Name, long[] Shape);

/// <summary>A reference case produced by .work/onnx_ref/*.py: model + manifest + inputs/outputs from onnxruntime.</summary>
public sealed class RefCase
{
    public required string Name { get; init; }
    public required string ModelPath { get; init; }
    public required string Prefix { get; init; }
    public required int[] Batches { get; init; }
    public required RefInput[] Inputs { get; init; }
    public required string[] Outputs { get; init; }
    public double Rtol { get; init; } = 1e-3;

    public static string RefRoot => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx", "ref");
    public static string SynthRoot => Path.Combine(RefRoot, "synth");

    public static RefCase? TryLoad(string manifestPath, string modelPath)
    {
        if (!File.Exists(manifestPath) || !File.Exists(modelPath)) return null;
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var r = doc.RootElement;
        return new RefCase
        {
            Name = Path.GetFileNameWithoutExtension(manifestPath),
            ModelPath = modelPath,
            Prefix = manifestPath[..^".json".Length],
            Batches = r.GetProperty("batches").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            Inputs = r.GetProperty("inputs").EnumerateArray().Select(e => new RefInput(e.GetProperty("name").GetString()!, e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray())).ToArray(),
            Outputs = r.GetProperty("outputs").EnumerateArray().Select(e => e.GetProperty("name").GetString()!).ToArray(),
            Rtol = r.TryGetProperty("rtol", out var rt) ? rt.GetDouble() : 1e-3,
        };
    }

    public static RefCase? Synthetic(string name) => TryLoad(Path.Combine(SynthRoot, name + ".json"), Path.Combine(SynthRoot, name + ".onnx"));
    public static RefCase? Model(string name, string? onnxFile = null) => TryLoad(Path.Combine(RefRoot, name + ".json"), Path.Combine(Path.GetDirectoryName(RefRoot)!, (onnxFile ?? name) + ".onnx"));

    public static IEnumerable<string> SyntheticNames()
        => Directory.Exists(SynthRoot) ? Directory.GetFiles(SynthRoot, "*.json").Select(f => Path.GetFileNameWithoutExtension(f)!).OrderBy(x => x) : [];

    public float[] ReadInput(int batch, int k) => ReadF32($"{Prefix}.b{batch}.in{k}.bin");
    public float[] ReadOutput(int batch, int k) => ReadF32($"{Prefix}.b{batch}.out{k}.bin");

    static float[] ReadF32(string path)
    {
        var b = File.ReadAllBytes(path);
        var f = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, f, 0, b.Length);
        return f;
    }
}

public readonly record struct Comparison(double MaxAbs, double MaxRel, double Scale, bool Ok, int Worst);

public static class Compare
{
    /// <summary>|a-e| &lt;= atol + rtol*|e| with atol scaled by the output magnitude.</summary>
    public static Comparison Check(float[] actual, float[] expected, double rtol)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1.0, expected.Select(Math.Abs).DefaultIfEmpty(0).Max());
        double atol = 2e-4 * scale;
        double maxAbs = 0, maxRel = 0; bool ok = true; int worst = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)actual[i] - expected[i]);
            if (double.IsNaN(actual[i])) { ok = false; worst = i; maxAbs = double.NaN; break; }
            if (d > maxAbs) { maxAbs = d; worst = i; }
            maxRel = Math.Max(maxRel, d / Math.Max(Math.Abs(expected[i]), 1e-2 * scale));
            if (d > atol + rtol * Math.Abs(expected[i])) ok = false;
        }
        return new Comparison(maxAbs, maxRel, scale, ok, worst);
    }

    public static int ArgMax(ReadOnlySpan<float> v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }
}

/// <summary>Imports, compiles and runs an ONNX file on a provider.</summary>
public static class OnnxExec
{
    public sealed record Result(float[][] Outputs, double ImportMs, double CompileMs, double RunMs, ComputeGraph Graph);

    public static ComputeGraph Import(string path, OnnxImportOptions? options, out ModelDefinition def)
    {
        var src = OnnxImporter.Open(path, options);
        def = OnnxGraphAdapter.Build(src);
        return def.Graph(OnnxGraphAdapter.MainGraph);
    }

    public static Result Run(string path, OnnxImportOptions? options, IComputeDevice device, int batch, IReadOnlyList<float[]> inputs, int timedRuns = 1)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var graph = Import(path, options, out var def);
        using var _ = new Disposer(def.OwnedResources);
        double importMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        using var compiled = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = batch } },
            ReservePerDevice = 128L << 20,
        });
        double compileMs = sw.Elapsed.TotalMilliseconds;
        var run = new RunParameters();
        run.Symbols[OnnxGraphAdapter.BatchSymbol] = batch;
        for (int i = 0; i < graph.Inputs.Count; i++)
        {
            if (graph.Inputs[i].Type == Sharplight.Tensors.DataType.I32) compiled.SetInput<int>(graph.Inputs[i], inputs[i].Select(v => (int)v).ToArray());
            else compiled.SetInput<float>(graph.Inputs[i], inputs[i]);
        }
        compiled.Run(run);
        var outs = ReadOutputs(compiled, graph, run);   // also synchronizes
        double best = double.MaxValue;
        for (int r = 0; r < timedRuns - 1 || (timedRuns == 1 && r < 0); r++) { }
        if (timedRuns > 1)
            for (int r = 0; r < timedRuns; r++)
            {
                sw.Restart(); compiled.Run(run); ReadOutputs(compiled, graph, run);
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
        return new Result(outs, importMs, compileMs, best == double.MaxValue ? 0 : best, graph);
    }

    static float[][] ReadOutputs(CompiledModel compiled, ComputeGraph graph, RunParameters run)
    {
        var outs = new List<float[]>();
        foreach (var (name, v) in graph.Outputs)
        {
            var buf = new float[v.Shape.Elements(run.Symbols)];
            if (v.Type == Sharplight.Tensors.DataType.I32) { var ib = new int[buf.Length]; compiled.GetOutput<int>(name, ib); for (int q = 0; q < ib.Length; q++) buf[q] = ib[q]; }
            else compiled.GetOutput<float>(name, buf);
            outs.Add(buf);
        }
        return [.. outs];
    }

    sealed class Disposer(IReadOnlyList<IDisposable> items) : IDisposable
    {
        public void Dispose() { foreach (var i in items) i.Dispose(); }
    }

    public static OnnxImportOptions OptionsFor(RefCase c, int batch = 1)
    {
        var o = new OnnxImportOptions();
        // whisper-style attention reshapes [N,T,D] -> [N*heads,T,dh]: the batch cannot stay symbolic, so fix it.
        foreach (var i in c.Inputs) o.InputShapes[i.Name] = c.Name.StartsWith("whisper") ? i.Shape.Select(d => d < 0 ? batch : d).ToArray() : i.Shape;
        return o;
    }
}
