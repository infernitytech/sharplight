using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;

namespace Sharplight.Onnx.Tests;

/// <summary>Imports (or opens a package converted from), compiles and runs an ONNX model on a provider.</summary>
internal static class OnnxExec
{
    public static ComputeGraph Import(string path, OnnxImportOptions? options, out ModelDefinition def)
    {
        var src = OnnxImporter.Open(path, options);
        def = OnnxGraphAdapter.Build(src);
        return def.Graph(OnnxGraphAdapter.MainGraph);
    }

    /// <summary>Runs an ONNX file or a package (.slm) converted from one.</summary>
    /// <param name="path">The ONNX file or package.</param>
    /// <param name="options">Import options of an ONNX file (a package stores its shapes).</param>
    /// <param name="device">The device to run on.</param>
    /// <param name="batch">Value of the batch symbol.</param>
    /// <param name="inputs">One array per runtime input.</param>
    /// <param name="timedRuns">Runs to time after the first.</param>
    /// <param name="artifacts">Compilation artifacts to compile with and record into; null = those stored in a package (none for an ONNX file).</param>
    public static OnnxExecResult Run(string path, OnnxImportOptions? options, IComputeDevice device, int batch, IReadOnlyList<float[]> inputs, int timedRuns = 1,
        CompilationArtifacts? artifacts = null)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        ComputeGraph graph;
        ModelDefinition def;
        if (ModelPackage.IsPackage(path))
        {
            var source = new ModelPackageImporter().Open(path);
            def = source.Definition!;
            graph = def.Graph(OnnxGraphAdapter.MainGraph);
            artifacts ??= new CompilationArtifacts(source.CompilationCaches);
        }
        else
        {
            graph = Import(path, options, out def);
        }

        using var _ = new Disposer(def.OwnedResources);
        double importMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        using var compiled = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Fallback = device.Info.Kind == DeviceKind.Cpu ? null : new DeviceSlot { Device = Providers.Cpu },   // data-dependent ops run on CPU
            Compile = new CompileOptions
            {
                SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = batch },
                CacheFor = artifacts is null ? null : artifacts.For,
            },
            ReservePerDevice = 128L << 20,
        });
        double compileMs = sw.Elapsed.TotalMilliseconds;
        var run = new RunParameters();
        run.Symbols[OnnxGraphAdapter.BatchSymbol] = batch;
        for (int i = 0; i < graph.Inputs.Count; i++)
        {
            if (graph.Inputs[i].Type == Sharplight.Tensors.DataType.I32)
            {
                compiled.SetInput<int>(graph.Inputs[i], inputs[i].Select(v => (int)v).ToArray());
            }
            else
            {
                compiled.SetInput<float>(graph.Inputs[i], inputs[i]);
            }
        }
        compiled.Run(run);
        var outs = ReadOutputs(compiled, graph, run);   // also synchronizes
        double best = double.MaxValue;
        for (int r = 0; r < timedRuns - 1 || (timedRuns == 1 && r < 0); r++)
        {
        }
        if (timedRuns > 1)
        {
            for (int r = 0; r < timedRuns; r++)
            {
                sw.Restart();
                compiled.Run(run);
                ReadOutputs(compiled, graph, run);
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
        }

        return new OnnxExecResult(outs, importMs, compileMs, best == double.MaxValue ? 0 : best, graph, artifacts);
    }

    private static float[][] ReadOutputs(CompiledModel compiled, ComputeGraph graph, RunParameters run)
    {
        var outs = new List<float[]>();
        foreach (var (name, v) in graph.Outputs)
        {
            var buf = new float[compiled.OutputShape(name, run).Aggregate(1L, (a, b) => a * b)];   // data-dependent sizes are known after the run
            if (v.Type == Sharplight.Tensors.DataType.I32)
            {
                var ib = new int[buf.Length];
                compiled.GetOutput<int>(name, ib);
                for (int q = 0; q < ib.Length; q++)
                {
                    buf[q] = ib[q];
                }
            }
            else
            {
                compiled.GetOutput<float>(name, buf);
            }

            outs.Add(buf);
        }
        return [.. outs];
    }

    private sealed class Disposer(IReadOnlyList<IDisposable> items) : IDisposable
    {
        public void Dispose()
        {
            foreach (var i in items)
            {
                i.Dispose();
            }
        }
    }

    public static OnnxImportOptions OptionsFor(RefCase c, int batch = 1)
    {
        var o = new OnnxImportOptions();
        // whisper-style attention reshapes [N,T,D] -> [N*heads,T,dh]: the batch cannot stay symbolic, so fix it.
        foreach (var (k, v) in c.Constants)
        {
            o.ConstantInputs[k] = v;
        }

        foreach (var i in c.Inputs)
        {
            o.InputShapes[i.Name] = c.FixBatch || c.Name.StartsWith("whisper") ? i.Shape.Select(d => d < 0 ? batch : d).ToArray() : i.Shape;
        }

        return o;
    }
}
