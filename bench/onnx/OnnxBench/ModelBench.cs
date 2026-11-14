using System.Diagnostics;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace OnnxBench;

/// <summary>One model on one provider: import, compile, parity against the reference, timing, optional profile.</summary>
internal sealed class ModelBench(BenchArguments arguments, ReferenceManifest manifest, Dictionary<string, object?> result)
{
    private const int MinRuns = 5;

    public int Run()
    {
        result["stage"] = "start";
        try
        {
            return RunCore();
        }
        catch (Exception e)
        {
            result["error"] = Truncate(e.GetType().Name + ": " + e.Message, 1500);
            return 1;
        }
    }

    private int RunCore()
    {
        int batch = arguments.Batch ?? manifest.Batches[0];
        result["model"] = manifest.Name;
        result["batch"] = batch;
        result["reducedPrecision"] = arguments.ReducedPrecision;

        IExecutionProvider provider = arguments.Provider == "cpu" ? new CpuExecutionProvider() : new VulkanProvider();
        var availability = provider.CheckAvailability();
        if (!availability.IsAvailable)
        {
            result["stage"] = "provider";
            result["error"] = availability.Reason;
            return 2;
        }

        var deviceInfo = provider.GetDevices()[arguments.Device];
        using var device = provider.OpenDevice(deviceInfo);
        result["device"] = deviceInfo.ToString();
        // Ops without a GPU kernel (data-dependent ones) run on the CPU, as in the ONNX tests.
        var cpu = new CpuExecutionProvider();
        using var fallback = arguments.Provider == "cpu" ? null : cpu.OpenDevice(cpu.GetDevices()[0]);

        var (definition, graph, compiled) = Compile(device, fallback, batch);
        using var compiledModel = compiled;
        var run = new RunParameters();
        run.Symbols[OnnxGraphAdapter.BatchSymbol] = batch;

        result["stage"] = "run";
        SetInputs(compiledModel, graph, run, batch);
        compiledModel.Run(run);
        CheckParity(compiledModel, graph, run, batch);
        Time(compiledModel, graph, run);
        if (arguments.Profile)
        {
            Profile(compiledModel, run);
        }

        result["stage"] = "ok";
        foreach (var resource in definition.OwnedResources)
        {
            resource.Dispose();
        }

        return 0;
    }

    /// <summary>Imports with a symbolic batch first and falls back to a fixed batch for graphs that reshape across it.</summary>
    private (ModelDefinition, ComputeGraph, CompiledModel) Compile(IComputeDevice device, IComputeDevice? fallback, int batch)
    {
        bool[] attempts = manifest.FixBatch ? [true] : [false, true];
        Exception? last = null;
        foreach (bool fixBatch in attempts)
        {
            CompiledModel? compiled = null;
            try
            {
                var options = new OnnxImportOptions { AllowHalfBlockScales = arguments.ReducedPrecision };
                foreach (var (name, values) in manifest.Constants)
                {
                    options.ConstantInputs[name] = values;
                }

                foreach (var (name, shape) in manifest.Inputs)
                {
                    options.InputShapes[name] = fixBatch ? shape.Select(d => d < 0 ? batch : d).ToArray() : shape;
                }

                result["stage"] = "import";
                var watch = Stopwatch.StartNew();
                var source = OnnxImporter.Open(manifest.ModelPath, options);
                var definition = OnnxGraphAdapter.Build(source);
                var graph = definition.Graph(OnnxGraphAdapter.MainGraph);
                result["importMs"] = Math.Round(watch.Elapsed.TotalMilliseconds);
                result["nodes"] = graph.Nodes.Count;

                result["stage"] = "compile";
                watch.Restart();
                compiled = CompiledModel.Compile(graph, new PlacementOptions
                {
                    Devices = [new DeviceSlot { Device = device }],
                    Fallback = fallback is null ? null : new DeviceSlot { Device = fallback },
                    Compile = new CompileOptions
                    {
                        SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = batch },
                        AllowReducedPrecision = arguments.ReducedPrecision,
                    },
                    ReservePerDevice = 256L << 20,
                });
                result["compileMs"] = Math.Round(watch.Elapsed.TotalMilliseconds);
                result["segments"] = compiled.Plan.Segments.Count;
                result["fallbackSegments"] = compiled.Plan.Segments.Count(s => s.DeviceSlot != 0);
                result["fixedBatch"] = fixBatch;
                return (definition, graph, compiled);
            }
            catch (Exception e) when (!fixBatch)
            {
                compiled?.Dispose();
                last = e;
                result["symbolicError"] = Truncate(e.Message);
            }
        }

        throw last ?? new InvalidOperationException("no compile attempt");
    }

    /// <summary>Feeds the reference inputs. The manifest lists pinned constants too, and the files are indexed by manifest position.</summary>
    private void SetInputs(CompiledModel compiled, ComputeGraph graph, RunParameters run, int batch)
    {
        var runtimeInputs = Enumerable.Range(0, manifest.Inputs.Length).Where(k => !manifest.Constants.ContainsKey(manifest.Inputs[k].Name)).ToArray();
        for (int gi = 0; gi < graph.Inputs.Count; gi++)
        {
            var value = graph.Inputs[gi];
            int k = Array.FindIndex(manifest.Inputs, x => x.Name == value.Name);
            if (k < 0)
            {
                k = runtimeInputs[gi];
            }

            var data = manifest.ReadData("in", k, batch) ?? throw new FileNotFoundException($"no reference input {k} for '{value.Name}'");
            var bytes = TensorCodec.Encode(data, value.Type);
            long want = value.Type.ByteSize(value.Shape.Elements(run.Symbols));
            if (bytes.Length != want)
            {
                throw new InvalidOperationException($"input '{value.Name}' {value.Type} {value.Shape}: have {bytes.Length} bytes, graph wants {want}");
            }

            compiled.SetInput(value, bytes);
        }
    }

    private void CheckParity(CompiledModel compiled, ComputeGraph graph, RunParameters run, int batch)
    {
        var outputs = graph.Outputs.Select(o => (o.Key, Data: TensorCodec.ReadOutput(compiled, o.Key, o.Value, run))).ToList();
        var comparisons = new List<object>();
        bool allOk = true;
        double worstRms = 0;
        for (int k = 0; k < manifest.Outputs.Length; k++)
        {
            int gi = outputs.FindIndex(o => o.Key == manifest.Outputs[k]);
            gi = gi < 0 ? k : gi;
            var expected = manifest.ReadData("out", k, batch);
            if (gi >= outputs.Count || expected is null)
            {
                comparisons.Add(new { name = manifest.Outputs[k], error = "output or reference missing" });
                allOk = false;
                continue;
            }

            var actual = outputs[gi].Data;
            if (manifest.ScoresOnly)
            {
                expected = OutputComparison.ScoreColumn(expected);
                actual = OutputComparison.ScoreColumn(actual);
            }

            if (expected.Length != actual.Length)
            {
                comparisons.Add(new { name = manifest.Outputs[k], error = $"size {actual.Length} vs reference {expected.Length}" });
                allOk = false;
                continue;
            }

            var c = OutputComparison.Compare(manifest.Outputs[k], actual, expected, manifest);
            allOk &= c.Ok;
            worstRms = Math.Max(worstRms, double.IsNaN(c.Rms) ? 1e9 : c.Rms);
            comparisons.Add(c);
        }

        result["outputs"] = comparisons;
        result["parityOk"] = allOk;
        result["worstRms"] = worstRms;
    }

    /// <summary>Run plus a read-back of every output (forces completion); min / median wall time and min GPU time.</summary>
    private void Time(CompiledModel compiled, ComputeGraph graph, RunParameters run)
    {
        var buffers = graph.Outputs.Select(o => (o.Key, Buffer: new byte[(int)o.Value.Type.ByteSize(o.Value.Shape.Elements(run.Symbols))])).ToArray();
        var segments = compiled.Segments.OfType<VulkanSegment>().ToArray();
        var wall = new List<double>();
        var gpu = new List<double>();
        var total = Stopwatch.StartNew();
        var watch = new Stopwatch();
        while (wall.Count < arguments.Runs && (wall.Count < MinRuns || total.Elapsed.TotalSeconds < arguments.BudgetSeconds))
        {
            watch.Restart();
            compiled.Run(run);
            foreach (var (name, buffer) in buffers)
            {
                compiled.GetOutput(name, buffer);
            }

            wall.Add(watch.Elapsed.TotalMilliseconds);
            if (segments.Length > 0)
            {
                gpu.Add(segments.Sum(s => s.LastGpuMs));
            }
        }

        wall.Sort();
        result["minMs"] = Math.Round(wall[0], 3);
        result["medMs"] = Math.Round(wall[wall.Count / 2], 3);
        result["runs"] = wall.Count;
        if (gpu.Count > 0)
        {
            gpu.Sort();
            result["gpuMinMs"] = Math.Round(gpu[0], 3);
            result["gpuMedMs"] = Math.Round(gpu[gpu.Count / 2], 3);
        }
    }

    private void Profile(CompiledModel compiled, RunParameters run)
    {
        var segments = compiled.Segments.OfType<VulkanSegment>().ToArray();
        foreach (var segment in segments)
        {
            segment.Profile = true;
            segment.ResetProfile();
        }

        for (int i = 0; i < 3; i++)
        {
            compiled.Run(run);
        }

        result["profile"] = string.Join("\n", segments.Select(s => s.ProfileReport(40)));
    }

    private static string Truncate(string? s, int n = 600) => s is null ? "" : s.Length > n ? s[..n] + "..." : s;
}
