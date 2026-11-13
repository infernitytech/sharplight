using System.Diagnostics;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Vulkan;

namespace Sharplight.OnnxRun;

/// <summary>Imports an ONNX model, compiles it for one device, runs it repeatedly and reports timings and the first output.</summary>
internal sealed class OnnxRunner
{
    private const long ReservePerDeviceBytes = 256L << 20;

    private readonly OnnxRunSettings _settings;
    private readonly TextWriter _output;

    public OnnxRunner(OnnxRunSettings settings, TextWriter output)
    {
        _settings = settings;
        _output = output;
    }

    public void Run(CancellationToken cancellationToken)
    {
        var graph = Import();
        using var device = OpenDevice();
        using var compiled = Compile(graph, device);

        var run = new RunParameters();
        run.Symbols[OnnxGraphAdapter.BatchSymbol] = _settings.BatchSize;
        new InputBinder().Bind(compiled, graph, run, _settings.InputFile);

        var (outputName, outputValue) = graph.Outputs.First();
        var result = new float[outputValue.Shape.Elements(run.Symbols)];
        var timings = TimeRuns(compiled, run, outputName, result, cancellationToken);
        _output.WriteLine(timings.Summarize(_settings.BatchSize));

        compiled.GetOutput<float>(outputName, result);
        OutputReporter.Report(outputName, outputValue, result, _settings.BatchSize, _settings.OutputFile, _output);
    }

    private ComputeGraph Import()
    {
        var stopwatch = Stopwatch.StartNew();
        var options = new OnnxImportOptions();
        foreach (var (name, shape) in _settings.InputShapes)
        {
            options.InputShapes[name] = shape;
        }

        OnnxImporter.Register();
        if (!File.Exists(_settings.ModelPath))
        {
            throw new SampleException($"Model '{_settings.ModelPath}' does not exist.");
        }

        var source = OnnxImporter.Open(_settings.ModelPath, options);
        var definition = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = source });
        var graph = definition.Graph(OnnxGraphAdapter.MainGraph);
        _output.WriteLine(
            $"imported {Path.GetFileName(_settings.ModelPath)} in {stopwatch.ElapsedMilliseconds} ms: {graph.Nodes.Count} nodes, {definition.Weights.Count} weights, "
            + $"producer {source.Get("onnx.producer_name", "?")}, opset {source.Get("onnx.opset", 0L)}");
        if (_settings.DumpGraph)
        {
            _output.WriteLine(graph.Dump());
        }

        return graph;
    }

    private IComputeDevice OpenDevice()
    {
        IExecutionProvider provider = _settings.Provider == OnnxProvider.Vulkan ? new VulkanProvider() : new CpuExecutionProvider();
        var availability = provider.CheckAvailability();
        if (!availability.IsAvailable)
        {
            throw new SampleException($"{_settings.Provider} unavailable: {availability.Reason}");
        }

        var info = SampleRuntime.GetDevice(provider, _settings.DeviceIndex);
        _output.WriteLine($"device: {info}");
        return provider.OpenDevice(info);
    }

    private CompiledModel Compile(ComputeGraph graph, IComputeDevice device)
    {
        var stopwatch = Stopwatch.StartNew();
        var compileOptions = new CompileOptions
        {
            SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = _settings.BatchSize },
        };
        var compiled = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = compileOptions,
            ReservePerDevice = ReservePerDeviceBytes,
        });
        _output.WriteLine($"compiled in {stopwatch.ElapsedMilliseconds} ms ({compiled.Plan.Segments.Count} segment(s))");
        return compiled;
    }

    private RunTimings TimeRuns(CompiledModel compiled, RunParameters run, string outputName, float[] scratch, CancellationToken cancellationToken)
    {
        compiled.Run(run);

        var timings = new RunTimings();
        for (int i = 0; i < _settings.TimedRuns; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var stopwatch = Stopwatch.StartNew();
            compiled.Run(run);

            // Reading the output forces completion on providers that run asynchronously.
            compiled.GetOutput<float>(outputName, scratch);
            timings.Add(stopwatch.Elapsed.TotalMilliseconds);
        }

        return timings;
    }
}
