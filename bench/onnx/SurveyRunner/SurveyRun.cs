using System.Diagnostics;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Import.Onnx;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

/// <summary>One survey: import, compile, run, compare and time a model, recording each stage in the result dictionary.</summary>
internal sealed class SurveyRun
{
    private readonly SurveyOptions _options;
    private readonly Dictionary<string, object?> _result;
    private readonly SurveyManifest _manifest;
    private readonly int _batch;

    /// <summary>Prepares a survey of the model named by the manifest of <paramref name="options"/>.</summary>
    public SurveyRun(SurveyOptions options, Dictionary<string, object?> result)
    {
        _options = options;
        _result = result;
        _manifest = SurveyManifest.Load(options.ManifestPath);
        _batch = options.BatchOverride ?? _manifest.Batch;
    }

    /// <summary>Shortens <paramref name="text"/> to at most <paramref name="length"/> characters.</summary>
    public static string Truncate(string? text, int length = 600)
    {
        if (text is null)
        {
            return string.Empty;
        }

        return text.Length > length ? text[..length] + "..." : text;
    }

    /// <summary>Runs the survey; returns the process exit code (0 ok, 2 provider unavailable).</summary>
    public int Execute()
    {
        _result["model"] = _manifest.ModelPath;
        _result["batch"] = _batch;
        _result["modelMB"] = new FileInfo(_manifest.ModelPath).Length >> 20;

        IExecutionProvider provider = _options.ProviderName == "cpu" ? new CpuExecutionProvider() : new VulkanProvider();
        var availability = provider.CheckAvailability();
        if (!availability.IsAvailable)
        {
            _result["stage"] = "provider";
            _result["error"] = availability.Reason;
            return 2;
        }

        var deviceInfo = provider.GetDevices()[_options.DeviceIndex];
        using var device = provider.OpenDevice(deviceInfo);
        _result["device"] = deviceInfo.ToString();

        var (definition, graph, compiled) = ImportAndCompile(device);
        using var compiledScope = compiled;
        var run = new RunParameters();
        run.Symbols[OnnxGraphAdapter.BatchSymbol] = _batch;
        _result["stage"] = "run";
        FeedInputs(graph, compiled, run);
        compiled.Run(run);
        var outputs = ReadOutputs(graph, compiled, run);
        DumpOutputs(outputs);
        Compare(outputs);
        Time(graph, compiled, run);
        if (_options.Profile && _options.ProviderName == "vulkan")
        {
            Profile(compiled, run);
        }

        _result["stage"] = "ok";
        foreach (var resource in definition.OwnedResources)
        {
            resource.Dispose();
        }

        return 0;
    }

    /// <summary>Imports and compiles with a symbolic batch, falling back to a fixed batch if that fails.</summary>
    private (ModelDefinition Definition, ComputeGraph Graph, CompiledModel Compiled) ImportAndCompile(IDevice device)
    {
        foreach (bool fixBatch in _manifest.FixedBatch ? new[] { true } : new[] { false, true })
        {
            try
            {
                var attempt = ImportAndCompile(device, fixBatch);
                _result["fixedBatch"] = fixBatch;
                if (fixBatch && !_manifest.FixedBatch)
                {
                    _result["note"] = "symbolic batch failed (" + Truncate((string?)_result.GetValueOrDefault("symbolicError")) + "); fixed batch used";
                }

                return attempt;
            }
            catch (Exception e) when (!fixBatch && !_manifest.FixedBatch)
            {
                _result["symbolicError"] = Truncate(e.Message);
            }
        }

        throw new InvalidOperationException("import failed");
    }

    private (ModelDefinition Definition, ComputeGraph Graph, CompiledModel Compiled) ImportAndCompile(IDevice device, bool fixBatch)
    {
        var importOptions = new OnnxImportOptions();
        foreach (var (name, values) in _manifest.Constants)
        {
            importOptions.ConstantInputs[name] = values;
        }

        foreach (var (name, bound) in _options.Bounds())
        {
            importOptions.InputValueBounds[name] = bound;
        }

        foreach (var (name, shape) in _manifest.Inputs)
        {
            importOptions.InputShapes[name] = fixBatch ? shape.Select(d => d < 0 ? _batch : d).ToArray() : shape;
        }

        var stopwatch = Stopwatch.StartNew();
        _result["stage"] = "import";
        var source = OnnxImporter.Open(_manifest.ModelPath, importOptions);
        var definition = OnnxGraphAdapter.Build(source);
        var graph = definition.Graph(OnnxGraphAdapter.MainGraph);
        _result["importMs"] = Math.Round(stopwatch.Elapsed.TotalMilliseconds);
        _result["nodes"] = graph.Nodes.Count;
        _result["opset"] = source.Get("onnx.opset", 0L);

        stopwatch.Restart();
        _result["stage"] = "compile";
        var compiled = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions
            {
                SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = _batch },
                AllowReducedPrecision = _options.ReducedPrecision,
            },
            ReservePerDevice = 256L << 20,
        });
        _result["compileMs"] = Math.Round(stopwatch.Elapsed.TotalMilliseconds);
        _result["segments"] = compiled.Plan.Segments.Count;
        return (definition, graph, compiled);
    }

    /// <summary>Sets every graph input; the manifest order includes pinned constants (not graph inputs), the bins are indexed by manifest position.</summary>
    private void FeedInputs(ComputeGraph graph, CompiledModel compiled, RunParameters run)
    {
        var inputs = _manifest.Inputs;
        for (int index = 0; index < graph.Inputs.Count; index++)
        {
            var input = graph.Inputs[index];
            int position = Array.FindIndex(inputs, x => x.Name == input.Name);
            if (position < 0)
            {
                var runtimeInputs = Enumerable.Range(0, inputs.Length).Where(q => !_manifest.Constants.ContainsKey(inputs[q].Name)).ToArray();
                position = runtimeInputs[index];
            }

            byte[] bytes = TensorCodec.FromF32(TensorCodec.ReadF32($"{_manifest.Prefix}.b{_batch}.in{position}.bin"), input.Type);
            long wanted = input.Type.ByteSize(input.Shape.Elements(run.Symbols));
            if (bytes.Length != wanted)
            {
                throw new InvalidOperationException($"input '{input.Name}' {input.Type} {input.Shape}: have {bytes.Length} bytes, graph wants {wanted}");
            }

            compiled.SetInput(input, bytes);
        }
    }

    private static List<(string Name, float[] Data)> ReadOutputs(ComputeGraph graph, CompiledModel compiled, RunParameters run)
    {
        var outputs = new List<(string Name, float[] Data)>();
        foreach (var (name, value) in graph.Outputs)
        {
            outputs.Add((name, TensorCodec.ReadOutput(compiled, name, value, run)));
        }

        return outputs;
    }

    /// <summary>Writes every graph output as raw f32 (<c>DIR/k.bin</c>, k = graph output index) for offline comparison.</summary>
    private void DumpOutputs(List<(string Name, float[] Data)> outputs)
    {
        if (_options.DumpDirectory is not { } directory)
        {
            return;
        }

        Directory.CreateDirectory(directory);
        for (int k = 0; k < outputs.Count; k++)
        {
            TensorCodec.WriteF32(Path.Combine(directory, $"{k}.bin"), outputs[k].Data);
        }
    }

    private void Compare(List<(string Name, float[] Data)> outputs)
    {
        var comparisons = new List<object>();
        bool allOk = true;
        double worstRms = 0;
        var names = _manifest.OutputNames;
        for (int k = 0; k < names.Length; k++)
        {
            int index = outputs.FindIndex(o => o.Name == names[k]);
            if (index < 0)
            {
                index = k;
            }

            if (index >= outputs.Count)
            {
                comparisons.Add(new { name = names[k], error = "output missing" });
                allOk = false;
                continue;
            }

            var expected = TensorCodec.ReadF32($"{_manifest.Prefix}.b{_batch}.out{k}.bin");
            var actual = outputs[index].Data;
            if (_manifest.Mode == "scores6" && expected.Length % 6 == 0 && actual.Length == expected.Length)
            {
                expected = ScoreColumn(expected);
                actual = ScoreColumn(actual);
            }

            if (expected.Length != actual.Length)
            {
                comparisons.Add(new { name = names[k], error = $"size {actual.Length} vs ref {expected.Length}" });
                allOk = false;
                continue;
            }

            var check = OutputCheck.Compare(actual, expected, _manifest.RelativeTolerance, _manifest.AbsoluteTolerance);
            bool ok = _manifest.RmsTolerance is { } tolerance ? !double.IsNaN(check.MaxAbs) && check.Rms <= tolerance : check.Ok;
            allOk &= ok;
            worstRms = Math.Max(worstRms, double.IsNaN(check.Rms) ? 1e9 : check.Rms);
            comparisons.Add(new { name = names[k], n = expected.Length, maxAbs = check.MaxAbs, maxRel = check.MaxRel, rms = check.Rms, scale = check.Scale, ok });
        }

        _result["outputs"] = comparisons;
        _result["parityOk"] = allOk;
        _result["worstRms"] = worstRms;
    }

    /// <summary>The score column (index 4) of rows of six values.</summary>
    private static float[] ScoreColumn(float[] rows)
    {
        return Enumerable.Range(0, rows.Length / 6).Select(row => rows[row * 6 + 4]).ToArray();
    }

    /// <summary>Times Run plus reading back every output (which forces completion): minimum and median over the runs.</summary>
    private void Time(ComputeGraph graph, CompiledModel compiled, RunParameters run)
    {
        var times = new List<double>();
        var total = Stopwatch.StartNew();
        var buffers = graph.Outputs.ToDictionary(o => o.Key, o => new byte[(int)o.Value.Type.ByteSize(o.Value.Shape.Elements(run.Symbols))]);
        var stopwatch = new Stopwatch();
        while (times.Count < _options.MaxRuns && (times.Count < 3 || total.Elapsed.TotalSeconds < _options.BudgetSeconds))
        {
            stopwatch.Restart();
            compiled.Run(run);
            foreach (var (name, buffer) in buffers)
            {
                compiled.GetOutput(name, buffer);
            }

            times.Add(stopwatch.Elapsed.TotalMilliseconds);
        }

        _result["minMs"] = Math.Round(times.Min(), 3);
        _result["medMs"] = Math.Round(times.OrderBy(x => x).ElementAt(times.Count / 2), 3);
        _result["runs"] = times.Count;
    }

    private void Profile(CompiledModel compiled, RunParameters run)
    {
        foreach (var segment in compiled.Segments.OfType<VulkanSegment>())
        {
            segment.Profile = true;
            segment.ResetProfile();
        }

        for (int i = 0; i < 3; i++)
        {
            compiled.Run(run);
        }

        _result["profile"] = string.Join("\n", compiled.Segments.OfType<VulkanSegment>().Select(segment => segment.ProfileReport(12)));
    }
}
