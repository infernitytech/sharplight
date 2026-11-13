using System.Text;
using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Onnx.Conformance;

/// <summary>
/// Runs one ONNX backend node test (model.onnx + test_data_set_*) on a device: imports the model with the input shapes of the data set,
/// compiles it, feeds the inputs, and compares every output with the expected data.
/// </summary>
/// <remarks>
/// Sharplight's IR erases some ONNX element types (int64 travels as i32, double and float16 compute as f32, bool as 0/1), so values are
/// compared numerically: integer and bool outputs exactly, float outputs with <c>|a - e| &lt;= atol + rtol·|e|</c>. A scalar and a
/// one-element vector are the same shape.
/// </remarks>
internal static class NodeTestRunner
{
    public const double Rtol = 1e-3;
    public const double Atol = 1e-5;

    public static NodeTestOutcome Run(string testDirectory, IComputeDevice device, IComputeDevice? fallback)
    {
        string modelPath = Path.Combine(testDirectory, "model.onnx");
        var sets = Directory.GetDirectories(testDirectory, "test_data_set_*").OrderBy(d => d, StringComparer.Ordinal).ToList();
        var result = NodeTestOutcome.Passed;
        foreach (var set in sets)
        {
            var outcome = RunDataSet(modelPath, set, device, fallback);
            if (outcome.Status != NodeTestStatus.Pass)
            {
                return sets.Count > 1 ? outcome with { Message = $"{Path.GetFileName(set)}: {outcome.Message}" } : outcome;
            }

            result = outcome.Pinned.Length > 0 ? outcome : result;
        }

        return result;
    }

    private static NodeTestOutcome RunDataSet(string modelPath, string setDirectory, IComputeDevice device, IComputeDevice? fallback)
    {
        string[] pinnedNames = [];
        var bytes = File.ReadAllBytes(modelPath);
        OnnxModel model;
        try
        {
            model = OnnxParser.Parse(bytes);
        }
        catch (Exception e)
        {
            return new(NodeTestStatus.ImportFail, "parse: " + e.Message);
        }

        var initializers = model.Graph.Initializers.Select(t => t.Name).ToHashSet(StringComparer.Ordinal);
        var inputInfos = model.Graph.Inputs.Where(i => !initializers.Contains(i.Name)).ToList();
        var outputInfos = model.Graph.Outputs;
        if (inputInfos.Concat(outputInfos).FirstOrDefault(i => i.Kind == OnnxValueKind.Map || (i.Kind != OnnxValueKind.Tensor && i.Innermost is null)) is { } map)
        {
            return new(NodeTestStatus.HarnessUnsupported, $"'{map.Name}' is a map or a sequence/optional without a tensor element type");
        }

        if (inputInfos.Concat(outputInfos).FirstOrDefault(i => i.Innermost?.ElemType == OnnxType.String) is { } stringValue)
        {
            return new(NodeTestStatus.HarnessUnsupported, $"'{stringValue.Name}' is a string tensor");
        }

        var values = new Dictionary<string, TestValue>(StringComparer.Ordinal);
        var expected = new List<TestValue>();
        try
        {
            for (int k = 0; File.Exists(Path.Combine(setDirectory, $"input_{k}.pb")); k++)
            {
                values[inputInfos[k].Name] = TestValue.Load(Path.Combine(setDirectory, $"input_{k}.pb"), inputInfos[k]);
            }

            for (int k = 0; File.Exists(Path.Combine(setDirectory, $"output_{k}.pb")); k++)
            {
                expected.Add(TestValue.Load(Path.Combine(setDirectory, $"output_{k}.pb"), outputInfos[k]));
            }
        }
        catch (Exception e) when (e is NotSupportedException or InvalidDataException)
        {
            return new(NodeTestStatus.HarnessUnsupported, e.Message);
        }

        // Sequence inputs are fed element by element (OnnxGraphAdapter.SequenceElementName); empty optional inputs are absent.
        var inputs = new Dictionary<string, TestTensor>(StringComparer.Ordinal);
        var owner = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (name, value) in values)
        {
            var tensors = value.Elements?.Select((e, k) => (OnnxGraphAdapter.SequenceElementName(name, k), e))
                ?? (value.Tensor is { } t ? [(name, t)] : []);
            foreach (var (tensorName, tensor) in tensors)
            {
                inputs[tensorName] = tensor;
                owner[tensorName] = name;
            }
        }

        // Sharplight compiles static shapes: inputs that define shapes, axes or other import-time parameters (Reshape targets, reduction
        // axes, pads, ...) are passed as tensors by the node tests but are initializers in real models. When the importer needs them as
        // constants, the test is retried with them pinned (OnnxImportOptions.ConstantInputs), first the integer inputs, then every input
        // but the first (float scales, bounds and weights).
        ComputeGraph? graph = null;
        string importError = "";
        int pinLevel = 0;
        for (; pinLevel <= 2 && graph is null; pinLevel++)
        {
            var pinned = inputs.Where(p => pinLevel switch
            {
                0 => false,
                1 => !p.Value.IsFloat,
                _ => owner[p.Key] != inputInfos[0].Name,
            }).ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
            if (pinLevel > 0 && (pinned.Count == 0 || pinned.Count == inputs.Count))
            {
                continue;
            }

            var options = new OnnxImportOptions();
            foreach (var (name, value) in values)
            {
                if (value.IsNone)
                {
                    options.AbsentInputs.Add(name);
                }
                else if (value.Elements is { } elements)
                {
                    options.SequenceInputLengths[name] = elements.Count;
                }
            }

            foreach (var (name, tensor) in inputs)
            {
                options.InputShapes[name] = tensor.Dims;
                if (pinned.ContainsKey(name))
                {
                    options.ConstantInputs[name] = tensor.Values;
                }
            }

            var attempt = OnnxParser.Parse(bytes);
            attempt.Options = options;
            try
            {
                var definition = OnnxGraphAdapter.Build(OnnxImporter.FromModel(attempt, modelPath));
                graph = definition.Graph(OnnxGraphAdapter.MainGraph);
                pinnedNames = [.. pinned.Keys];
            }
            catch (Exception e)
            {
                if (pinLevel == 0)
                {
                    importError = Describe(e);
                }

                if (!NeedsConstant(e.Message))
                {
                    break;
                }
            }
        }

        if (graph is null)
        {
            return new(NodeTestStatus.ImportFail, importError);
        }

        CompiledModel compiled;
        try
        {
            compiled = CompiledModel.Compile(graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Fallback = fallback is null ? null : new DeviceSlot { Device = fallback },
                Compile = new CompileOptions { SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = 1 } },
                ReservePerDevice = 16L << 20,
            });
        }
        catch (Exception e)
        {
            return new(NodeTestStatus.CompileFail, Describe(e));
        }

        using (compiled)
        {
            var run = new RunParameters();
            run.Symbols[OnnxGraphAdapter.BatchSymbol] = 1;
            var actual = new Dictionary<string, (long[] Shape, double[] Values)>(StringComparer.Ordinal);
            try
            {
                foreach (var input in graph.Inputs)
                {
                    if (!inputs.TryGetValue(input.Name, out var tensor))
                    {
                        return new(NodeTestStatus.RunFail, $"no test data for graph input '{input.Name}'");
                    }

                    compiled.SetInput(input, ElementCodec.Encode(input.Type, tensor.Values));
                }

                compiled.Run(run);
                foreach (var (name, value) in graph.Outputs)
                {
                    var shape = compiled.OutputShape(name, run);
                    var buffer = new byte[value.Type.ByteSize(shape.Aggregate(1L, (a, b) => a * b))];
                    compiled.GetOutput(name, buffer);
                    actual[name] = (shape, ElementCodec.Decode(value.Type, buffer));
                }
            }
            catch (Exception e)
            {
                return new(NodeTestStatus.RunFail, Describe(e));
            }

            for (int k = 0; k < expected.Count; k++)
            {
                if (CompareValue(outputInfos[k].Name, expected[k], actual) is { } mismatch)
                {
                    return new(NodeTestStatus.Wrong, mismatch);
                }
            }
        }

        return pinnedNames.Length == 0 ? NodeTestOutcome.Passed : NodeTestOutcome.Passed with { Pinned = pinnedNames };
    }

    /// <summary>Import errors that pinning inputs to constants can resolve.</summary>
    private static bool NeedsConstant(string message)
        => message.Contains("compile-time constant", StringComparison.Ordinal)
            || message.Contains("data-dependent", StringComparison.Ordinal)
            || message.Contains("runtime-dependent", StringComparison.Ordinal)
            || message.Contains("runtime tensor", StringComparison.Ordinal);

    /// <summary>
    /// Compares an expected output with the graph outputs: a tensor by its name, a sequence either element by element
    /// (<see cref="OnnxGraphAdapter.SequenceElementName"/>) or as one output stacking the elements, an empty optional by its absence.
    /// </summary>
    private static string? CompareValue(string name, TestValue expected, Dictionary<string, (long[] Shape, double[] Values)> actual)
    {
        if (expected.Tensor is { } tensor)
        {
            return actual.TryGetValue(name, out var a) ? Compare(name, tensor, a.Shape, a.Values) : $"output '{name}' is missing";
        }

        if (expected.Elements is not { } elements)
        {
            return actual.ContainsKey(name) || actual.ContainsKey(OnnxGraphAdapter.SequenceElementName(name, 0)) ? $"output '{name}' should be an empty optional" : null;
        }

        var produced = new List<(long[] Shape, double[] Values)>();
        if (actual.TryGetValue(name, out var stacked))
        {
            long count = stacked.Shape.Length == 0 ? 1 : stacked.Shape[0];
            int size = count == 0 ? 0 : stacked.Values.Length / (int)count;
            for (int k = 0; k < count; k++)
            {
                produced.Add(([.. stacked.Shape.Skip(1)], stacked.Values[(k * size)..((k + 1) * size)]));
            }
        }
        else
        {
            for (int k = 0; actual.TryGetValue(OnnxGraphAdapter.SequenceElementName(name, k), out var element); k++)
            {
                produced.Add(element);
            }
        }

        if (produced.Count != elements.Count)
        {
            return $"output '{name}': sequence of {produced.Count} elements, expected {elements.Count}";
        }

        for (int k = 0; k < elements.Count; k++)
        {
            if (Compare(OnnxGraphAdapter.SequenceElementName(name, k), elements[k], produced[k].Shape, produced[k].Values) is { } mismatch)
            {
                return mismatch;
            }
        }

        return null;
    }

    private static string? Compare(string name, TestTensor expected, long[] shape, double[] values)
    {
        if (!SameShape(expected.Dims, shape))
        {
            return $"output '{name}': shape [{string.Join(", ", shape)}], expected [{string.Join(", ", expected.Dims)}]";
        }

        for (int i = 0; i < values.Length; i++)
        {
            if (!Close(expected.Values[i], values[i], expected.IsFloat))
            {
                int bad = Enumerable.Range(0, values.Length).Count(j => !Close(expected.Values[j], values[j], expected.IsFloat));
                return $"output '{name}' [{i}] = {values[i]}, expected {expected.Values[i]} ({bad} of {values.Length} elements differ)";
            }
        }

        return null;
    }

    private static bool Close(double e, double a, bool isFloat)
        => isFloat ? (double.IsNaN(e) && double.IsNaN(a)) || e == a || Math.Abs(a - e) <= Atol + Rtol * Math.Abs(e) : a == e;

    private static bool SameShape(long[] expected, long[] actual)
    {
        static IEnumerable<long> Squeeze(long[] d) => d.Length == 0 ? [1] : d;
        return Squeeze(expected).SequenceEqual(Squeeze(actual));
    }

    private static string Describe(Exception e)
    {
        var text = new StringBuilder(e.GetType().Name).Append(": ").Append(e.Message);
        if (e.InnerException is { } inner && !e.Message.Contains(inner.Message, StringComparison.Ordinal))
        {
            text.Append(" -> ").Append(inner.GetType().Name).Append(": ").Append(inner.Message);
        }

        if (e is not NotSupportedException and not InvalidDataException)
        {
            text.Append(" @ ").Append(e.StackTrace?.Split('\n').FirstOrDefault()?.Trim());
        }

        return text.ToString();
    }
}
