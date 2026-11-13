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
        if (inputInfos.Concat(outputInfos).FirstOrDefault(i => i.ElemType == 0) is { } nonTensor)
        {
            return new(NodeTestStatus.HarnessUnsupported, $"'{nonTensor.Name}' is a sequence, map or optional value");
        }

        if (inputInfos.Concat(outputInfos).FirstOrDefault(i => i.ElemType == OnnxType.String) is { } stringValue)
        {
            return new(NodeTestStatus.HarnessUnsupported, $"'{stringValue.Name}' is a string tensor");
        }

        var inputs = new Dictionary<string, TestTensor>(StringComparer.Ordinal);
        var expected = new List<TestTensor>();
        try
        {
            for (int k = 0; File.Exists(Path.Combine(setDirectory, $"input_{k}.pb")); k++)
            {
                inputs[inputInfos[k].Name] = TestTensor.Load(Path.Combine(setDirectory, $"input_{k}.pb"));
            }

            for (int k = 0; File.Exists(Path.Combine(setDirectory, $"output_{k}.pb")); k++)
            {
                expected.Add(TestTensor.Load(Path.Combine(setDirectory, $"output_{k}.pb")));
            }
        }
        catch (Exception e) when (e is NotSupportedException or InvalidDataException)
        {
            return new(NodeTestStatus.HarnessUnsupported, e.Message);
        }

        // Sharplight compiles static shapes: inputs that define shapes, axes or other import-time parameters (Reshape targets, reduction
        // axes, pads, ...) are passed as tensors by the node tests but are initializers in real models. When the importer needs them as
        // constants, the test is retried with them pinned (OnnxImportOptions.ConstantInputs), first the integer inputs, then also small
        // float inputs (scales, Range bounds, ...).
        ComputeGraph? graph = null;
        string importError = "";
        int pinLevel = 0;
        for (; pinLevel <= 2 && graph is null; pinLevel++)
        {
            var pinned = inputs.Where(p => pinLevel switch
            {
                0 => false,
                1 => !p.Value.IsFloat,
                _ => !p.Value.IsFloat || p.Value.Elements <= 8,
            }).ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal);
            if (pinLevel > 0 && (pinned.Count == 0 || pinned.Count == inputs.Count))
            {
                continue;
            }

            var options = new OnnxImportOptions();
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
            var actual = new List<(long[] Shape, double[] Values)>();
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
                foreach (var info in outputInfos)
                {
                    var shape = compiled.OutputShape(info.Name, run);
                    var type = graph.Outputs[info.Name].Type;
                    var buffer = new byte[type.ByteSize(shape.Aggregate(1L, (a, b) => a * b))];
                    compiled.GetOutput(info.Name, buffer);
                    actual.Add((shape, ElementCodec.Decode(type, buffer)));
                }
            }
            catch (Exception e)
            {
                return new(NodeTestStatus.RunFail, Describe(e));
            }

            for (int k = 0; k < expected.Count; k++)
            {
                if (Compare(outputInfos[k].Name, expected[k], actual[k].Shape, actual[k].Values) is { } mismatch)
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
