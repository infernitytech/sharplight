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
        foreach (var set in sets)
        {
            var outcome = RunDataSet(modelPath, set, device, fallback);
            if (outcome.Status != NodeTestStatus.Pass)
            {
                return sets.Count > 1 ? outcome with { Message = $"{Path.GetFileName(set)}: {outcome.Message}" } : outcome;
            }
        }

        return NodeTestOutcome.Passed;
    }

    private static NodeTestOutcome RunDataSet(string modelPath, string setDirectory, IComputeDevice device, IComputeDevice? fallback)
    {
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

        var options = new OnnxImportOptions();
        foreach (var (name, tensor) in inputs)
        {
            options.InputShapes[name] = tensor.Dims;
        }

        model.Options = options;
        ComputeGraph graph;
        try
        {
            var definition = OnnxGraphAdapter.Build(OnnxImporter.FromModel(model, modelPath));
            graph = definition.Graph(OnnxGraphAdapter.MainGraph);
        }
        catch (Exception e)
        {
            return new(NodeTestStatus.ImportFail, Describe(e));
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

        return NodeTestOutcome.Passed;
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
