using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Import.Onnx.Tests;

/// <summary>
/// Zero-size inputs (empty constants whose consumers fold: reductions over an empty set give their identity, empty results stay empty)
/// and graph outputs whose ONNX layout has a data-dependent inner axis (<see cref="OnnxOutputLayout"/>).
/// </summary>
public sealed class EmptyTensorAndLayoutTests
{
    [Fact]
    public void ReductionsOverAnEmptySet_GiveTheirIdentity()
    {
        var model = new TestModel().Opset(18).Input("x", "a", "b", 3).InitI64("axes", [1], 1)
            .Output("max", 2, 1, 3).Output("prod", 2, 1, 3).Output("lse", 2, 1, 3).Output("l2", 2, 1, 3).Output("same", 2, 0, 3)
            .Node("ReduceMax", ["x", "axes"], ["max"]).Node("ReduceProd", ["x", "axes"], ["prod"])
            .Node("ReduceLogSumExp", ["x", "axes"], ["lse"]).Node("ReduceL2", ["x", "axes"], ["l2"]).Node("Exp", ["x"], ["same"]);
        var definition = model.Build(new OnnxImportOptions { InputShapes = { ["x"] = [2, 0, 3] } });
        Assert.Empty(definition.Graph("main").Inputs);

        var outputs = Run(definition, []);
        Assert.All(outputs["max"].Values, v => Assert.Equal(float.NegativeInfinity, v));
        Assert.All(outputs["prod"].Values, v => Assert.Equal(1f, v));
        Assert.All(outputs["lse"].Values, v => Assert.Equal(float.NegativeInfinity, v));
        Assert.All(outputs["l2"].Values, v => Assert.Equal(0f, v));
        Assert.Equal([2L, 1, 3], outputs["max"].Shape);
        Assert.Equal([2L, 0, 3], outputs["same"].Shape);
        Assert.Empty(outputs["same"].Values);
    }

    [Fact]
    public void NonZeroGraphOutput_IsTransposedBackToTheOnnxLayout()
    {
        var model = new TestModel().Opset(13).Input("x", 2, 2).Output("y", 2, "n").Node("NonZero", ["x"], ["y"]);
        var definition = model.Build();
        var outputs = Run(definition, [[1f, 0f, 1f, 1f]]);
        Assert.Equal([2L, 3], outputs["y"].Shape);
        Assert.Equal([0f, 1f, 1f, 0f, 0f, 1f], outputs["y"].Values);
    }

    [Fact]
    public void UniqueAlongAnInnerAxis_ReturnsTheUniqueColumns()
    {
        var model = new TestModel().Opset(11).Input("x", 3, 3).Output("y", 3, "u").Output("first", "u").Output("inverse", 3).Output("counts", "u")
            .Node("Unique", ["x"], ["y", "first", "inverse", "counts"], TestModel.Attr("axis", -1), TestModel.Attr("sorted", 1));
        var definition = model.Build();
        var outputs = Run(definition, [[1f, 0f, 0f, 1f, 0f, 0f, 2f, 3f, 3f]]);
        Assert.Equal([3L, 2], outputs["y"].Shape);
        Assert.Equal([0f, 1f, 0f, 1f, 3f, 2f], outputs["y"].Values);
        Assert.Equal([1f, 0f], outputs["first"].Values);
        Assert.Equal([1f, 0f, 0f], outputs["inverse"].Values);
        Assert.Equal([2f, 1f], outputs["counts"].Values);
    }

    [Fact]
    public void OutputLayout_PrependsUnitDimsAndPermutes()
    {
        var layout = new OnnxOutputLayout(1, [0, 2, 1]);
        Assert.Equal([1L, 3, 2], layout.OnnxShape([2, 3]));
        Assert.Equal([1, 4, 2, 5, 3, 6], layout.ToOnnx<int>([1, 2, 3, 4, 5, 6], [2, 3]));
    }

    /// <summary>Compiles and runs a definition on the CPU; returns every output in its ONNX layout.</summary>
    private static Dictionary<string, (long[] Shape, float[] Values)> Run(ModelDefinition definition, float[][] inputs)
    {
        var graph = definition.Graph(OnnxGraphAdapter.MainGraph);
        var layouts = definition.Properties.TryGetValue(OnnxGraphAdapter.OutputLayoutsProperty, out var property)
            ? (IReadOnlyDictionary<string, OnnxOutputLayout>)property
            : new Dictionary<string, OnnxOutputLayout>();
        using var compiled = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Providers.Cpu }],
            Compile = new CompileOptions { SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = 1 } },
        });
        var run = new RunParameters();
        run.Symbols[OnnxGraphAdapter.BatchSymbol] = 1;
        for (int i = 0; i < graph.Inputs.Count; i++)
        {
            compiled.SetInput<float>(graph.Inputs[i], inputs[i]);
        }

        compiled.Run(run);
        var outputs = new Dictionary<string, (long[] Shape, float[] Values)>(StringComparer.Ordinal);
        foreach (var (name, value) in graph.Outputs)
        {
            var shape = compiled.OutputShape(name, run);
            var values = new float[shape.Aggregate(1L, (a, b) => a * b)];
            if (value.Type == DataType.I32)
            {
                var ints = new int[values.Length];
                compiled.GetOutput<int>(name, ints);
                values = [.. ints.Select(v => (float)v)];
            }
            else
            {
                compiled.GetOutput<float>(name, values);
            }

            outputs[name] = layouts.TryGetValue(name, out var layout) ? (layout.OnnxShape(shape), layout.ToOnnx<float>(values, shape)) : (shape, values);
        }

        return outputs;
    }
}
