using Sharplight.Graph;
using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>
/// The ONNX random operators: deterministic for a seed (across runs, compilations and providers) and statistically sound. Their node tests
/// cannot match numpy's generator, so these tests stand in for them.
/// </summary>
public class RandomOpTests
{
    private const int Count = 20_000;

    [Fact]
    public void RandomUniformLike_IsDeterministicAndUniform()
    {
        var model = new TestModel().Input("x", 1, Count).Output("y", 1, Count)
            .Node("RandomUniformLike", ["x"], ["y"], TestModel.AttrF("low", -1f), TestModel.AttrF("high", 3f), TestModel.AttrF("seed", 3f));
        var zeros = new float[Count];
        var first = Run(model, Providers.Cpu, zeros);
        Assert.Equal(first, Run(model, Providers.Cpu, zeros));
        Assert.All(first, v => Assert.InRange(v, -1f, 2.9999999f));
        Assert.InRange(first.Average(), 0.95, 1.05);
        Assert.InRange(Variance(first), (16.0 / 12) - 0.05, (16.0 / 12) + 0.05);

        var otherSeed = new TestModel().Input("x", 1, Count).Output("y", 1, Count)
            .Node("RandomUniformLike", ["x"], ["y"], TestModel.AttrF("low", -1f), TestModel.AttrF("high", 3f), TestModel.AttrF("seed", 4f));
        Assert.NotEqual(first, Run(otherSeed, Providers.Cpu, zeros));
    }

    [Fact]
    public void RandomNormal_HasTheRequestedMoments()
    {
        var model = new TestModel().Input("x", 1, Count).Output("y", 1, Count)
            .Node("RandomNormal", [], ["r"], TestModel.AttrInts("shape", 1, Count), TestModel.AttrF("mean", 2f), TestModel.AttrF("scale", 0.5f),
                TestModel.AttrF("seed", 5f))
            .Node("Add", ["r", "x"], ["y"]);
        var y = Run(model, Providers.Cpu, new float[Count]);
        Assert.InRange(y.Average(), 1.98, 2.02);
        Assert.InRange(Math.Sqrt(Variance(y)), 0.48, 0.52);
        Assert.Equal(y, Run(model, Providers.Cpu, new float[Count]));
    }

    [Fact]
    public void UnseededNodes_AreReproducibleButDistinct()
    {
        var model = new TestModel().Input("x", 1, Count).Output("a", 1, Count).Output("b", 1, Count)
            .Node("RandomNormalLike", ["x"], ["a"])
            .Node("RandomNormalLike", ["x"], ["b"]);
        var zeros = new float[Count];
        var (a, b) = RunTwo(model, Providers.Cpu, zeros);
        var (a2, _) = RunTwo(model, Providers.Cpu, zeros);
        Assert.Equal(a, a2);
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void Bernoulli_FollowsTheProbabilities()
    {
        var model = new TestModel().Input("p", 1, Count).Output("y", 1, Count).Node("Bernoulli", ["p"], ["y"], TestModel.AttrF("seed", 1f));
        var p = Enumerable.Range(0, Count).Select(i => i % 2 == 0 ? 0.2f : 0.9f).ToArray();
        var y = Run(model, Providers.Cpu, p);
        Assert.All(y, v => Assert.True(v is 0f or 1f));
        Assert.InRange(y.Where((_, i) => i % 2 == 0).Average(), 0.18, 0.22);
        Assert.InRange(y.Where((_, i) => i % 2 == 1).Average(), 0.88, 0.92);
        Assert.Equal(y, Run(model, Providers.Cpu, p));
    }

    [Fact]
    public void Multinomial_SamplesTheSoftmaxDistribution()
    {
        var model = new TestModel().Input("x", 2, 3).Output("y", 2, Count)
            .Node("Multinomial", ["x"], ["y"], TestModel.Attr("sample_size", Count), TestModel.AttrF("seed", 2f), TestModel.Attr("dtype", 6));
        float[] logits = [MathF.Log(0.2f), MathF.Log(0.3f), MathF.Log(0.5f), 0f, 0f, 10f];
        var y = Run(model, Providers.Cpu, logits);
        var first = y.Take(Count).ToArray();
        Assert.InRange(first.Count(v => v == 0) / (double)Count, 0.19, 0.21);
        Assert.InRange(first.Count(v => v == 1) / (double)Count, 0.29, 0.31);
        Assert.InRange(first.Count(v => v == 2) / (double)Count, 0.49, 0.51);
        Assert.True(y.Skip(Count).Count(v => v == 2) > Count * 0.999);
        Assert.Equal(y, Run(model, Providers.Cpu, logits));
    }

    [Fact]
    public void SeededGraph_GivesTheSameValuesOnVulkanAndCpu()
    {
        var device = Providers.VulkanOrSkip();
        var model = new TestModel().Input("x", 1, Count).Output("y", 1, Count)
            .Node("RandomUniformLike", ["x"], ["u"], TestModel.AttrF("seed", 9f))
            .Node("Mul", ["u", "x"], ["y"]);
        var x = Enumerable.Range(0, Count).Select(i => (float)(i % 7)).ToArray();
        Assert.Equal(Run(model, Providers.Cpu, x), Run(model, device, x));
    }

    private static double Variance(float[] v)
    {
        double mean = v.Average();
        return v.Sum(x => (x - mean) * (x - mean)) / v.Length;
    }

    private static float[] Run(TestModel model, IComputeDevice device, float[] input) => Execute(model, device, input)[0];

    private static (float[], float[]) RunTwo(TestModel model, IComputeDevice device, float[] input)
    {
        var outputs = Execute(model, device, input);
        return (outputs[0], outputs[1]);
    }

    private static float[][] Execute(TestModel model, IComputeDevice device, float[] input)
    {
        var graph = model.Build().Graph(OnnxGraphAdapter.MainGraph);
        using var compiled = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Fallback = device.Info.Kind == DeviceKind.Cpu ? null : new DeviceSlot { Device = Providers.Cpu },
            Compile = new CompileOptions { SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = 1 } },
        });
        var run = new RunParameters();
        run.Symbols[OnnxGraphAdapter.BatchSymbol] = 1;
        compiled.SetInput<float>(graph.Inputs[0], input);
        compiled.Run(run);
        return [.. graph.Outputs.Select(o => Read(compiled, o.Key, o.Value, run))];
    }

    private static float[] Read(CompiledModel compiled, string name, Value value, RunParameters run)
    {
        long count = compiled.OutputShape(name, run).Aggregate(1L, (a, b) => a * b);
        if (value.Type != Sharplight.Tensors.DataType.I32)
        {
            var f = new float[count];
            compiled.GetOutput<float>(name, f);
            return f;
        }

        var i = new int[count];
        compiled.GetOutput<int>(name, i);
        return [.. i.Select(v => (float)v)];
    }
}
