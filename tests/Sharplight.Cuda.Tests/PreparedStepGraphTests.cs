using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// A step shape prepared ahead (<see cref="CompiledModel.Prepare"/>, as the serving engine prepares its decode steps) is captured into a
/// CUDA graph at once: its first run replays the graph instead of launching its recording one launch at a time, and gives the results of
/// the same shape run without preparation, bit for bit. Shapes run after it keep their own recordings and results.
/// </summary>
public sealed class PreparedStepGraphTests
{
    private const int Hidden = 512;

    private const int Layers = 4;

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(48)]
    public void PreparedShape_FirstRunReplaysAGraph_WithTheResultsOfAnUnpreparedRun(int rows)
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        using var device = provider.OpenDevice(provider.GetDevices()[0]);
        var (graph, x) = BuildStack();
        var placement = new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = 64, ContextLength = 64 },
            ReservePerDevice = 0,
        };
        using var prepared = CompiledModel.Compile(graph, placement);
        using var plain = CompiledModel.Compile(graph, placement);
        var input = new float[rows * Hidden];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = MathF.Sin(i * 0.37f) * (1 + i % 3);
        }

        var parameters = new RunParameters();
        parameters.Symbols["T"] = rows;
        prepared.Prepare(parameters);

        var (fromGraph, timings) = Run(prepared, x, input, rows);
        Assert.True(timings.Replayed, $"{rows} rows: the prepared shape's first run did not replay a graph");
        Assert.False(timings.Recorded, $"{rows} rows: the prepared shape was recorded again at its first run");

        var (launched, first) = Run(plain, x, input, rows);
        Assert.False(first.Replayed, "an unprepared shape's first run replayed a graph");
        Assert.Equal(launched, fromGraph);

        // Another shape after the prepared one, then the prepared one again: each keeps its own recording and results.
        int other = rows == 1 ? 2 : rows - 1;
        var otherInput = new float[other * Hidden];
        for (int i = 0; i < otherInput.Length; i++)
        {
            otherInput[i] = MathF.Cos(i * 0.21f);
        }

        Assert.Equal(Run(plain, x, otherInput, other).Output, Run(prepared, x, otherInput, other).Output);
        Assert.Equal(launched, Run(prepared, x, input, rows).Output);
    }

    private static (ComputeGraph Graph, Value X) BuildStack()
    {
        var tg = new TG("prepared-step-graph");
        var x = tg.InF("x", TG.T, Hidden);
        var h = x;
        for (int l = 0; l < Layers; l++)
        {
            var n = tg.B.RmsNorm(h, null, 1e-5f);
            h = tg.B.Add(h, tg.B.MatMul(n, tg.RandW($"w{l}", DataType.BF16, 0.05f, Hidden, Hidden)));
        }

        tg.B.Output("y", h);
        return (tg.B.Build(), x);
    }

    private static (float[] Output, SegmentRunTimings Timings) Run(CompiledModel model, Value x, float[] input, int rows)
    {
        model.SetInput<float>(x, input);
        var parameters = new RunParameters();
        parameters.Symbols["T"] = rows;
        model.Run(parameters);
        var output = new float[rows * Hidden];
        model.GetOutput<float>("y", output);
        return (output, model.LastRunTimings);
    }
}
