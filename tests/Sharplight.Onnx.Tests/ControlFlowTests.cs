using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Onnx.Tests;

/// <summary>Runtime Loop and Scan (the IR Loop node run by a host segment), executed on the CPU and on Vulkan.</summary>
public class ControlFlowTests
{
    private const int Float = 1;
    private const int Int64 = 7;
    private const int Bool = 9;

    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    public void Loop_WithRuntimeTripCount_StacksScanOutputs(string provider)
    {
        // Range written as a Loop: out[k] = start + k·delta for k < trip; final = start + trip·delta.
        var body = Graph(
            "body",
            [TestModel.ValueInfo("i", Int64), TestModel.ValueInfo("c", Bool), TestModel.ValueInfo("prev", Float)],
            [TestModel.ValueInfo("c_out", Bool), TestModel.ValueInfo("cur", Float), TestModel.ValueInfo("range", Float)],
            ("Identity", ["c"], ["c_out"], []),
            ("Add", ["prev", "delta"], ["cur"], []),
            ("Identity", ["prev"], ["range"], []));
        var model = new TestModel()
            .Input(TestModel.ValueInfo("trip", Int64)).Input(TestModel.ValueInfo("start", Float)).Input(TestModel.ValueInfo("delta", Float))
            .Output(TestModel.ValueInfo("final", Float)).Output(TestModel.ValueInfo("out", Float, "K"))
            .Node("Loop", ["trip", "", "start"], ["final", "out"], GraphAttr("body", body));
        using var run = new Runner(model, provider);
        Assert.Contains(run.Graph.Nodes, n => n.Op.Name == "Loop");
        foreach (int trip in new[] { 5, 2 })
        {
            var outputs = run.Run([trip], [1.5f], [0.5f]);
            Assert.Equal([1.5f + trip * 0.5f], outputs["final"]);
            Assert.Equal(Enumerable.Range(0, trip).Select(k => 1.5f + k * 0.5f), outputs["out"]);
        }
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    public void Loop_WithRuntimeCondition_RunsUntilTheBodyStops(string provider)
    {
        // while (cur < limit) cur += step; the scan output records every value, including the one that ends the loop.
        var body = Graph(
            "body",
            [TestModel.ValueInfo("i", Int64), TestModel.ValueInfo("c", Bool), TestModel.ValueInfo("prev", Float, 2)],
            [TestModel.ValueInfo("c_out", Bool), TestModel.ValueInfo("cur", Float, 2), TestModel.ValueInfo("seen", Float, 2)],
            ("Add", ["prev", "step"], ["cur"], []),
            ("ReduceMax", ["cur"], ["peak"], [TestModel.Attr("keepdims", 0)]),
            ("Less", ["peak", "limit"], ["c_out"], []),
            ("Identity", ["cur"], ["seen"], []));
        var model = new TestModel()
            .Input("s0", 2).Input("step", 2).Input(TestModel.ValueInfo("limit", Float))
            .Output("last", 2).Output("seen", "K", 2)
            .Node("Loop", ["", "", "s0"], ["last", "seen"], GraphAttr("body", body));
        using var run = new Runner(model, provider);
        var outputs = run.Run([0f, 1f], [1f, 2f], [7f]);
        Assert.Equal([3f, 7f], outputs["last"]);
        Assert.Equal([1f, 3f, 2f, 5f, 3f, 7f], outputs["seen"]);
        outputs = run.Run([0f, 0f], [1f, 1f], [2.5f]);
        Assert.Equal([3f, 3f], outputs["last"]);
        Assert.Equal(3 * 2, outputs["seen"].Length);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    public void Scan_OverTheSymbolicBatchAxis_InReverse(string provider)
    {
        // Reverse cumulative sum of the rows of x[N, 3].
        var body = Graph(
            "body",
            [TestModel.ValueInfo("sum_in", Float, 3), TestModel.ValueInfo("row", Float, 3)],
            [TestModel.ValueInfo("sum_out", Float, 3), TestModel.ValueInfo("partial", Float, 3)],
            ("Add", ["sum_in", "row"], ["sum_out"], []),
            ("Identity", ["sum_out"], ["partial"], []));
        var model = new TestModel()
            .Input("init", 3).Input("x", "N", 3)
            .Output("total", 3).Output("partials", "N", 3)
            .Node("Scan", ["init", "x"], ["total", "partials"], GraphAttr("body", body), TestModel.Attr("num_scan_inputs", 1),
                TestModel.AttrInts("scan_input_directions", 1));
        using var run = new Runner(model, provider, batch: 8);
        var x = Enumerable.Range(0, 15).Select(v => (float)v).ToArray();
        var outputs = run.Run([[1f, 1f, 1f], x], batch: 5);
        var expected = new float[15];
        var sum = new[] { 1f, 1f, 1f };
        for (int row = 4; row >= 0; row--)
        {
            for (int c = 0; c < 3; c++)
            {
                sum[c] += x[row * 3 + c];
                expected[(4 - row) * 3 + c] = sum[c];
            }
        }

        Assert.Equal(sum, outputs["total"]);
        Assert.Equal(expected, outputs["partials"]);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    public void Loop_BodyWithRuntimeIf_ReadsValuesOfBothEnclosingGraphs(string provider)
    {
        // acc = acc + a on even iterations, acc * a on odd ones: the If branches capture 'a' (main graph) and 'acc' (loop body).
        var then = Graph("then", [], [TestModel.ValueInfo("t", Float, 2)], ("Add", ["acc", "a"], ["t"], []));
        var other = Graph("else", [], [TestModel.ValueInfo("e", Float, 2)], ("Mul", ["acc", "a"], ["e"], []));
        var body = Graph(
            "body",
            [TestModel.ValueInfo("i", Int64), TestModel.ValueInfo("c", Bool), TestModel.ValueInfo("acc", Float, 2)],
            [TestModel.ValueInfo("c_out", Bool), TestModel.ValueInfo("acc_out", Float, 2)],
            ("Identity", ["c"], ["c_out"], []),
            ("Mod", ["i", "two"], ["parity"], []),
            ("Equal", ["parity", "zero"], ["even"], []),
            ("If", ["even"], ["acc_out"], [GraphAttr("then_branch", then), GraphAttr("else_branch", other)]));
        var model = new TestModel()
            .Input(TestModel.ValueInfo("trip", Int64)).Input("acc0", 2).Input("a", 2)
            .Output("acc", 2)
            .InitI64("two", [], 2).InitI64("zero", [], 0)
            .Node("Loop", ["trip", "", "acc0"], ["acc"], GraphAttr("body", body));
        using var run = new Runner(model, provider);
        var outputs = run.Run([5], [1f, 2f], [2f, 3f]);
        float[] acc = [1f, 2f];
        float[] a = [2f, 3f];
        for (int i = 0; i < 5; i++)
        {
            acc = [.. acc.Select((v, k) => i % 2 == 0 ? v + a[k] : v * a[k])];
        }

        Assert.Equal(acc, outputs["acc"]);
    }

    [Theory]
    [InlineData("cpu")]
    [InlineData("vulkan")]
    public void Loop_AppendingToASequence_GivesARuntimeLengthSequence(string provider)
    {
        // seq = []; for i < trip: seq.append(x * i); outputs: the sequence (stacked) and ConcatFromSequence(seq, axis 0).
        var body = Graph(
            "body",
            [TestModel.ValueInfo("i", Int64), TestModel.ValueInfo("c", Bool), SequenceInfo("seq_in")],
            [TestModel.ValueInfo("c_out", Bool), SequenceInfo("seq_out")],
            ("Identity", ["c"], ["c_out"], []),
            ("Cast", ["i"], ["fi"], [TestModel.Attr("to", Float)]),
            ("Mul", ["x", "fi"], ["row"], []),
            ("SequenceInsert", ["seq_in", "row"], ["seq_out"], []));
        var model = new TestModel()
            .Input(TestModel.ValueInfo("trip", Int64)).Input("x", 1, 3)
            .Output(SequenceInfo("rows")).Output("joined", "K", 3)
            .Node("SequenceEmpty", [], ["empty"])
            .Node("Loop", ["trip", "", "empty"], ["rows"], GraphAttr("body", body))
            .Node("ConcatFromSequence", ["rows"], ["joined"], TestModel.Attr("axis", 0));
        using var run = new Runner(model, provider);
        Assert.Contains("rows", run.Graph.Outputs.Keys);
        var outputs = run.Run([4], [1f, 2f, 3f]);
        float[] expected = [.. Enumerable.Range(0, 4).SelectMany(i => new[] { 1f * i, 2f * i, 3f * i })];
        Assert.Equal(expected, outputs["rows"]);
        Assert.Equal(expected, outputs["joined"]);
    }

    private static ProtoWriter SequenceInfo(string name)
    {
        var element = new ProtoWriter().Msg(1, new ProtoWriter().Int(1, Float));
        return new ProtoWriter().Str(1, name).Msg(2, new ProtoWriter().Msg(4, new ProtoWriter().Msg(1, element)));
    }

    private static ProtoWriter GraphAttr(string name, ProtoWriter graph) => new ProtoWriter().Str(1, name).Msg(6, graph).Int(20, 5);

    private static ProtoWriter Graph(string name, ProtoWriter[] inputs, ProtoWriter[] outputs, params (string Op, string[] In, string[] Out, ProtoWriter[] Attrs)[] nodes)
    {
        var g = new ProtoWriter();
        foreach (var (op, ins, outs, attrs) in nodes)
        {
            var node = new ProtoWriter();
            foreach (var i in ins)
            {
                node.Str(1, i);
            }

            foreach (var o in outs)
            {
                node.Str(2, o);
            }

            node.Str(3, outs[0] + "_node").Str(4, op);
            foreach (var a in attrs)
            {
                node.Msg(5, a);
            }

            g.Msg(1, node);
        }

        g.Str(2, name);
        foreach (var i in inputs)
        {
            g.Msg(11, i);
        }

        foreach (var o in outputs)
        {
            g.Msg(12, o);
        }

        return g;
    }

    /// <summary>Imports a test model, compiles it once and runs it with float inputs (integer inputs are converted).</summary>
    private sealed class Runner : IDisposable
    {
        private readonly CompiledModel _compiled;

        public Runner(TestModel model, string provider, int batch = 1)
        {
            Graph = model.Build().Graph(OnnxGraphAdapter.MainGraph);
            var device = Providers.Get(provider);
            _compiled = CompiledModel.Compile(Graph, new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = device }],
                Fallback = device.Info.Kind == DeviceKind.Cpu ? null : new DeviceSlot { Device = Providers.Cpu },
                Compile = new CompileOptions { SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = batch } },
                ReservePerDevice = 16L << 20,
            });
        }

        public ComputeGraph Graph { get; }

        public Dictionary<string, float[]> Run(params float[][] inputs) => Run(inputs, 1);

        public Dictionary<string, float[]> Run(float[][] inputs, int batch)
        {
            var parameters = new RunParameters();
            parameters.Symbols[OnnxGraphAdapter.BatchSymbol] = batch;
            for (int i = 0; i < Graph.Inputs.Count; i++)
            {
                if (Graph.Inputs[i].Type == DataType.I32)
                {
                    _compiled.SetInput<int>(Graph.Inputs[i], inputs[i].Select(v => (int)v).ToArray());
                }
                else
                {
                    _compiled.SetInput<float>(Graph.Inputs[i], inputs[i]);
                }
            }

            _compiled.Run(parameters);
            var outputs = new Dictionary<string, float[]>();
            foreach (var (name, value) in Graph.Outputs)
            {
                var data = new float[_compiled.OutputShape(name, parameters).Aggregate(1L, (a, b) => a * b)];
                Assert.Equal(DataType.F32, value.Type);
                _compiled.GetOutput<float>(name, data);
                outputs[name] = data;
            }

            return outputs;
        }

        public void Dispose() => _compiled.Dispose();
    }
}
