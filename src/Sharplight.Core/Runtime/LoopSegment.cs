using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Runtime;

/// <summary>
/// Host segment of a runtime <see cref="StandardOps.Loop"/>: the body is compiled once as a nested model on the same devices; a run feeds it
/// the iteration number, the condition, the loop-carried values, one slice of every scan input and the captured values, runs it until the
/// trip count is reached or it returns a false condition, and collects the carried values and scan outputs in host memory. The number of
/// iterations is published as the node's dynamic symbol when it has one.
/// </summary>
internal sealed class LoopSegment : ICompiledSegment
{
    private readonly Node _node;
    private readonly CompiledModel _body;
    private readonly Value?[] _bodyInputs;
    private readonly string[] _bodyOutputs;
    private readonly int _states;
    private readonly int _scanInputs;
    private readonly long[] _scanInputReverse;
    private readonly long[] _scanOutputReverse;
    private readonly string _tripSymbol;
    private readonly string _countSymbol;
    private readonly long _countBound;
    private readonly Dictionary<Value, byte[]> _inputs = [];
    private readonly Dictionary<Value, byte[]> _outputs = [];

    public LoopSegment(GraphSegment segment, PlacementOptions placement)
    {
        Segment = segment;
        _node = segment.Nodes.Single();
        var a = _node.Attributes;
        _body = CompiledModel.Compile(a.Get<ComputeGraph>("body"), placement);
        _bodyInputs = [.. a.Get<string[]>("body_inputs").Select(name => _body.Graph.Inputs.FirstOrDefault(v => v.Name == name))];
        _bodyOutputs = a.Get<string[]>("body_outputs");
        _states = (int)a.Get<long>("state_count");
        _scanInputs = (int)a.Get<long>("scan_input_count");
        _scanInputReverse = a.Get<long[]>("scan_input_reverse");
        _scanOutputReverse = a.Get<long[]>("scan_output_reverse");
        _tripSymbol = a.Get<string>("trip_symbol");
        _countSymbol = a.Get<string>("count_symbol");
        _countBound = _countSymbol.Length > 0 ? placement.Compile.Bound(_countSymbol) : long.MaxValue;
    }

    public GraphSegment Segment { get; }

    public long DeviceMemoryBytes => _body.AcceleratorMemoryBytes;

    public void SetInput(Value value, ReadOnlySpan<byte> data)
    {
        if (!_inputs.TryGetValue(value, out var buffer) || buffer.Length != data.Length)
        {
            _inputs[value] = buffer = new byte[data.Length];
        }

        data.CopyTo(buffer);
    }

    public void Execute(RunParameters parameters)
    {
        var inputs = _node.Inputs;
        long trip = HostScalars.ReadCount(inputs[0].Type, Input(inputs[0]));
        if (_tripSymbol.Length > 0)
        {
            trip = Math.Min(trip, SymbolicExpression.Evaluate(_tripSymbol, parameters.Symbols));
        }

        int firstScan = 2 + _states;
        var scanExtents = new long[_scanInputs];
        for (int j = 0; j < _scanInputs; j++)
        {
            var scan = inputs[firstScan + j];
            scanExtents[j] = scan.Shape[0].Resolve(parameters.Symbols);
            trip = Math.Min(trip, scanExtents[j]);
        }

        // Captured values do not change between iterations: they are passed to the body once.
        for (int k = firstScan + _scanInputs; k < inputs.Length; k++)
        {
            Feed(k, Input(inputs[k]));
        }

        var states = Enumerable.Range(0, _states).Select(k => Input(inputs[2 + k])).ToArray();
        var scans = Enumerable.Range(0, _bodyOutputs.Length - 1 - _states).Select(_ => new List<byte[]>()).ToArray();
        bool running = HostScalars.IsTrue(inputs[1].Type, Input(inputs[1]));
        long done = 0;
        for (; done < trip && running; done++)
        {
            if (done >= _countBound)
            {
                throw new InvalidOperationException(
                    $"Loop node {_node.Id} runs more than {_countBound} iterations, the bound its scan outputs were compiled for (symbol {_countSymbol}).");
            }

            RunIteration(done, states, scanExtents, parameters);
            running = _bodyOutputs[0].Length == 0 || HostScalars.IsTrue(_body.Graph.Outputs[_bodyOutputs[0]].Type, Read(_bodyOutputs[0], parameters));
            for (int k = 0; k < _states; k++)
            {
                states[k] = Read(_bodyOutputs[1 + k], parameters);
            }

            for (int j = 0; j < scans.Length; j++)
            {
                scans[j].Add(Read(_bodyOutputs[1 + _states + j], parameters));
            }
        }

        if (_countSymbol.Length > 0)
        {
            parameters.Symbols[_countSymbol] = done;
        }

        for (int k = 0; k < _states; k++)
        {
            _outputs[_node.Outputs[k]] = states[k];
        }

        for (int j = 0; j < scans.Length; j++)
        {
            var slices = _scanOutputReverse[j] == 1 ? Enumerable.Reverse(scans[j]) : scans[j];
            _outputs[_node.Outputs[_states + j]] = [.. slices.SelectMany(s => s)];
        }
    }

    public void GetOutput(Value value, Span<byte> destination) => _outputs[value].AsSpan(0, destination.Length).CopyTo(destination);

    public void ResetState() => _body.ResetState();

    public void Dispose() => _body.Dispose();

    private void RunIteration(long iteration, byte[][] states, long[] scanExtents, RunParameters parameters)
    {
        Feed(0, HostScalars.Encode(DataType.I32, iteration));
        if (_bodyInputs[1] is { } condition)
        {
            _body.SetInput(condition, HostScalars.Encode(condition.Type, 1));
        }

        for (int k = 0; k < _states; k++)
        {
            Feed(2 + k, states[k]);
        }

        for (int j = 0; j < _scanInputs; j++)
        {
            int k = 2 + _states + j;
            var scan = _node.Inputs[k];
            var whole = Input(scan);
            int slice = (int)(whole.Length / Math.Max(1, scanExtents[j]));
            long index = _scanInputReverse[j] == 1 ? scanExtents[j] - 1 - iteration : iteration;
            Feed(k, whole.AsSpan((int)(index * slice), slice));
        }

        _body.Run(parameters);
    }

    /// <summary>The data of a node input: constants are embedded in the graph, everything else is forwarded by the executor.</summary>
    private byte[] Input(Value value) => value.ConstantData ?? _inputs[value];

    private void Feed(int bodyInput, ReadOnlySpan<byte> data)
    {
        if (_bodyInputs[bodyInput] is { } input)
        {
            _body.SetInput(input, data);
        }
    }

    private byte[] Read(string output, RunParameters parameters)
    {
        var value = _body.Graph.Outputs[output];
        var bytes = new byte[CompiledModel.ByteCount(value, parameters)];
        _body.GetOutput(output, bytes);
        return bytes;
    }
}
