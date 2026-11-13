using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Runtime;

/// <summary>
/// Host segment of a runtime <see cref="StandardOps.If"/>: both branches are compiled as nested models on the same devices; a run reads the
/// condition, runs the selected branch on the captured values and keeps its outputs in host memory. Outputs whose outer dimension differs
/// between the branches publish its value as the dynamic symbol named by the node.
/// </summary>
internal sealed class IfSegment : ICompiledSegment
{
    private readonly Node _node;
    private readonly CompiledModel _then;
    private readonly CompiledModel _else;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly string[] _outputSymbols;
    private readonly Dictionary<Value, byte[]> _inputs = [];
    private readonly Dictionary<Value, byte[]> _outputs = [];

    public IfSegment(GraphSegment segment, PlacementOptions placement)
    {
        Segment = segment;
        _node = segment.Nodes.Single();
        var a = _node.Attributes;
        _inputNames = a.Get<string[]>("inputs");
        _outputNames = a.Get<string[]>("outputs");
        _outputSymbols = a.Get<string[]>("output_symbols");
        _then = CompiledModel.Compile(a.Get<ComputeGraph>("then_branch"), placement);
        try
        {
            _else = CompiledModel.Compile(a.Get<ComputeGraph>("else_branch"), placement);
        }
        catch
        {
            _then.Dispose();
            throw;
        }
    }

    public GraphSegment Segment { get; }

    public long DeviceMemoryBytes => _then.AcceleratorMemoryBytes + _else.AcceleratorMemoryBytes;

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
        var branch = Condition() ? _then : _else;
        for (int k = 0; k < _inputNames.Length; k++)
        {
            var captured = _node.Inputs[k + 1];
            var input = branch.Graph.Inputs.FirstOrDefault(v => v.Name == _inputNames[k]);
            if (input is not null)
            {
                branch.SetInput(input, _inputs[captured]);
            }
        }

        branch.Run(parameters);
        for (int k = 0; k < _outputNames.Length; k++)
        {
            var shape = branch.OutputShape(_outputNames[k], parameters);
            if (_outputSymbols[k].Length > 0)
            {
                parameters.Symbols[_outputSymbols[k]] = shape[0];
            }

            var output = _node.Outputs[k];
            long bytes = output.Type.ByteSize(shape.Aggregate(1L, (x, y) => x * y));
            if (!_outputs.TryGetValue(output, out var buffer) || buffer.Length != bytes)
            {
                _outputs[output] = buffer = new byte[bytes];
            }

            branch.GetOutput(_outputNames[k], buffer);
        }
    }

    public void GetOutput(Value value, Span<byte> destination) => _outputs[value].AsSpan(0, destination.Length).CopyTo(destination);

    public void ResetState()
    {
        _then.ResetState();
        _else.ResetState();
    }

    public void Dispose()
    {
        _then.Dispose();
        _else.Dispose();
    }

    private bool Condition()
    {
        var cond = _node.Inputs[0];
        var bytes = _inputs[cond];
        int size = (int)cond.Type.ByteSize(1);
        return cond.Type == DataType.F32 ? BitConverter.ToSingle(bytes, 0) != 0 : bytes.AsSpan(0, size).IndexOfAnyExcept((byte)0) >= 0;
    }
}
