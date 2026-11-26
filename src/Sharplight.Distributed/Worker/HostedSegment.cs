using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Distributed.Worker;

/// <summary>A segment a coordinator compiled on this worker: the compiled graph and what it exchanges with neighbours.</summary>
internal sealed class HostedSegment(CompiledModel model, ComputeGraph graph, IReadOnlyList<string> outputNames) : IDisposable
{
    /// <summary>The segment compiled over this worker's devices.</summary>
    public CompiledModel Model { get; } = model;

    /// <summary>The segment as a graph (inputs and outputs in the coordinator's order).</summary>
    public ComputeGraph Graph { get; } = graph;

    /// <summary>The output names in the coordinator's order.</summary>
    public IReadOnlyList<string> OutputNames { get; } = outputNames;

    /// <summary>Reused for every execution.</summary>
    public RunParameters Parameters { get; } = new();

    /// <summary>Executions so far (the number of the next one): matches a forwarded value with the execution that takes it.</summary>
    public long Executions { get; set; }

    /// <summary>Outputs sent to a neighbour after every execution: output index, the neighbour's address ("" for this worker) and the tag.</summary>
    public List<(int Output, string Endpoint, long Tag)> Forwards { get; } = [];

    /// <summary>Inputs taken from a neighbour at every execution: input index and tag.</summary>
    public List<(int Input, long Tag)> ForwardedInputs { get; } = [];

    /// <summary>How the narrowed inputs and outputs travel.</summary>
    public BoundaryPrecision Precision { get; init; }

    /// <summary>The inputs that travel in <see cref="Precision"/> (by input index).</summary>
    public bool[] NarrowInputs { get; init; } = [];

    /// <summary>The outputs that travel in <see cref="Precision"/> (by output index).</summary>
    public bool[] NarrowOutputs { get; init; } = [];

    /// <summary>The precision input <paramref name="index"/> travels in.</summary>
    public BoundaryPrecision InputPrecision(int index) => NarrowInputs[index] ? Precision : BoundaryPrecision.Float32;

    /// <summary>The precision output <paramref name="index"/> travels in.</summary>
    public BoundaryPrecision OutputPrecision(int index) => NarrowOutputs[index] ? Precision : BoundaryPrecision.Float32;

    /// <summary>Byte size of output <paramref name="index"/> in the current execution.</summary>
    public int OutputBytes(int index) => checked((int)CompiledModel.ByteCount(Graph.Outputs[OutputNames[index]], Parameters));

    public void Dispose() => Model.Dispose();
}
