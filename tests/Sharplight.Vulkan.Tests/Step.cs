using Sharplight.Graph;

namespace Sharplight.Vulkan.Tests;

/// <summary>One execution of a compiled graph: symbol bindings, runtime scalars and input data.</summary>
internal sealed record Step(long T, long Past, params (Value V, Array Data)[] Inputs)
{
    /// <summary>Value of the secondary symbol "R" (e.g. number of gathered rows); defaults to T.</summary>
    public long? R { get; init; }
    public Step With(params (Value, Array)[] more) => this with { Inputs = [.. Inputs, .. more] };
}
