using Sharplight.Graph;

namespace Sharplight.Vulkan.Tests;

/// <summary>One execution of a compiled graph: symbol bindings, runtime scalars and input data.</summary>
internal sealed record Step(long T, long Past, params (Value V, Array Data)[] Inputs)
{
    /// <summary>Value of the secondary symbol "R" (e.g. number of gathered rows); defaults to T.</summary>
    public long? R { get; init; }
    /// <summary>Further symbol bindings (e.g. the run count "U" of a multi-sequence graph).</summary>
    public IReadOnlyDictionary<string, long>? Symbols { get; init; }
    public Step With(params (Value, Array)[] more) => this with { Inputs = [.. Inputs, .. more] };
}
