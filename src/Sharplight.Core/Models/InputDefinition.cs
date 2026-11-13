namespace Sharplight.Models;

/// <summary>Name and semantic role of a graph input.</summary>
public sealed class InputDefinition
{
    /// <summary>Input name, unique within its graph.</summary>
    public required string Name { get; init; }
    /// <summary>What the input carries, so generic runtimes know how to feed it.</summary>
    public required InputRole Role { get; init; }
    /// <inheritdoc/>
    public override string ToString() => $"{Name} ({Role})";
}
