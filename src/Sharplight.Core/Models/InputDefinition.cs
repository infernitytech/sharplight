namespace Sharplight.Models;

public sealed class InputDefinition
{
    public required string Name { get; init; }
    public required InputRole Role { get; init; }
    public override string ToString() => $"{Name} ({Role})";
}
