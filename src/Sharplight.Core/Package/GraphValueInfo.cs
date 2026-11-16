using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>A graph input, output or node output.</summary>
/// <param name="Name">Value name (the output name for graph outputs).</param>
/// <param name="Type">Element type.</param>
/// <param name="Shape">Shape; may contain symbolic dimensions (<c>T</c>, <c>ctx</c>, <c>batch</c>, ...).</param>
/// <param name="Role">Input role (<see cref="Models.InputRole"/>) for graph inputs, empty otherwise.</param>
public sealed record GraphValueInfo(string Name, DataType Type, Shape Shape, string Role)
{
    /// <inheritdoc/>
    public override string ToString() => Role.Length > 0 ? $"{Name} {Type}{Shape} ({Role})" : $"{Name} {Type}{Shape}";
}
