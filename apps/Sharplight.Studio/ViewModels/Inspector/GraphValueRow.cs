using Sharplight.Package;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A graph input or output, or a node output.</summary>
/// <param name="Name">Value name.</param>
/// <param name="Type">Element type.</param>
/// <param name="Shape">Shape.</param>
/// <param name="Role">Input role, empty for outputs.</param>
internal sealed record GraphValueRow(string Name, string Type, string Shape, string Role)
{
    /// <summary>Type and shape, e.g. <c>f32[T, 1024]</c>.</summary>
    public string TypeShape => Type + Shape;

    /// <summary>The row of a value.</summary>
    public static GraphValueRow From(GraphValueInfo value) => new(value.Name, value.Type.Name, value.Shape.ToString(), value.Role);
}
