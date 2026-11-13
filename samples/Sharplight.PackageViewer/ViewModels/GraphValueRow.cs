using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>A graph input or output.</summary>
public sealed record GraphValueRow(string Name, string Type, string Shape, string Role)
{
    public static GraphValueRow From(GraphValueInfo value) => new(value.Name, value.Type.Name, value.Shape.ToString(), value.Role);
}
