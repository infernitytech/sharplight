using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>One entry of the weight index.</summary>
public sealed record WeightRow(string Name, string Type, string Shape, long Offset, long Bytes) : IFilterableRow
{
    public string Size => ByteSizeFormat.Format(Bytes);

    public static WeightRow From(PackageWeightInfo weight) => new(weight.Name, weight.Type.Name, weight.Shape.ToString(), weight.Offset, weight.ByteSize);

    public bool Matches(string[] terms) => TextFilter.Matches(terms, Name, Type, Shape);
}
