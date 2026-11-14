using System.Globalization;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>One entry of the weight index.</summary>
/// <param name="Name">Weight name.</param>
/// <param name="Type">Stored element type (quantization format).</param>
/// <param name="Shape">Shape.</param>
/// <param name="Elements">Number of elements.</param>
/// <param name="Offset">Absolute file offset of the bytes.</param>
/// <param name="Bytes">Stored size.</param>
internal sealed record WeightRow(string Name, string Type, string Shape, long Elements, long Offset, long Bytes) : IFilterableRow
{
    /// <summary>The size for display.</summary>
    public string Size => ByteFormat.Format(Bytes);

    /// <summary>The offset for display.</summary>
    public string OffsetText => ByteFormat.Number(Offset);

    /// <summary>Stored bits per element, e.g. <c>4.50</c>.</summary>
    public double BitsPerElement => Elements > 0 ? 8.0 * Bytes / Elements : 0;

    /// <summary>Bits per element for display.</summary>
    public string Bits => BitsPerElement.ToString("0.##", CultureInfo.InvariantCulture);

    /// <summary>The row of an index entry.</summary>
    public static WeightRow From(PackageWeightInfo weight) =>
        new(weight.Name, weight.Type.Name, weight.Shape.ToString(), weight.Shape.IsStatic ? weight.Shape.Elements() : 0, weight.Offset, weight.ByteSize);

    /// <inheritdoc/>
    public bool Matches(string[] terms) => TextFilter.Matches(terms, Name, Type, Shape);
}
