namespace Sharplight.PackageViewer.ViewModels;

/// <summary>Weights of one stored element type.</summary>
/// <param name="Type">Element type name.</param>
/// <param name="Count">Number of weights.</param>
/// <param name="Bytes">Sum of their sizes.</param>
/// <param name="Percent">Share of all weight bytes, 0 to 100.</param>
public sealed record TypeTotalRow(string Type, int Count, long Bytes, double Percent)
{
    public string Size => ByteSizeFormat.Format(Bytes);
}
