namespace Sharplight.PackageViewer.ViewModels;

/// <summary>One entry of an artifact set.</summary>
public sealed record ArtifactEntryRow(string Key, long Bytes) : IFilterableRow
{
    public string Size => ByteSizeFormat.Format(Bytes);

    public bool Matches(string[] terms) => TextFilter.Matches(terms, Key);
}
