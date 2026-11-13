namespace Sharplight.PackageViewer;

/// <summary>A table row that the filter box of its table can match.</summary>
internal interface IFilterableRow
{
    /// <summary>Whether every term occurs in one of the row's fields (see <see cref="TextFilter.Matches"/>).</summary>
    public bool Matches(string[] terms);
}
