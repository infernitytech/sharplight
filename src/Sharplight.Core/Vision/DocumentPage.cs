namespace Sharplight.Vision;

/// <summary>The structured result of <see cref="DocumentReader.ReadAsync"/>.</summary>
/// <param name="Regions">The layout regions in reading order, with their text.</param>
public sealed record DocumentPage(IReadOnlyList<DocumentRegion> Regions)
{
    /// <summary>The text of the page: the texts of the regions that were read, in reading order, separated by blank lines.</summary>
    public string Text => string.Join("\n\n", Regions.Where(r => !string.IsNullOrWhiteSpace(r.Text)).Select(r => r.Text));
}
