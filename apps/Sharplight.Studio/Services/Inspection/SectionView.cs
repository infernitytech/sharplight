namespace Sharplight.Studio.Services.Inspection;

/// <summary>How the inspector presents one kind of file region.</summary>
/// <param name="Title">Readable name of the kind.</param>
/// <param name="Description">What the region holds.</param>
/// <param name="Tab">The tab that shows its decoded contents.</param>
/// <param name="IsDecoded">False for regions this version cannot decode (sections of a newer format), shown as raw byte ranges only.</param>
internal sealed record SectionView(string Title, string Description, InspectorTab Tab, bool IsDecoded);
