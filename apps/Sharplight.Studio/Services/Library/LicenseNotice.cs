namespace Sharplight.Studio.Services.Library;

/// <summary>The licence notice of a model: its text and the id of the catalog entry it belongs to (the key of "shown once").</summary>
/// <param name="EntryId">The catalog entry's id.</param>
/// <param name="Text">The notice.</param>
internal sealed record LicenseNotice(string EntryId, string Text);
