namespace Sharplight.Studio.Services.Library;

/// <summary>The licence notice of a model: its text and the id of the catalog entry it belongs to (the key of "shown once").</summary>
/// <param name="EntryId">The catalog entry's id.</param>
/// <param name="Text">The notice.</param>
/// <param name="AcceptedKeys">The other keys the model's licence may have been accepted under before its download (<c>feed:</c> package, <c>hub:</c> repository).</param>
internal sealed record LicenseNotice(string EntryId, string Text, IReadOnlyList<string>? AcceptedKeys = null);
