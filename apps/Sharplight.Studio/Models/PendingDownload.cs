namespace Sharplight.Studio.Models;

/// <summary>An unfinished download kept across restarts (its partial files stay in the store).</summary>
/// <param name="Reference"><c>owner/name:tag</c>.</param>
/// <param name="ExpectedBytes">Bytes to download, for the progress bar before the pull reports.</param>
/// <param name="Paused">Paused by the user: restored paused instead of resuming.</param>
/// <param name="Kind">The model kind inferred from the Hub's pipeline tag (the downloaded model has no description to read yet), or null.</param>
internal sealed record PendingDownload(string Reference, long ExpectedBytes, bool Paused, string? Kind = null);
