namespace Sharplight.Studio.Services.Updates;

/// <summary>A newer version that can be installed.</summary>
/// <param name="Version">The version, e.g. <c>0.2.7</c> or <c>0.3.1-alpha.gab12cd3</c>.</param>
/// <param name="ReleaseNotes">Markdown release notes; null or empty when the release has none.</param>
/// <param name="DownloadSize">Size of the package in bytes (0 when unknown).</param>
/// <param name="IsDowngrade">The version is older than the running one (moving to a channel that is behind).</param>
internal sealed record UpdateOffer(string Version, string? ReleaseNotes, long DownloadSize, bool IsDowngrade = false);
