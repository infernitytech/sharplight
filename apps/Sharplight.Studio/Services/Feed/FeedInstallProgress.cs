namespace Sharplight.Studio.Services.Feed;

/// <summary>Progress of a feed install across the package and its dependencies.</summary>
/// <param name="Phase">The step.</param>
/// <param name="PackageId">The package being handled, or null while resolving.</param>
/// <param name="Received">Bytes present over all packages to install.</param>
/// <param name="Total">Bytes of all packages to install (0 until known).</param>
internal sealed record FeedInstallProgress(FeedInstallPhase Phase, string? PackageId, long Received, long Total);
