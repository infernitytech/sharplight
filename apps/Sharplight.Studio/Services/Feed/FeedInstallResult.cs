using Sharplight.Hub;

namespace Sharplight.Studio.Services.Feed;

/// <summary>What a feed install stored.</summary>
/// <param name="Installed">The models stored by this install (the requested package and the dependencies that were missing).</param>
/// <param name="Requested">The record of the requested package (stored now, or already there at this version).</param>
internal sealed record FeedInstallResult(IReadOnlyList<ModelManifest> Installed, ModelManifest Requested);
