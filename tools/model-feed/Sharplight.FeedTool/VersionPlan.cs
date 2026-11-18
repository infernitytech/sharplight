using Sharplight.ModelFeed;

namespace Sharplight.FeedTool;

/// <summary>The version of a build and, for a manifest-only version, the published manifest whose package file it reuses.</summary>
/// <param name="Version">The version.</param>
/// <param name="ReusesPayloadOf">The published manifest holding the identical package file, or <see langword="null"/> for a new package file.</param>
internal sealed record VersionPlan(string Version, PackageManifest? ReusesPayloadOf);
