namespace Sharplight.ModelFeed.Resolution;

/// <summary>A package the caller wants: an id and an allowed version range (an exact version pins it).</summary>
/// <param name="Id">Package id.</param>
/// <param name="Range">Version range (<see cref="Versioning.VersionRange"/>), an exact version as <c>[1.2.0]</c>, or <see langword="null"/> for the latest.</param>
public sealed record PackageRequest(string Id, string? Range = null);
