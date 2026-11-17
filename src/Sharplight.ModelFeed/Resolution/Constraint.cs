using Sharplight.ModelFeed.Versioning;

namespace Sharplight.ModelFeed.Resolution;

/// <summary>A range on a package id and who put it there.</summary>
/// <param name="Range">The range.</param>
/// <param name="From">The dependent package id, or <see langword="null"/> for a request.</param>
/// <param name="Role">The dependency role.</param>
internal sealed record Constraint(VersionRange Range, string? From, string? Role);
