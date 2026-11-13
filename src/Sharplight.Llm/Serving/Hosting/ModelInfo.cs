namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Read-only view of a registered model for listings and metrics.</summary>
public sealed record ModelInfo(
    string Name, ModelKind Kind, ModelState State, bool Pinned, bool LoadAtStartup, long ResidentBytes, long EstimatedBytes, int ActiveLeases,
    DateTimeOffset? LoadedAt, DateTimeOffset? LastUsed, string? Error, string? Description);
