namespace Sharplight.Llm.Serving.Hosting;

/// <summary>Read-only view of a registered model for listings and metrics.</summary>
/// <param name="Name">The registered name.</param>
/// <param name="Kind">What the model does.</param>
/// <param name="State">Where it is in its lifecycle.</param>
/// <param name="Pinned">Never evicted and never unloaded when idle.</param>
/// <param name="LoadAtStartup">Loaded by <see cref="ModelHost.LoadStartupModelsAsync"/>.</param>
/// <param name="ResidentBytes">Device bytes it holds while loaded (0 = unknown or not loaded).</param>
/// <param name="EstimatedBytes">Its estimated footprint from the last load attempt.</param>
/// <param name="ActiveLeases">Requests holding it right now.</param>
/// <param name="LoadedAt">When it finished loading.</param>
/// <param name="LastUsed">When it was last acquired or released.</param>
/// <param name="Error">Why the last load failed.</param>
/// <param name="Description">Free text from the registration.</param>
/// <param name="Devices">
/// Ids of the memory pools (<see cref="IMemoryDomain.Id"/>, normally <c>provider:index</c>) its last load attempt was charged against,
/// primary device first; empty before the first attempt.
/// </param>
public sealed record ModelInfo(
    string Name, ModelKind Kind, ModelState State, bool Pinned, bool LoadAtStartup, long ResidentBytes, long EstimatedBytes, int ActiveLeases,
    DateTimeOffset? LoadedAt, DateTimeOffset? LastUsed, string? Error, string? Description, IReadOnlyList<string>? Devices = null);
