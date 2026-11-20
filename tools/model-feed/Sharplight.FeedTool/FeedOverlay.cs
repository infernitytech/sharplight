namespace Sharplight.FeedTool;

/// <summary>
/// Feed-only data that is not part of the verified catalog (<c>tools/model-feed/feed-overlay.json</c>): dependencies between packages,
/// summaries, minimum server versions and listing state, by model (catalog entry id, shared by every variant package) and by package id.
/// </summary>
internal sealed record FeedOverlay
{
    /// <summary>Schema version (1).</summary>
    public int SchemaVersion { get; init; } = 1;

    /// <summary>Overrides by package id.</summary>
    public IReadOnlyDictionary<string, FeedOverlayPackage> Packages { get; init; } = new Dictionary<string, FeedOverlayPackage>();

    /// <summary>Defaults for every variant package of a model, by catalog entry id (a package entry overrides them field by field).</summary>
    public IReadOnlyDictionary<string, FeedOverlayPackage> Models { get; init; } = new Dictionary<string, FeedOverlayPackage>();

    /// <summary>
    /// The overlay of one package: its own entry over its model's (summary, server and languages when the package sets them, dependencies of
    /// both, listed only when both are), or <see langword="null"/> when neither exists.
    /// </summary>
    /// <param name="entryId">Catalog entry id (the model group).</param>
    /// <param name="packageId">Package id.</param>
    /// <returns>The merged overlay.</returns>
    public FeedOverlayPackage? For(string entryId, string packageId)
    {
        var model = Models.GetValueOrDefault(entryId);
        var package = Packages.GetValueOrDefault(packageId);
        if (model is null || package is null)
        {
            return package ?? model;
        }

        return new FeedOverlayPackage
        {
            Summary = package.Summary ?? model.Summary,
            Server = package.Server ?? model.Server,
            Languages = package.Languages.Count > 0 ? package.Languages : model.Languages,
            Dependencies = [.. model.Dependencies.Where(dependency => package.Dependencies.All(own => own.Id != dependency.Id)), .. package.Dependencies],
            Listed = model.Listed && package.Listed,
        };
    }
}
