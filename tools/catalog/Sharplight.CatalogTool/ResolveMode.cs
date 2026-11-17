namespace Sharplight.CatalogTool;

/// <summary>Which file sets <see cref="CatalogResolver"/> resolves.</summary>
internal enum ResolveMode
{
    /// <summary>Only sets without a pinned commit or files (new entries).</summary>
    Missing,

    /// <summary>Every set again at its pinned commit (after changing patterns or tags).</summary>
    Refresh,

    /// <summary>Every set at the repository's current <c>main</c>: pins newer commits (verify them first).</summary>
    Update,
}
