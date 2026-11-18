namespace Sharplight.Studio.Services.FirstRun;

/// <summary>How one starter role is filled from the feed: which packages qualify and which families come first.</summary>
internal sealed record StarterRolePreference
{
    /// <summary>The role.</summary>
    public StarterRole Role { get; init; }

    /// <summary>The model kind the feed lists the packages under (<c>text-generation</c>, <c>speech-recognition</c>, <c>embeddings</c>).</summary>
    public string Kind { get; init; } = "";

    /// <summary>A task the package must list (<c>chat</c>, <c>embedding</c>), or null for any.</summary>
    public string? Task { get; init; }

    /// <summary>
    /// Text fragments ranked best first. A package ranks by the first fragment found in its family, name, architecture or id (case-insensitive);
    /// packages that match none rank last.
    /// </summary>
    public IReadOnlyList<string> Prefer { get; init; } = [];

    /// <summary>The role is ticked in the welcome flow.</summary>
    public bool Preselected { get; init; }

    /// <summary>Packages whose download (with required dependencies) is larger are not offered, in MiB; 0 for no limit.</summary>
    public long MaxDownloadMiB { get; init; }
}
