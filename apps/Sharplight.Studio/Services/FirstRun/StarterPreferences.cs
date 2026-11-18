namespace Sharplight.Studio.Services.FirstRun;

/// <summary>
/// The knobs of the starter recommendation, read from the JSON resource built into Studio or from <c>starter-models.json</c> in the settings
/// folder, which replaces it. No model id is named: packages qualify by kind and task and are ranked by family fragments.
/// </summary>
internal sealed record StarterPreferences
{
    /// <summary>The connection speed the download time estimate assumes, in megabits per second.</summary>
    public double AssumedDownloadMbps { get; init; } = 100;

    /// <summary>The share of a GPU's memory a model may need.</summary>
    public double GpuMemoryFraction { get; init; } = 0.8;

    /// <summary>The share of system memory a model may need when it runs on the CPU.</summary>
    public double CpuMemoryFraction { get; init; } = 0.4;

    /// <summary>The most memory a CPU model may need, in MiB.</summary>
    public long CpuMaxMiB { get; init; } = 3072;

    /// <summary>Chat models whose download is larger are not recommended, in MiB.</summary>
    public long MaxDownloadMiB { get; init; } = 20480;

    /// <summary>Disk space that stays free after the downloads, in MiB.</summary>
    public long DiskReserveMiB { get; init; } = 2048;

    /// <summary>Questions offered as the first message of the new chat; the first one is used.</summary>
    public IReadOnlyList<string> ExamplePrompts { get; init; } = [];

    /// <summary>The roles, in the order they are offered.</summary>
    public IReadOnlyList<StarterRolePreference> Roles { get; init; } = [];
}
