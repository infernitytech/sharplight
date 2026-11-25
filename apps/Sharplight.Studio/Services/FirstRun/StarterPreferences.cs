namespace Sharplight.Studio.Services.FirstRun;

/// <summary>
/// The knobs of the starter recommendation, read from the JSON resource built into Studio or from <c>starter-models.json</c> in the settings
/// folder, which replaces it. No model id is named: packages qualify by kind and task and are ranked by family fragments.
/// </summary>
internal sealed record StarterPreferences
{
    /// <summary>The connection speed the download time estimate assumes, in megabits per second.</summary>
    public double AssumedDownloadMbps { get; init; } = 100;

    /// <summary>The share of a GPU's free memory a model with its context may need (the rest is left to the driver and the desktop).</summary>
    public double GpuMemoryFraction { get; init; } = 0.9;

    /// <summary>The share of system memory a model may need when it runs on the CPU.</summary>
    public double CpuMemoryFraction { get; init; } = 0.4;

    /// <summary>The most memory a CPU model may need, in MiB.</summary>
    public long CpuMaxMiB { get; init; } = 3072;

    /// <summary>The context length a chat model must have room for next to its weights, in tokens.</summary>
    public int ContextTokens { get; init; } = 4096;

    /// <summary>KV cache headroom added to a chat model's memory when the feed gives no context figure, as a share of its weights.</summary>
    public double KvHeadroomFraction { get; init; } = 0.12;

    /// <summary>The least KV cache headroom added to a chat model's memory, in MiB.</summary>
    public long KvHeadroomMinMiB { get; init; } = 256;

    /// <summary>Chat models whose download is larger are not recommended, in MiB.</summary>
    public long MaxDownloadMiB { get; init; } = 20480;

    /// <summary>Disk space that stays free after the downloads, in MiB.</summary>
    public long DiskReserveMiB { get; init; } = 2048;

    /// <summary>Questions offered as the first message of the new chat; the first one is used.</summary>
    public IReadOnlyList<string> ExamplePrompts { get; init; } = [];

    /// <summary>The roles, in the order they are offered.</summary>
    public IReadOnlyList<StarterRolePreference> Roles { get; init; } = [];
}
