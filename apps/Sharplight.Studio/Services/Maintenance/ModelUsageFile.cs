namespace Sharplight.Studio.Services.Maintenance;

/// <summary>The content of <c>model-usage.json</c>: the last load time of every model path.</summary>
internal sealed record ModelUsageFile
{
    /// <summary>Last load time by model path.</summary>
    public Dictionary<string, DateTimeOffset> LastUsed { get; init; } = [];
}
