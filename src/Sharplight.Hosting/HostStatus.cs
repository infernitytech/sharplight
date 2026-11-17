using System.Text.Json;

namespace Sharplight.Hosting;

/// <summary>A point-in-time view of a host: devices, models and load (<see cref="IModelHostManager.GetStatus"/>). Serializable to JSON.</summary>
public sealed record HostStatus
{
    /// <summary>When the snapshot was taken.</summary>
    public required DateTimeOffset Timestamp { get; init; }

    /// <summary>Memory kept free on every device, in bytes.</summary>
    public required long ReserveBytes { get; init; }

    /// <summary>Resident model limit (0 = none).</summary>
    public required int MaxLoadedModels { get; init; }

    /// <summary>Every device of the host (opened ones with live figures) and every memory pool a model was charged against.</summary>
    public required IReadOnlyList<DeviceStatus> Devices { get; init; }

    /// <summary>Every registered model, in registration order.</summary>
    public required IReadOnlyList<HostedModelStatus> Models { get; init; }

    /// <summary>Models currently loaded.</summary>
    public int LoadedModels => Models.Count(model => model.State == Llm.Serving.Hosting.ModelState.Loaded);

    /// <summary>Requests waiting for a slot, over all models.</summary>
    public int QueueDepth => Models.Sum(model => model.QueueDepth);

    /// <summary>Requests being generated, over all models.</summary>
    public int ActiveSequences => Models.Sum(model => model.ActiveSequences);

    /// <summary>Generated tokens per second over the last seconds, over all models.</summary>
    public double TokensPerSecond => Models.Sum(model => model.TokensPerSecond);

    /// <summary>The snapshot as JSON (camelCase names, enums as strings), e.g. for a cluster coordinator.</summary>
    /// <returns>The JSON text.</returns>
    public string ToJson() => JsonSerializer.Serialize(this, HostStatusJsonContext.Default.HostStatus);

    /// <summary>Reads a snapshot written by <see cref="ToJson"/>.</summary>
    /// <param name="json">The JSON text.</param>
    /// <returns>The snapshot.</returns>
    /// <exception cref="JsonException">The text is not a status snapshot.</exception>
    public static HostStatus FromJson(string json) => JsonSerializer.Deserialize(json, HostStatusJsonContext.Default.HostStatus)
        ?? throw new JsonException("The status is empty (null).");
}
