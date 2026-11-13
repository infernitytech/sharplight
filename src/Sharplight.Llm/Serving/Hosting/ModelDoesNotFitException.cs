namespace Sharplight.Llm.Serving.Hosting;

/// <summary>The model does not fit the free device memory, even after evicting idle models and offloading.</summary>
public sealed class ModelDoesNotFitException(string name, long needed, long available, long reserve, string detail)
    : ModelHostException(
        $"The model '{name}' needs about {needed / 1048576.0:F0} MiB of device memory but only {Math.Max(0, available) / 1048576.0:F0} MiB are free after keeping {reserve / 1048576.0:F0} MiB in reserve. {detail}",
        "insufficient_memory")
{
    /// <summary>Estimated bytes the model needs on its primary device.</summary>
    public long Needed { get; } = needed;
    /// <summary>Bytes that were available.</summary>
    public long Available { get; } = available;
}
