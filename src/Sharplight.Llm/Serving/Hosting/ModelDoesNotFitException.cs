namespace Sharplight.Llm.Serving.Hosting;

public sealed class ModelDoesNotFitException(string name, long needed, long available, long reserve, string detail)
    : ModelHostException(
        $"The model '{name}' needs about {needed / 1048576.0:F0} MiB of device memory but only {Math.Max(0, available) / 1048576.0:F0} MiB are free after keeping {reserve / 1048576.0:F0} MiB in reserve. {detail}",
        "insufficient_memory")
{
    public long Needed { get; } = needed;
    public long Available { get; } = available;
}
