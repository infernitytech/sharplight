namespace Sharplight.Samples;

/// <summary>Device and sizing choices shared by the LLM samples, as read from <see cref="RuntimeOptionSet"/>.</summary>
/// <param name="UseCpu">Run on the CPU provider instead of a GPU.</param>
/// <param name="DeviceIndex">Index of the device within the selected provider.</param>
/// <param name="ContextLength">Context length in tokens (per sequence).</param>
/// <param name="BatchTokens">Maximum rows per forward pass.</param>
/// <param name="DisabledFeatures">Device features that must not be used (see <c>DeviceFeatures</c>).</param>
/// <param name="CpuOffload">Let layers that do not fit on the device and unsupported ops run on the CPU.</param>
public sealed record RuntimeSettings(
    bool UseCpu,
    int DeviceIndex,
    int ContextLength,
    int BatchTokens,
    IReadOnlyList<string> DisabledFeatures,
    bool CpuOffload);
