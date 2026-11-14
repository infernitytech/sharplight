namespace Sharplight.Studio.Models;

/// <summary>A device models can be loaded on, as offered in the load options.</summary>
/// <param name="Id"><c>provider:index</c>, e.g. <c>vulkan:0</c>.</param>
/// <param name="Name">Device name.</param>
/// <param name="Provider">Execution provider name.</param>
/// <param name="IsGpu">A GPU (else the CPU).</param>
/// <param name="TotalMemory">Device memory in bytes (system memory for the CPU).</param>
/// <param name="Features">Detected device features (provider-neutral names).</param>
internal sealed record ComputeDeviceOption(string Id, string Name, string Provider, bool IsGpu, long TotalMemory, IReadOnlyList<string> Features)
{
    /// <summary>Name and memory for pickers.</summary>
    public string Display => $"{Name} · {ByteFormat.Format(TotalMemory)}";
}
