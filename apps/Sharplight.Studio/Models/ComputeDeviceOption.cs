namespace Sharplight.Studio.Models;

/// <summary>A device models can be loaded on, as offered in the load options.</summary>
/// <param name="Id"><c>provider:index</c>, e.g. <c>vulkan:0</c>, or <see cref="AllGpusId"/>.</param>
/// <param name="Name">Device name.</param>
/// <param name="Provider">Execution provider name.</param>
/// <param name="IsGpu">A GPU (else the CPU).</param>
/// <param name="TotalMemory">Device memory in bytes (system memory for the CPU).</param>
/// <param name="Features">Detected device features (provider-neutral names).</param>
/// <param name="GpuKey">Identity of the physical GPU (the same for the entries several providers list for it), or null (the CPU, the split choice).</param>
/// <param name="Variant">The provider's title (<c>Vulkan</c>, <c>CUDA</c>) when the GPU is listed by more than one provider, else null.</param>
internal sealed record ComputeDeviceOption(string Id, string Name, string Provider, bool IsGpu, long TotalMemory, IReadOnlyList<string> Features,
    string? GpuKey = null, string? Variant = null)
{
    /// <summary>The id of the choice that splits a model's layers over every GPU.</summary>
    public const string AllGpusId = "gpu:all";

    /// <summary>Name and memory for pickers.</summary>
    public string Display => Variant is null ? $"{Name} · {ByteFormat.Format(TotalMemory)}" : $"{Name} ({Variant}) · {ByteFormat.Format(TotalMemory)}";

    /// <summary>The choice that splits a model over every GPU (<see cref="AllGpusId"/>).</summary>
    public bool IsSplit => Id == AllGpusId;

    /// <summary>The choice that splits a model over <paramref name="gpus"/>: their memory added up, the features they all have.</summary>
    /// <param name="gpus">The GPUs (at least two).</param>
    public static ComputeDeviceOption AllGpus(IReadOnlyList<ComputeDeviceOption> gpus)
    {
        IEnumerable<string> features = gpus.Count == 0 ? [] : gpus[0].Features;
        foreach (var gpu in gpus.Skip(1))
        {
            features = features.Intersect(gpu.Features, StringComparer.Ordinal);
        }

        return new ComputeDeviceOption(AllGpusId, $"All GPUs ({gpus.Count}, split)", "", true, gpus.Sum(gpu => gpu.TotalMemory), [.. features]);
    }
}
