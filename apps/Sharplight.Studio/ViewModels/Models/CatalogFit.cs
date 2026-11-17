namespace Sharplight.Studio.ViewModels.Models;

/// <summary>Where a catalog variant's estimated memory need fits on this machine.</summary>
internal enum CatalogFit
{
    /// <summary>No memory figure for the variant or the machine.</summary>
    Unknown,

    /// <summary>Fits the GPU memory.</summary>
    Gpu,

    /// <summary>Too large for the GPUs but fits the system memory.</summary>
    Cpu,

    /// <summary>Larger than the GPU and the system memory.</summary>
    TooLarge,
}
