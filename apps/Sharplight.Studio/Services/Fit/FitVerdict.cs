namespace Sharplight.Studio.Services.Fit;

/// <summary>Whether a model fits the memory of the devices chosen for it, from best to worst.</summary>
internal enum FitVerdict
{
    /// <summary>It fits in the free memory, with the reserve to spare.</summary>
    Fits,

    /// <summary>It fits once other resident models are unloaded.</summary>
    FitsAfterUnloading,

    /// <summary>It fits only with part of it running on the CPU (slower).</summary>
    NeedsOffload,

    /// <summary>It does not fit, not even with CPU offload into system memory.</summary>
    DoesNotFit,
}
