namespace Sharplight.ModelFeed;

/// <summary>
/// One weight type of a package and the engine paths that compute with it (<c>requirements.kernels</c>): devices with one of
/// <see cref="NativeFeatures"/> run the fast kernels; the others run the generic kernel when <see cref="Fallback"/> is set.
/// </summary>
public sealed record PackageKernel
{
    /// <summary>Engine weight type (<c>q4_k</c>, <c>mxfp4</c>, <c>f8e4m3_b128</c>, <c>bf16</c>, ...).</summary>
    public required string WeightType { get; init; }

    /// <summary>Weight bytes stored in this type.</summary>
    public long Bytes { get; init; }

    /// <summary>Neutral device feature names (Sharplight.Runtime.DeviceFeatures) whose kernels accelerate this type; any one helps.</summary>
    public IReadOnlyList<string> NativeFeatures { get; init; } = [];

    /// <summary>
    /// Whether every provider (CPU, Vulkan, CUDA, Metal) has a generic kernel for this type, so a device without the native features
    /// still runs the package (slower, same memory). When <see langword="false"/>, a device needs one of <see cref="NativeFeatures"/>.
    /// </summary>
    public bool Fallback { get; init; } = true;
}
