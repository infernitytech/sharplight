namespace Sharplight.Runtime;

/// <summary>
/// An execution provider that can generate device code for a <see cref="DeviceProfile"/> without the device being present
/// (offline precompilation of a model's code-tier artifacts).
/// </summary>
public interface IOfflineCompilationProvider
{
    /// <summary>Whether <paramref name="profile"/> describes a device of this provider whose code target it can generate.</summary>
    public bool CanCompileFor(DeviceProfile profile);

    /// <summary>
    /// Opens a compile-only device for <paramref name="profile"/>: <see cref="IComputeDevice.Compile"/> lowers segments and
    /// generates their device code exactly as an opened device with that profile would (the same code-tier
    /// <see cref="IComputeDevice.ArtifactTargets"/> descriptor and cache keys), but allocates nothing and the returned segments
    /// cannot run. Its memory figures come from <see cref="DeviceProfile.TotalMemory"/> (plenty when unknown) so placement works
    /// as usual. Needs no driver. The caller disposes it.
    /// </summary>
    /// <param name="profile">The device profile.</param>
    /// <param name="options">Device options (disabled features, kernel cache directory, settings), or <see langword="null"/> for the defaults.</param>
    /// <exception cref="NotSupportedException"><see cref="CanCompileFor"/> is false for the profile.</exception>
    public IComputeDevice OpenProfile(DeviceProfile profile, DeviceOptions? options = null);
}
