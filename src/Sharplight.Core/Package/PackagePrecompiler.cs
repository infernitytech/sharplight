using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>
/// Precompiles a package through the two-stage contract of the execution providers and stores the result in the package
/// (<see cref="ModelPackage.AddArtifacts"/>: appended in place, crash-safe, without rewriting the weights): for each device, the
/// portable stage (<see cref="IExecutionProvider.CompilePortableAsync"/>, SPIR-V for Vulkan) generates the model's code for the device's
/// profile, and the device-specific stage (<see cref="IExecutionProvider.CompileDeviceSpecificAsync"/>, pipeline binaries for Vulkan) has
/// its driver compile that code. Devices with the same profile share one portable set, devices of the same model and driver one device
/// set. Providers without precompilable code (the CPU) return empty sets and are reported as such. Graphs are compiled the way the
/// loaders compile them (see <see cref="PrecompileOptions"/>), so the stored code is what a later load asks for.
/// </summary>
/// <remarks>
/// The package must not be loaded for writing by an engine that might rewrite it; appending is safe for readers that have it open
/// (existing bytes are never modified, the header switches to the new section table last), but hosts that keep models resident should
/// precompile only models that are not (Sharplight Studio does). A failure before the switch leaves the file exactly as it was.
/// </remarks>
public static class PackagePrecompiler
{
    /// <summary>Precompiles the package for every device of every available provider in <paramref name="providers"/>.</summary>
    /// <param name="packagePath">The package to compile and update.</param>
    /// <param name="providers">The execution providers to precompile for; unavailable ones are reported in <see cref="PackagePrecompileReport.Skipped"/>.</param>
    /// <param name="options">How to compile, or <see langword="null"/> for the defaults.</param>
    /// <param name="progress">Receives the stage of each provider and device, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancels between kernels; nothing is written then.</param>
    /// <returns>What was produced and stored.</returns>
    /// <exception cref="MissingSymbolBoundException">A non-LLM graph has a symbolic dimension without a bound.</exception>
    public static async Task<PackagePrecompileReport> PrecompileAsync(string packagePath, IEnumerable<IExecutionProvider> providers,
        PrecompileOptions? options = null, IProgress<PrecompileProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(providers);
        using var session = new PrecompileSession(packagePath, options ?? new PrecompileOptions(), progress, cancellationToken);
        foreach (var provider in providers)
        {
            await session.CompileProviderAsync(provider).ConfigureAwait(false);
        }

        return session.Store();
    }

    /// <summary>Precompiles the package for one device of <paramref name="provider"/> (both stages).</summary>
    /// <param name="packagePath">The package to compile and update.</param>
    /// <param name="provider">The execution provider of the device.</param>
    /// <param name="device">The device.</param>
    /// <param name="options">How to compile, or <see langword="null"/> for the defaults.</param>
    /// <param name="progress">Receives the stages, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancels between kernels; nothing is written then.</param>
    /// <returns>What was produced and stored.</returns>
    /// <exception cref="MissingSymbolBoundException">A non-LLM graph has a symbolic dimension without a bound.</exception>
    public static async Task<PackagePrecompileReport> PrecompileAsync(string packagePath, IExecutionProvider provider, DeviceInfo device,
        PrecompileOptions? options = null, IProgress<PrecompileProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentNullException.ThrowIfNull(device);
        using var session = new PrecompileSession(packagePath, options ?? new PrecompileOptions(), progress, cancellationToken);
        await session.CompileDeviceAsync(provider, device).ConfigureAwait(false);
        return session.Store();
    }

    /// <summary>
    /// Precompiles the package for a device profile without the device: only the portable stage (the device-specific stage needs the
    /// device and its driver).
    /// </summary>
    /// <param name="packagePath">The package to compile and update.</param>
    /// <param name="provider">The provider that generates code for the profile (<see cref="IExecutionProvider.CanCompileFor"/>).</param>
    /// <param name="profile">The device profile.</param>
    /// <param name="options">How to compile, or <see langword="null"/> for the defaults.</param>
    /// <param name="progress">Receives the stages, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancels between kernels; nothing is written then.</param>
    /// <returns>What was produced and stored.</returns>
    /// <exception cref="NotSupportedException">The provider does not compile for the profile.</exception>
    /// <exception cref="MissingSymbolBoundException">A non-LLM graph has a symbolic dimension without a bound.</exception>
    public static async Task<PackagePrecompileReport> PrecompileAsync(string packagePath, IExecutionProvider provider, DeviceProfile profile,
        PrecompileOptions? options = null, IProgress<PrecompileProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentNullException.ThrowIfNull(profile);
        if (!provider.CanCompileFor(profile))
        {
            throw new NotSupportedException($"Provider '{provider.Name}' does not generate code for {profile}.");
        }

        using var session = new PrecompileSession(packagePath, options ?? new PrecompileOptions(), progress, cancellationToken);
        await session.CompileProfileAsync(provider, profile).ConfigureAwait(false);
        return session.Store();
    }
}
