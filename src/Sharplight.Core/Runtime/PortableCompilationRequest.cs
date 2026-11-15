namespace Sharplight.Runtime;

/// <summary>What <see cref="IExecutionProvider.CompilePortableAsync"/> compiles, and for which device profile.</summary>
public sealed class PortableCompilationRequest
{
    /// <summary>Creates a request.</summary>
    /// <param name="target">The device profile to generate code for (<see cref="IComputeDevice.CreateProfile"/> of an opened device, a
    /// profile file, ...); <see langword="null"/> for devices without one (providers without a portable stage return an empty set).</param>
    /// <param name="workload">Compiles the model's graphs on the device it is given.</param>
    public PortableCompilationRequest(DeviceProfile? target, CompilationWorkload workload)
    {
        ArgumentNullException.ThrowIfNull(workload);
        Target = target;
        Workload = workload;
    }

    /// <summary>The device profile to generate code for, or <see langword="null"/> when the device has none.</summary>
    public DeviceProfile? Target { get; }

    /// <summary>Compiles the model's graphs.</summary>
    public CompilationWorkload Workload { get; }

    /// <summary>
    /// Artifact sets already stored with the model, or <see langword="null"/>: matching code is reused instead of generated again
    /// (<see cref="CodeSet.ReusedEntries"/>).
    /// </summary>
    public ICompilationCacheStore? Stored { get; init; }

    /// <summary>
    /// Options of the compile-only device (disabled features, kernel cache directory, provider settings), or <see langword="null"/> for
    /// the defaults. Disabled features are removed from the generated code as when a device is opened with them.
    /// </summary>
    public DeviceOptions? DeviceOptions { get; init; }
}
