namespace Sharplight.Storage;

/// <summary>What selected the <see cref="StorageMode"/> of a <see cref="StorageLayout"/>.</summary>
public enum StorageModeSource
{
    /// <summary>Nothing was configured: the system folders.</summary>
    Default,

    /// <summary>The portable marker file next to the package (<see cref="StorageLayout.PortableMarkerFileName"/>).</summary>
    PortableMarker,

    /// <summary>The pointer file in the system configuration folder (<see cref="StoragePointer"/>).</summary>
    PointerFile,

    /// <summary>The <c>SHARPLIGHT_HOME</c> environment variable (<see cref="StorageLayout.HomeVariable"/>).</summary>
    EnvironmentVariable,

    /// <summary>A layout built for a given mode (<see cref="StorageLayout.Create"/>), e.g. to preview a change.</summary>
    Explicit,
}
