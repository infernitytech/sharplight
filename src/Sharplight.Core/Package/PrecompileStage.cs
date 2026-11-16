namespace Sharplight.Package;

/// <summary>What <see cref="PackagePrecompiler"/> is doing for a provider and device (see <see cref="PrecompileProgress"/>).</summary>
public enum PrecompileStage
{
    /// <summary>Opening the device and reading its profile.</summary>
    Opening,

    /// <summary>Stage one: generating portable code for the device's profile (<see cref="Runtime.IExecutionProvider.CompilePortableAsync"/>).</summary>
    Portable,

    /// <summary>Stage two: the driver compiles the portable code for this device (<see cref="Runtime.IExecutionProvider.CompileDeviceSpecificAsync"/>).</summary>
    DeviceSpecific,

    /// <summary>The device is done (or shares the code of an identical device compiled before).</summary>
    Completed,

    /// <summary>The provider or device was skipped (unavailable, nothing to precompile, failed).</summary>
    Skipped,

    /// <summary>Writing the new artifact sets into the package.</summary>
    Writing,
}
