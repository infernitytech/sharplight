namespace Sharplight.Runtime;

/// <summary>Class of a compute device.</summary>
public enum DeviceKind
{
    /// <summary>A discrete GPU with its own device-local memory.</summary>
    DiscreteGpu,

    /// <summary>A GPU integrated into the CPU package that shares system memory.</summary>
    IntegratedGpu,

    /// <summary>A virtualized or software-emulated GPU (for example llvmpipe).</summary>
    VirtualGpu,

    /// <summary>The host CPU.</summary>
    Cpu,

    /// <summary>Any other kind of accelerator.</summary>
    Other,
}
